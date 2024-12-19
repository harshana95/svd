import argparse
import os

import numpy as np
import scipy
import torch
from einops import einops
from torch.utils.data import Dataset

from dataset.BasicDataset import get_default_preprocess, ImageDataset, get_default_transforms
from dataset.DatasetWrapper import DatasetWrapper

from dataset.SVconv import SVConv_Brute, SVConv_fast, conv_kernel
from utils.dataset_utils import crop_from_center


class ConvolveImageDatasetWrapper(DatasetWrapper):
    @staticmethod
    def convert_key(key, inplace):
        if inplace:
            return key
        return key + "_blur"

    def __init__(self, dataset, valid_keys, blur_op, basis_psf, basis_weights=None, kernel_key=None,
                 inplace=False, metadata=None, metadata_injector=None, injection=0, transform=None, **kwargs):
        super().__init__(dataset, inplace, metadata, metadata_injector, injection, transform, **kwargs)
        self.dataset = dataset
        self.valid_keys = valid_keys
        self.kernel_key = kernel_key
        self.blur_operation = blur_op  # define the operation for GT images with shape (n, c, h, w)
        if kernel_key is not None:  # get kernel from sample
            assert basis_weights is None and basis_psf is None
            self.basis_psf = None
            self.basis_weight = None
        else:
            self.basis_psf = torch.tensor(basis_psf) if type(basis_psf) != torch.Tensor else basis_psf
            self.basis_weight = torch.tensor(basis_weights) if type(basis_weights) != torch.Tensor else basis_weights

    def operator(self, sample, **kwargs):
        for key in self.valid_keys:
            x = sample[key]
            ispatched = len(x.shape) != 3
            if not ispatched:
                x = x[None, :]

            kernels = self.basis_psf
            weights = self.basis_weight
            if kernels is None:
                kernels = sample.get(self.kernel_key)
            new_key = self.convert_key(key, self.inplace)
            sample[new_key], kernels, weights = self.blur_operation(x, kernels, weights)  # x: n, c, H, W
            if not ispatched:
                sample[new_key] = sample[new_key][0]
            assert torch.sum(torch.isnan(sample[new_key])) == 0, "Nan value in image"

        return sample


# class Onthefly_Dataset(Dataset):
#     def __init__(self, dataset, getitem_f, kernels, transform=None, preprocess=None, method='scattering'):
#         self.dataset = dataset
#         self.getitem_f = getitem_f
#         self.transform = transform
#         self.preprocess = preprocess
#         self.kernels = kernels
#         self.method = method
#         self.blur_operation = None  # define the operation for GT images with shape (n, c, h, w)
#         self.current_ker = None  # store the currently using kernel.
#         # (this is updated in blur operation where kernels are generated)
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         gt = self.getitem_f(self.dataset, idx)
#         #
#         # if gt.shape[-1] == 3 or gt.shape[-1] == 1:
#         #     gt = einops.rearrange(gt, 'h w c -> c h w')
#         # if len(gt.shape) == 3:
#         #     if gt.shape[0] == 3 or gt.shape[0] == 1:
#         #         gt = np.expand_dims(gt, 0)  # shape = (3/1, h, w) -> (1, 3/1, h, w)
#         #     else:
#         #         gt = np.expand_dims(gt, 1)  # shape = (n, h, w) -> (n, 1, h, w)
#         if self.preprocess:
#             gt = self.preprocess(gt)
#         blurred = self.blur_operation(gt[None, :])[0]  # blur operation takes n, c, H, W
#         sample = {'im_gt': gt, 'meas': blurred}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         if self.kernels is None:
#             sample['ker'] = self.current_ker
#         assert torch.sum(torch.isnan(sample['meas'])) == 0, "Nan value in image"
#         return sample
#
#
# class SVB_onthefly_Dataset(Onthefly_Dataset):
#     """
#     Get spatially varying blurred image and GT image dataset using the pixel-wise kernels
#     spatially varying kernels of shape (channels, height, width, kernel_size, kernel_size)
#
#     """
#
#     def __init__(self, dataset, getitem_f, kernels, transform=None, preprocess=None, method='scattering'):
#         super().__init__(dataset, getitem_f, kernels, transform, preprocess, method)
#         self.blur_operation = SVConv_Brute((kernels.shape[1], kernels.shape[2]), kernels, method=method)
#
#
# class SVB_weighted_onthefly_Dataset(Onthefly_Dataset):
#     """
#     Get spatially varying blurred image and GT image dataset using the basis kernels and weights
#     spatially varying kernels of shape (inc/groups=1, outc, n_psfs, kernel_size, kernel_size)
#     """
#
#     def __init__(self, dataset, getitem_f, kernels, weights, shape, blur_mode, transform=None, preprocess=None,
#                  method='gathering'):
#         super().__init__(dataset, getitem_f, kernels, transform, preprocess, method)
#         if method == 'gathering' and blur_mode == 0:
#             print("Weight first in gathering might be incorrect! Check!!!")
#         if method == 'scattering' and blur_mode == 1:
#             print("Blur first in scattering might be incorrect! Check!!!")
#         # self.blur_operation = SVConv_Brute(shape, kernels, weights=weights, blur_mode=blur_mode, method=method)
#         self.blur_operation = SVConv_fast(shape, kernels, weights=weights, blur_mode=blur_mode, method=method)
#
#
# class SiVB_onthefly_Dataset(Onthefly_Dataset):
#     """
#         Get spatially invariant blurred image and GT image dataset
#         If you provide the kernel, that kernel will be used for all images.
#         If you do not prove the kernel, random kernel will be generated from the psf_generator.
#             Please provide necessary kwargs for your PSF type needed.
#             Refer SV_PSF.py to get an idea about kwargs that you need.
#             psf_size also needed in this case
#     """
#
#     def blur_with_onthefly_kernels(self, x):
#         ker = self.psf_generator.generate_psf(rand_index_func(0, self.image_shape[0]),
#                                               rand_index_func(0, self.image_shape[1]),
#                                               0, **self.kwargs)
#         ker = ker[None, None, :, :]
#         ker = torch.from_numpy(ker).to(x.device)
#         self.current_ker = ker
#
#         # _x = F.pad(x, self._pad, mode='reflect')
#         # out = F.conv2d(_x, ker, stride=1, padding='valid', groups=1)
#         # return out
#         out, _ = conv_kernel(ker[0, 0], x[:, 0], self._pad)
#         return out[:, None, :]
#
#     def __init__(self, dataset, getitem_f, kernel, image_shape, transform=None, preprocess=None, method='scattering',
#                  psf_size=-1, **kwargs):
#
#         if method == 'scattering':
#             print("Given blur method is scattering. But using gathering because for invariant case it doesn't matter")
#             method = 'gathering'
#         super().__init__(dataset, getitem_f, kernel, transform, preprocess, method)
#         self.image_shape = image_shape
#
#         if kernel is None:
#             if psf_size < 0:
#                 raise Exception("Please provide psf size")
#             self.psf_generator = Sim_PSF(image_shape, psf_size=psf_size, normalize=True)
#             self.kwargs = kwargs
#             self.wk, self.hk = psf_size, psf_size
#             self._pad = (self.wk // 2, self.wk // 2 + (self.wk % 2 - 1), self.hk // 2, self.hk // 2 + (self.hk % 2 - 1))
#             self.blur_operation = self.blur_with_onthefly_kernels
#         else:
#             # we want weights with shape (inc, n_psfs, kernel_size, kernel_size)
#             weights = np.ones((1, 1, *image_shape), dtype=float)
#             # we want kernels with shape (inc/groups=1, outc, n_psfs, kernel_size, kernel_size)
#             if len(kernel.shape) == 2:
#                 kernel = kernel[None, None, None, :, :]
#             else:
#                 raise Exception("Please prove a kernel with 2 dimensions only")
#             self.kernels = kernel
#             self.blur_operation = SVConv_Brute(image_shape, kernel, weights=weights, blur_mode=0, method=method,
#                                                mean_ker=False)


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt

    args = argparse.ArgumentParser().parse_args()
    args.image_size_h = 200
    args.image_size_w = 200
    args.psf_size = 32

    args.n_colors = 1
    args.mu_sigma = 0
    args.dc_sigma = 0.00
    args.max_sigma = 0.00
    args.peak_dc_poisson = -1
    args.peak_poisson = 0

    path = "F:\\Datasets\\Blur\\flickr30k_images_PCA_file_z10000_cartesian_x1"
    # path = "F:\\Datasets\\Blur\\flickr30k_images_PCA_zernike_x1"

    files = glob.glob(os.path.join(path, 'val_ground_truth', '*'))

    psfs = scipy.io.loadmat(os.path.join(path, 'PSFs.mat'))
    psfs = psfs['psfs']  # psfs shape (outc, inc, n, h, w)
    weights = scipy.io.loadmat(os.path.join(path, 'Weights.mat'))
    weights = weights['weights']  # weights shape (outc, inc, n, H, W)
    metadata = scipy.io.loadmat(os.path.join(path, 'Metadata.mat'))
    metadata = metadata['metadata']  # metadata shape (outc, inc, H, W, d)
    if psfs.shape[0] == 1:
        psfs = einops.repeat(psfs, '1 outc n h w -> 3 outc n h w').astype('float32')

    blur_op = SVConv_fast(weighted=True, blur_mode=1, method='gathering')

    ds = ImageDataset(gt=files, transform=get_default_preprocess(args, test=True))


    def injector(sample, metadata, image_key='gt', weight_key='weights'):
        sample[weight_key] = metadata['weights']
        sample['psfs'] = metadata['psfs']
        sample['metadata'] = metadata['psf_metadata']
        sample = crop_from_center(sample, ref_key=image_key, crop_keys=[weight_key, 'metadata'])
        return sample


    print("==================================================================== blur only")
    conv_ds = ConvolveImageDatasetWrapper(ds, valid_keys=('gt',), blur_op=blur_op,
                                          basis_psf=psfs, basis_weights=weights,
                                          transform=get_default_transforms(args, ('gt_blur',)),
                                          # metadata={'psfs': psfs, 'weights': weights, 'psf_metadata': metadata},
                                          # metadata_injector=injector,
                                          # injection=-1,
                                          # image_key='gt',
                                          # weight_key='weights'
                                          )
    dl = torch.utils.data.DataLoader(conv_ds, batch_size=1)
    for s in dl:
        for k in s.keys():
            print(k, s[k].shape)
        for i in range(len(s['gt'])):
            plt.imshow(einops.rearrange(s['gt'][i], 'c h w -> h w c'))
            plt.colorbar()
            plt.show()
            plt.imshow(einops.rearrange(s['gt_blur'][i], 'c h w -> h w c'))
            plt.colorbar()
            plt.show()
        break
    # print("==================================================================== blur and then patch")
    # patch_ds = PatchedImageDatasetWrapper(conv_ds, patch_size=128, stride=120,
    #                                       valid_keys=('gt_blur', 'weights'), )
    # dl = torch.utils.data.DataLoader(patch_ds, batch_size=1)
    #
    # for s in dl:
    #     for k in s.keys():
    #         print(k, s[k].shape)
    #     merged = PatchedImageDatasetWrapper.merge_patches(s['gt_blur_patch'], s['gt_blur_patch_pos'])
    #     for i in range(len(merged)):
    #         plt.imshow(einops.rearrange(merged[i], 'c h w -> h w c'))
    #         plt.colorbar()
    #         plt.show()
    #     break
    #
    # print("==================================================================== patch and then blur")
    # patch_ds = PatchedImageDatasetWrapper(ds, patch_size=128, stride=120, valid_keys=('gt', 'weights'),
    #                                       metadata={'psfs': psfs, 'weights': weights},
    #                                       metadata_injector=PatchedImageDatasetWrapper.global_processor,
    #                                       injection=-1)
    # conv_ds = ConvolveImageDatasetWrapper(patch_ds, valid_keys=('gt_patch',),
    #                                       kernel_key='psfs', weight_key='weights_patch',
    #                                       blur_op=blur_op, return_kernels=True)
    # dl = torch.utils.data.DataLoader(conv_ds, batch_size=1)
    # for s in dl:
    #     for k in s.keys():
    #         print(k, s[k].shape)
    #     merged = PatchedImageDatasetWrapper.merge_patches(s['gt_patch_blur'], s['gt_patch_pos'])
    #     for i in range(len(merged)):
    #         plt.imshow(einops.rearrange(merged[i], 'c h w -> h w c'))
    #         plt.colorbar()
    #         plt.show()
    #     break
