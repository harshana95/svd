import matplotlib
matplotlib.use("TkAgg")

import argparse
import glob
import os
import shutil
import cv2
import torch
from comet_ml.utils import shape
from datasets import load_dataset
from markdown_it.rules_inline import image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset.BasicDataset import ImageDataset
from dataset.default_datasets import get_SVB_Dataset
from psf.svpsf import PSFSimulator
from utils.dataset_utils import crop_arr, divisible_by, add_perlin_noise

from utils.utils import get_basename, generate_folder, parse_args, copy_folder_contents
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.pca_utils import get_pca_components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--psf_data_path', default="/scratch/gilbreth/wweligam/dataset/gaussian_synthetic")
    parser.add_argument('--psf_size', type=int, default=32)

    parser.add_argument('--psf_decomp', type=str, default='PCA',
                        choices=["PCA", "sections", "Zernike", "Uniform"])
    parser.add_argument('--interpolation_samples', type=int, default=128)

    parser.add_argument('--pca_n', type=int, default=50)
    parser.add_argument('--zernike_n', type=int, default=15)

    parser.add_argument("--save_dir", default="/scratch/gilbreth/wweligam/dataset/test_image_dataset")
    parser.add_argument('--gt_dataset_path', default='/scratch/gilbreth/wweligam/hybrid_Flickr2k_gt_v2/')
    parser.add_argument('--gt_dataset_file_pattern', default='*')
    # parser.add_argument('--cap_dataset_path', default='', help='captured image data path. Changes the mode of dataset from synthetic to real if provided')

    parser.add_argument('--gt_crop', action='store', type=int, nargs=4)
    # parser.add_argument('--cap_crop', action='store', type=int, nargs=4, default=None)

    parser.add_argument('--pixel_rescale', type=float, default=1.0, help='distance between pixels rescale factor')
    parser.add_argument('--translate_x', type=float, default=0.0, help='translate object along horizontal axis (in mm)')
    parser.add_argument('--translate_y', type=float, default=0.0, help='translate object along vertical axis (in mm)')

    parser.add_argument('--translate_by', default='[]', type=str, help='translate object. format [x1,y1|x2,y2|...]')
    parser.add_argument('--crop_before_resize', action='store', type=int, nargs=4, help='crop image before resizing')
    parser.add_argument('--crop_edge', action='store', type=int, nargs=4, default=None, help='Crop padding after blurring. format ys,ye,xs,xe')
    parser.add_argument('--rotate_by', default=None, type=float, help='rotate original image (in degrees)')

    parser.add_argument('--sensor_size_rescale', type=float, default=1.0)
    parser.add_argument('--psf_size_rescale', type=float, default=1.0)

    parser.add_argument('--n_psf_search', type=int, default=3)  # 3 is a good number
    parser.add_argument('--eps_search', type=float, default=60.0)

    parser.add_argument('--conv_method', type=str, default='scattering', choices=['scattering', 'gathering'])
    parser.add_argument('--convert2gathering', action='store_true')

    parser.add_argument('--data_percent', type=float, default=100, help='data % to use')
    parser.add_argument('--val_perc', type=float, default=1.0, help='percentage used for validation')

    parser.add_argument('--normalize_psf', action='store_true')
    parser.add_argument('--psf_type', type=str, default='file')

    parser.add_argument('--normalize_image', action='store_true')
    parser.add_argument('--normalize_image_scale', type=float, default=1.0)

    parser.add_argument('--add_perlin_noise', action='store_true')
    parser.add_argument('--correct_distortion', action='store_true')

    args = parse_args(parser)
    args.n_colors = 3
    args.divisible = 1
    args.patch_size = 0
    args.max_sigma = 0
    args.dc_sigma = 0
    args.mu_sigma = 0
    args.peak_dc_poisson = 0
    args.peak_poisson = -1

    translate_by = args.translate_by[1:-1]
    if len(translate_by) > 0:
        translate_by = [list(map(int, s.strip().split(','))) for s in translate_by.split('|')]
    else:
        translate_by = None

    psf_decomp = args.psf_decomp
    psf_size = args.psf_size
    pixel_dist_x = 1 * args.pixel_rescale
    pixel_dist_y = 1 * args.pixel_rescale
    dataset_name = f"synthetic_{get_basename(args.gt_dataset_path)}_{get_basename(args.psf_data_path)}_{psf_decomp}"

    # initialize save locations
    print(f"Saving to {join(args.save_dir, dataset_name)}", "Dataset name", dataset_name)
    ds_dir = join(args.save_dir, dataset_name)
    gt_dir = join(ds_dir, 'train', 'gt')
    meas_dir = join(ds_dir, 'train', 'blur')
    vgt_dir = join(ds_dir, 'val', 'gt')
    vmeas_dir = join(ds_dir, 'val', 'blur')
    md_dir = join(ds_dir, 'metadata')

    generate_folder(gt_dir)
    generate_folder(meas_dir)
    generate_folder(vgt_dir)
    generate_folder(vmeas_dir)
    generate_folder(md_dir)

    # save args to log
    with open(join(md_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
            f.flush()
        f.flush()

    # load gt image filenames
    _dataset_path = join(args.gt_dataset_path, args.gt_dataset_file_pattern)
    dataset = np.array(sorted(glob.glob(_dataset_path)))
    print(f"Number of images in {_dataset_path} = {len(dataset)} using {args.data_percent}%")
    dataset = dataset[:int(len(dataset) * args.data_percent / 100)]

    # load PSFs and simulator
    psf_ds = PSFSimulator.load_psfs(args.psf_data_path, "psfs.h5")
    image_shape = psf_ds['H_img'].shape[:2]
    args.image_size_h = image_shape[0]
    args.image_size_w = image_shape[1]
    psfsim = PSFSimulator(image_shape, psf_size=psf_size,  normalize=args.normalize_psf)
    psfsim.set_H_obj(homography=psf_ds['H_obj'][:])
    psfsim.set_H_img(homography=psf_ds['H_img'][:])
    all_psfs = psf_ds['psfs'][:]
    metadata = psf_ds['metadata'][:]
    psf_ds.close()
    # ========================================================================================= generate PSF basis
    if psf_decomp == "PCA":
        sample_shape = (args.interpolation_samples, args.interpolation_samples)
        PSFSimulator.display_psfs(image_shape=image_shape, psfs=all_psfs, weights=None, metadata=metadata,
                            title="Loaded PSFs", skip=1)
        plt.savefig(join(md_dir, 'psfs_all.png'))

        # fit PCA and create weights according to PCA coefficients
        basis_psfs, basis_weights, pca_components, pca_mean, pca_var = get_pca_components(all_psfs, args.pca_n)
        a, b, c = basis_weights.shape[:3]
        weights_resized = np.zeros((a, b, c, image_shape[0], image_shape[1]))
        for i in range(a):
            for j in range(b):
                weights_resized[i, j] = cv2.resize(basis_weights[i, j].transpose([1, 2, 0]),
                                                   (image_shape[1], image_shape[0]),
                                                   interpolation=cv2.INTER_LINEAR, ).transpose([2, 0, 1])
        basis_weights = weights_resized
        psfs_img = PSFSimulator.display_psfs(image_shape, basis_psfs, basis_weights, skip=image_shape[0] // 10)
        plt.savefig(join(md_dir, 'psfs_sampled_from_basis.png'))
        plt.cla()
    # =================================================================== Generate PSFs weighted by sectors of the image
    elif psf_decomp == "sections":
        raise NotImplementedError()
    # =================================================================== Generate PSFs weighted by Zernike coefficients
    elif psf_decomp == "Zernike":
        n = args.zernike_n
        raise NotImplementedError()
    else:
        raise Exception("invalid psf decomposition type")
    # ======================================================================================= End PSF generation

    # ============================================================================== pad the psfs to get the image shape
    print(f"Original PSF/Weight size {basis_weights.shape[-2:]}")
    _psf = crop_arr(basis_psfs, image_shape[-2] + psf_size, image_shape[-1] + psf_size)
    _wei = crop_arr(basis_weights, image_shape[-2] + psf_size, image_shape[-1] + psf_size)
    if args.crop_edge is not None:
        _wei = _wei[..., args.crop_edge[0]:args.crop_edge[1], args.crop_edge[2]:args.crop_edge[3]]
        _psf = crop_arr(_psf, _wei.shape[-2], _wei.shape[-1])  # PSFs should always do center crop
    print(f"After padding PSF/Weight size {_wei.shape[-2:]}")  # if psf_size == 2*crop_edge: no cropping
    _wei[:, :, -1] = 1.0  # mean should be always zero even after padding

    # ============================================================================= Create the dataset Synthetic data
    PSFSimulator.save_psfs(md_dir, all_psfs, metadata=metadata,
                           basis_psfs=_psf, basis_weights=_wei,
                           H_img=psfsim.H_img, H_obj=psfsim.H_obj, save_as_image=True, fname="PSFs_with_basis.h5")

    method = args.conv_method
    ds, new_keys = get_SVB_Dataset(
        args,
        dataset,
        basis_psfs=torch.from_numpy(_psf),
        basis_weights=torch.from_numpy(_wei),
        is_blur_and_patch=True,
        is_test=True,
        h=image_shape[0],
        w=image_shape[1],
        method=method,
        blur_mode=1 if method == 'gathering' else 0,
        do_augment=False,
        crop_before_resize=args.crop_before_resize,
        # crop_edge=None if len(args.crop_edge) == 0 else crop_edge,  # no need since we cropped edges of the PSFs
        repeats=len(translate_by) if translate_by is not None else 1,
        translate_by=translate_by,
        rotate_by=args.rotate_by,
        device='cuda'
    )

    gt_key, blur_key = new_keys[0], new_keys[1]
    sample = ds.__getitem__(0)
    print(f"################## Keys and value shapes of the dataset")
    for key in sample.keys():
        try:
            print(f"{key} {sample[key].shape} {sample[key].dtype}")
        except:
            print(key, sample[key])

    # ds_loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1)

    # ========================================================================================== build noise masks
    add_perlin = add_perlin_noise(keys=(blur_key,), )

    # ================================================================================ Iterate through the dataset
    idx = 0
    size = len(ds)
    ds = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    for data in tqdm(ds, 'Generating images'):
        if args.add_perlin_noise:
            data = add_perlin(data)
        _x = data[gt_key].cpu().numpy()
        _y = data[blur_key].cpu().numpy()
        # x = crop_arr(x, x.shape[-2] // 2, x.shape[-1] // 2)
        # y = crop_arr(y, y.shape[-2] // 2, y.shape[-1] // 2)
        for i in range(len(_x)):
            x = _x[i].transpose([1, 2, 0])
            y = _y[i].transpose([1, 2, 0])
            dir_gt, dir_meas = gt_dir, meas_dir
            if idx < size * args.val_perc / 100:
                if translate_by is not None:
                    t = len(translate_by)
                    if idx - idx % t + t < size * args.val_perc / 100:
                        dir_gt, dir_meas = vgt_dir, vmeas_dir
                else:
                    dir_gt, dir_meas = vgt_dir, vmeas_dir

            cv2.imwrite(os.path.join(dir_gt, f'{idx:05d}.png'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR) * 255)
            cv2.imwrite(os.path.join(dir_meas, f'{idx:05d}.png'), cv2.cvtColor(y, cv2.COLOR_RGB2BGR) * 255)
            # y = y / y.max()
            idx += 1

    # ================================================================================================== pushing to HF
    from datasets import disable_caching
    from huggingface_hub import HfApi

    disable_caching()

    shutil.copyfile('./dataset/loading_script.py', join(ds_dir, f'{dataset_name}.py'))
    shutil.rmtree('./fff', ignore_errors=True)
    dataset = load_dataset(ds_dir, trust_remote_code=True, cache_dir='./fff')
    shutil.rmtree('./fff', ignore_errors=True)
    print(f"Length of the created dataset {len(dataset)}")

    repoid = f"harshana95/{dataset_name}"
    dataset.push_to_hub(repoid, num_shards={'train': 100, 'val': 1})

    api = HfApi()
    api.upload_folder(
        folder_path=md_dir,
        repo_id=repoid,
        path_in_repo="metadata",
        repo_type="dataset",

    )

