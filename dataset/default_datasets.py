import numpy as np
from torchvision.transforms import transforms

from dataset.BasicDataset import get_default_preprocess, get_default_transforms, ImageDataset
from dataset.ConvolutionDataset import ConvolveImageDatasetWrapper
from dataset.PatchDataset import PatchedImageDatasetWrapper
from dataset.SVconv import SVConv_fast, conv_kernel
from utils.dataset_utils import crop_from_center, crop_arr, crop2d, divisible_by, resize, padding, translate_image, \
    crop_center, to_tensor, fix_image_shape, augment, grayscale, rotate


def get_SVB_Dataset(args, dataset, basis_psfs, basis_weights=None,
                    is_test=False, is_blur_and_patch=True, h=-1, w=-1,
                    inplace=False,
                    method='scattering', blur_mode=0,
                    do_augment=False,
                    crop_before_resize=None,
                    crop_edge=None,
                    repeats=1,
                    translate_by=None,
                    rotate_by=None,
                    additional_dataset=None,
                    grayscale_blurred=False,
                    key_name='',
                    rescale_patch_size_by=1,
                    device='cpu'):
    """
    Returns the default dataset generator with Spatially varying blur. Can be used with weighted PSFs or pixel-wise PSFs
    """
    print("################# Getting SVB dataset")
    if repeats > 1:
        dataset = np.repeat(dataset, repeats)
        if additional_dataset is not None:
            additional_dataset = np.repeat(additional_dataset, repeats)
    if basis_weights is None:
        raise NotImplementedError()
    else:
        if is_blur_and_patch:
            gt_keys = (key_name + 'gt',)
            additional_keys = tuple()
            if additional_dataset is not None:
                additional_keys = (key_name + 'gt2',)

            # blur and then patch
            blur_op = SVConv_fast(weighted=True, blur_mode=blur_mode, method=method,
                                  kernels=basis_psfs, weights=basis_weights, device=device)

            # PREPROCESS
            preprocess = [fix_image_shape()]
            if is_test and do_augment:
                preprocess.append(augment(gt_keys + additional_keys,
                                          (args.image_size_h, args.image_size_w),
                                          horizontal_flip=True, resize_crop=True))
            if crop_before_resize is not None:
                print(f"Crop {crop_before_resize} before resize")
                preprocess.append(crop2d(crop_indices=crop_before_resize, keys=gt_keys + additional_keys))
            if is_test:
                assert h > 0 and w > 0
                print(f"Resize to {h}x{w}")
                preprocess.append(resize(h, w, keys=gt_keys + additional_keys))
            preprocess.append(to_tensor())
            if args.n_colors == 1:
                preprocess.append(grayscale(gt_keys + additional_keys, channels=args.n_colors))
            if args.divisible > 1:
                preprocess.append(divisible_by(args.divisible, method='crop', keys=gt_keys + additional_keys))
            if rotate_by is not None:
                print(f"Rotate by {rotate_by}")
                preprocess.append(rotate(gt_keys + additional_keys, rotate_by))
            preprocess.append(padding(h=basis_weights.shape[-2], w=basis_weights.shape[-1], mode='constant',
                                      keys=gt_keys + additional_keys))
            if translate_by is not None:
                preprocess.append(translate_image(translate_by, gt_keys + additional_keys))

            _kwargs = {gt_keys[0]: dataset, "transform": transforms.Compose(preprocess)}
            if additional_dataset is not None:
                _kwargs[additional_keys[0]] = additional_dataset

            ds = ImageDataset(**_kwargs)

            blur_keys = tuple(ConvolveImageDatasetWrapper.convert_key(key, False) for key in gt_keys)
            new_keys = gt_keys + blur_keys + additional_keys

            # POSTPROCESS
            transform = get_default_transforms(args, blur_keys, compose=False)
            # if crop_edge is not None:
            #     # crop the edges to remove edge effect from PSFs
            #     transform.append(crop2d(crop_indices=crop_edge,
            #                             # [args.padding, args.image_size_h - args.padding, args.padding, args.image_size_w - args.padding],
            #                             keys=gt_keys + blur_keys + additional_keys
            #                             ))
            if grayscale_blurred:
                transform.append(grayscale(blur_keys, channels=args.n_colors))
            print(f"Image X f(X) dataset. Preprocess {[tmp.__class__.__name__ for tmp in preprocess]} "
                  f"Transforms {[tmp.__class__.__name__ for tmp in transform]}")

            if len(basis_psfs.shape) == 5:
                basis_psfs = basis_psfs[None]
            if len(basis_weights.shape) == 5:
                basis_weights = basis_weights[None]
            ds = ConvolveImageDatasetWrapper(ds, valid_keys=gt_keys, blur_op=blur_op, inplace=False,
                                             transform=transforms.Compose(transform),
                                             basis_psf=basis_psfs, basis_weights=basis_weights,

                                             # metadata={},
                                             # metadata_injector=injector,
                                             # injection=-1,
                                             # image_key='gt', weight_key=weights_keys[0]
                                             )
            if args.patch_size > 0:  # be careful when patching earlier than convolve. transforms are friendly only to images, don't add noise to position arrays
                valid_keys = gt_keys + blur_keys + additional_keys
                ds = PatchedImageDatasetWrapper(ds,
                                                patch_size=int(args.patch_size/rescale_patch_size_by),
                                                stride=int(args.stride/rescale_patch_size_by),
                                                valid_keys=valid_keys, inplace=inplace)
                new_keys = tuple(PatchedImageDatasetWrapper.convert_key(key, inplace=inplace) for key in valid_keys)
        else:
            raise NotImplementedError()
    return ds, new_keys


def get_SiVB_Dataset(args, dataset, is_test=False, is_blur_and_patch=True, inplace=False, augment=False, key_name=''):
    print("################# Getting SiVB dataset")
    psf_generator = Sim_PSF(image_shape=(0, 0), psf_size=args.psf_size,
                            normalize=True)  # image shape not needed for SiV
    preprocess = get_default_preprocess(args, compose=False, test=is_test, do_augment=augment,
                                        do_grayscale=args.n_colors == 1)
    preprocess.insert(-1, resize(args.image_size_h, args.image_size_w, keys=(key_name + 'gt',)))
    ds = ImageDataset(gt=dataset, transform=transforms.Compose(preprocess))

    def blur_op(x, ker, weights):
        hk, wk = ker.shape[-2:]
        pad = (wk // 2, wk // 2 + (wk % 2 - 1), hk // 2, hk // 2 + (hk % 2 - 1))
        out, _ = conv_kernel(ker, x, pad)
        return out, ker, weights

    def injector(sample, metadata, psf_generator, image_key, args):
        x = sample[image_key]
        h, w = x.shape[-2:]
        ker, psf_metadata = psf_generator.generate_psf(rand_index_func(0, h), rand_index_func(0, w), 0, **args.__dict__)
        sample['psfs'] = torch.from_numpy(ker[None, None, :, :]).to(x.device)
        sample['metadata'] = psf_metadata

        return sample

    if is_blur_and_patch:
        gt_keys = (key_name + 'gt',)
        blur_keys = tuple(ConvolveImageDatasetWrapper.convert_key(key, inplace=False) for key in gt_keys)
        transform = get_default_transforms(args, blur_keys, compose=False)
        ds = ConvolveImageDatasetWrapper(ds, basis_psf=None, valid_keys=gt_keys, blur_op=blur_op, inplace=False,
                                         kernel_key='psfs',
                                         transform=transforms.Compose(transform),
                                         metadata={}, metadata_injector=injector, injection=-1,
                                         psf_generator=psf_generator, image_key=gt_keys[0],
                                         args=args)  # injection kwargs
        if args.patch_size > 0:
            ds = PatchedImageDatasetWrapper(ds, patch_size=args.patch_size, stride=args.stride,
                                            valid_keys=gt_keys + blur_keys, inplace=inplace)
            blur_keys = tuple(PatchedImageDatasetWrapper.convert_key(key, inplace=inplace) for key in blur_keys)
            gt_keys = tuple(PatchedImageDatasetWrapper.convert_key(key, inplace=inplace) for key in gt_keys)
    else:
        raise NotImplementedError()
    new_keys = gt_keys + blur_keys
    return ds, new_keys