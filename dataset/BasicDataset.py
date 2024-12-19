import argparse
import os

import cv2
import einops
import numpy as np
import scipy
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from utils.dataset_utils import *


def get_default_preprocess(args, compose=True, test=False, do_augment=False, do_grayscale=False):
    if test:
        trans = [
            fix_image_shape(),
            # resize(args.image_size_h, args.image_size_w),
            # divisible_by(args.divisible),
            # padding(pad=(args.padding, args.padding, args.padding, args.padding))
        ]
    else:
        trans = [
            fix_image_shape(),
            # resize(args.image_size_h+2*args.padding, args.image_size_w+2*args.padding), # training data should be resized properly to this size
            # divisible_by(args.divisible),
        ]
        if do_augment:
            trans += [augment((args.image_size_h, args.image_size_w), horizontal_flip=True, resize_crop=True)]
            # ISSUE: if multiple images, the augmentation is not consistent
    if do_grayscale:
        trans += [grayscale()]
    if compose:
        trans = transforms.Compose(trans)
    return trans


def get_default_transforms(args, keys, compose=True):  # todo fix keys
    trans = [
        to_tensor(),
        add_poisson_noise(peak=args.peak_poisson, peak_dc=args.peak_dc_poisson, keys=keys),
        add_gaussian_noise(max_sigma=args.max_sigma, sigma_dc=args.dc_sigma, mu=args.mu_sigma, keys=keys)
    ]
    if compose:
        trans = transforms.Compose(trans)
    return trans


def get_default_ops_for_image_loading(args, compose=True):
    # do not include augmentation. augmentation should be done previously and saved in the dataset
    ops = [fix_image_shape(), divisible_by(8), to_tensor(), grayscale(),
           add_poisson_noise(peak=args.peak_poisson, peak_dc=args.peak_dc_poisson),
           add_gaussian_noise(max_sigma=args.max_sigma, sigma_dc=args.dc_sigma, mu=args.mu_sigma, keys=('meas',))]
    if compose:
        ops = transforms.Compose(ops)
    return ops


class ImageDataset(Dataset):

    def __init__(self, transform=None, **kwargs):
        self.labels = list(kwargs.keys())
        self.files = kwargs
        self.transform = transform

        assert len(self.labels) > 0, "provide at least one file list"
        length = -1
        for label in self.labels:
            if length == -1:
                length = len(self.files[label])
            else:
                assert length == len(self.files[label]), f"incompatible file lists {length} != {len(self.files[label])}"
        print(f"Image dataset length = {len(self)}")

    def __len__(self):
        return len(self.files[self.labels[0]])

    def __getitem__(self, idx):
        sample = {}
        for label in self.labels:
            img = cv2.cvtColor(cv2.imread(self.files[label][idx], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            bitdepth = 8
            if img.dtype == np.uint16:
                bitdepth = 16
            img = img.astype(np.float32) / (2**bitdepth - 1)

            sample[label] = img
            sample[label+'_f'] = self.files[label][idx]
            sample['idx'] = idx

        if self.transform:
            sample = self.transform(sample)

        return sample
