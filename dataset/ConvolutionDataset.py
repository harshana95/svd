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

