import argparse
import os

import einops
import numpy as np
import scipy
import torch

import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset.BasicDataset import ImageDataset, get_default_preprocess
from dataset.DatasetWrapper import DatasetWrapper
from utils.dataset_utils import crop_from_center


class PatchedImageDatasetWrapper(DatasetWrapper):
    """
    Get patched images from a given x, y image outputting dataset.
    """

    @staticmethod
    def convert_key(key, inplace):
        if inplace:
            return key
        return key + "_patch"

    # @staticmethod
    # def global_processor(sample, metadata, patch_size, image_key='gt', **kwargs):
    #     sample = default_global_processor(sample, metadata, image_key, **kwargs)
    #     hk, wk = sample['psfs'].shape[-2:]
    #     # top, left = hk // 2 - patch_size // 2, wk // 2 - patch_size // 2
    #     # if top >= 0 or left >= 0:
    #     #     sample['psfs'] = sample['psfs'][..., top:top + patch_size, left:left + patch_size]
    #     return sample

    @staticmethod
    def bartlett(ph=32, pw=32, color=True):
        x = np.arange(1, pw + 1, 1.0) / pw
        y = np.arange(1, ph + 1, 1.0) / ph

        xx, yy = np.meshgrid(x, y)
        a0 = 0.62
        a1 = 0.48
        a2 = 0.38
        win = a0 - a1 * np.abs(xx - 0.5) - a2 * np.cos(2 * np.pi * xx)
        win *= a0 - a1 * np.abs(yy - 0.5) - a2 * np.cos(2 * np.pi * yy)
        win = win[None, :, :]
        if color:
            win = np.repeat(win, 3, 0)
        return win

    @staticmethod
    def patchify_op(arr, patch_size, stride):

        size_w = arr.shape[-1]
        exp_size_w = int(np.ceil((size_w - patch_size) / stride)) * stride + patch_size
        exp_size_w = exp_size_w + patch_size if exp_size_w < size_w else exp_size_w

        size_h = arr.shape[-2]
        exp_size_h = int(np.ceil((size_h - patch_size) / stride)) * stride + patch_size
        exp_size_h = exp_size_h + patch_size if exp_size_h < size_h else exp_size_h

        if exp_size_w > size_w or exp_size_h > size_h:
            arr = F.pad(arr, (0, exp_size_w - size_w, 0, exp_size_h - size_h))

        # patchifying
        patched = arr.unfold(-2, patch_size, stride).unfold(-2, patch_size, stride)
        patched = einops.rearrange(patched, "... c n1 n2 ph pw -> ... (n1 n2) c ph pw")
        patched_pos = torch.tensor(np.mgrid[0:exp_size_h - patch_size + 1:stride,
                                   0:exp_size_w - patch_size + 1:stride].reshape(2, -1).T)
        return patched, patched_pos

    def patchify(self, sample, valid_keys, patch_size, stride):
        """
        unfold last two dimensions with size=patch_size and step=stride. pad with zero if necessary
        @param sample: dict
        @param valid_keys: keys to patchify
        @return: dict
        """

        for k1 in valid_keys:
            k2 = PatchedImageDatasetWrapper.convert_key(k1, self.inplace)
            k3 = k2 + '_pos'
            sample[k2], sample[k3] = PatchedImageDatasetWrapper.patchify_op(sample[k1], patch_size, stride)
        return sample

    def __init__(self, dataset, patch_size, stride, valid_keys,
                 inplace=False, metadata=None, metadata_injector=None, injection=0, transform=None, **kwargs):
        """

        @param dataset: dataset
        @param patch_size: patch size
        @param stride: how much to skip
        """
        super().__init__(dataset, inplace, metadata, metadata_injector, injection, transform, **kwargs)
        assert patch_size > stride
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.valid_keys = valid_keys
        self.injection_kwargs = {**kwargs}

    def operator(self, sample, **kwargs):
        return self.patchify(sample, self.valid_keys, self.patch_size, self.stride)

    @staticmethod
    def merge_patches(patches, pos):
        """

        @param patches: patches with shape (B, n, c, ph, pw)
        @param pos: patch top-left position (B, n, 2)
        @return:
        """
        B, n, c, ph, pw = patches.shape
        window = PatchedImageDatasetWrapper.bartlett(ph, pw, color=c == 3)

        merged = []
        for b in range(B):
            # find the image size
            h, w = pos[b, :, 0].max() + ph, pos[b, :, 1].max() + pw
            out = np.zeros((c, h, w))
            out_weights = np.zeros_like(out)
            for patch, p in zip(patches[b], pos[b]):
                i, j = p
                out[:, i:i + ph, j:j + pw] += patch * window
                out_weights[:, i:i + ph, j:j + pw] += window
            out /= (out_weights + 1e-6)
            merged.append(out)
        return merged

