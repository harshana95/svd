import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
from tqdm import tqdm

from utils.dataset_utils import crop_from_center, crop_arr


def fftn(x):
    x_fft = torch.fft.fftn(x, dim=[-2, -1])
    return x_fft


def ifftn(x):
    return torch.fft.ifftn(x, dim=[-2, -1])


def conv_fft(H, x):
    if x.ndim == 4:
        # Batched version of convolution
        if H.shape[0] != x.shape[0]:
            H = H.repeat([x.size(0), 1, 1, 1])
        Y_fft = fftn(x) * H
        y = ifftn(Y_fft)
    elif x.ndim == 3:
        # Non-batched version of convolution
        Y_fft = torch.fft.fftn(x, dim=[1, 2]) * H
        y = torch.fft.ifftn(Y_fft, dim=[1, 2])
    else:
        raise Exception("Not a valid shape")
    return torch.real(y)


def psf_to_otf(ker, size):
    if ker.shape[2] % 2 == 0:
        ker = F.pad(ker, (0, 1, 0, 1), "constant", 0)
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    centre = ker.shape[2] // 2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre - 1):, (centre - 1):]
    psf[:, :, :centre, -(centre - 1):] = ker[:, :, (centre - 1):, :(centre - 1)]
    psf[:, :, -(centre - 1):, :centre] = ker[:, :, : (centre - 1), (centre - 1):]
    psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :(centre - 1), :(centre - 1)]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[-2, -1])
    return psf, otf


def conv_kernel(k, x, pad, mode='cyclic'):
    # x : b, c, H, W
    # k : b/1, c/1, h, w
    x = F.pad(x, pad, mode='reflect')
    b, c, h, w = x.size()
    bk, ck, h1, w1 = k.shape
    k_pad, H = psf_to_otf(k, [bk, ck, h, w])

    Ax = conv_fft(H, x)

    Ax = Ax[..., pad[0]:-pad[1], pad[2]:-pad[3]]
    k_pad = k_pad[..., pad[0]:-pad[1], pad[2]:-pad[3]]
    return Ax, k_pad


def initialize_kernels_and_weights(x, kernels, weights):
    hx, wx = x.shape[-2:]
    hw, ww = weights.shape[-2:]
    if hx < hw or wx < ww:
        pad = list(map(int, (np.floor((ww - wx) / 2), np.ceil((ww - wx) / 2), np.floor((hw - hx) / 2), np.ceil((hw - hx) / 2))))
        x = F.pad(x, pad, mode='reflect')
    elif hx > hw or wx > ww:
        pad = list(map(int, (np.floor((wx - ww) / 2), np.ceil((wx - ww) / 2), np.floor((hx - hw) / 2), np.ceil((hx - hw) / 2))))
        weights = F.pad(weights, pad, mode='constant')
    # weights = crop_arr(weights, x.shape[-2], x.shape[-1])

    hk, wk = kernels.shape[-2:]
    pad = (wk // 2, wk // 2 + (wk % 2 - 1), hk // 2, hk // 2 + (hk % 2 - 1))
    return x, kernels, weights, pad


class SVConv(nn.Module):
    def __init__(self, weighted, image_shape=None, kernels=None, weights=None, method='scattering', blur_mode=-1,
                 mean_ker=True,
                 device='cpu', *args, **kwargs):
        """
        @param image_shape: (H, W)
        @param kernels:  spatially varying kernels of shape (channels, height, width, kernel_size, kernel_size) or
                                                            (inc/groups=1, outc, n_psfs, kernel_size, kernel_size)
        @param weights: weights of shape (inc, n_psf, h, w) or None
        @param method: 'scattering' or 'gathering'
        @param blur_mode:
            0: mask and then blur
            1: blur and then mask
        @param mean_ker: mean of the kernel at the end of kernel list?
        @param args:
        @param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.weighted = weighted
        self.device = device
        self.mean_ker = mean_ker
        self.mode = blur_mode
        self.kernels = None
        self.weights = None

        if kernels is not None and weights is not None:
            assert len(kernels.shape) == 5
            # _, kernels, weights, pad = initialize_kernels_and_weights(torch.zeros(image_shape), kernels, weights)
            print(f"SVConv: Kernels {kernels.shape} Weights: {'None' if weights is None else weights.shape}")
            # assert len(weights.shape) == 4, f"weights should be of shape (c, n, h, w) but got {weights.shape}"
            assert blur_mode >= 0, "Blur mode should be 1 or 0"

            self.c = kernels.shape[0]
            self.n = kernels.shape[2]
            self.kernels = torch.tensor(kernels).to(torch.float32).to(self.device)
            self.weights = torch.tensor(weights).to(torch.float32).to(self.device)
            # self.hx, self.wx = image_shape[0], image_shape[1]
            # self.hk, self.wk = kernels.shape[3], kernels.shape[4]
            # print(f"SVConv: Image size {self.hx}x{self.wx} Kernel size {self.hk}x{self.wk}")
        if kernels is not None and weights is None:
            self.c = kernels.shape[0]
            # self.hx, self.wx = kernels.shape[1], kernels.shape[2]
            # self.hk, self.wk = kernels.shape[3], kernels.shape[4]
            assert image_shape[0] == image_shape[0] and image_shape[1] == image_shape[1], "Image shape mismatch"
            self.kernels = torch.tensor(kernels).to(torch.float32).to(self.device).unsqueeze(0)

        # set the forward function
        if method == 'gathering':
            self._method = 0
            self._forward = self._gather_w if weighted else self._gather
            if weighted:
                print("Provide mean at the end of kernel list" if mean_ker else "")
        elif method == 'scattering':
            self._method = 1
            self._forward = self._scatter_w if weighted else self._scatter
        else:
            raise Exception("Wrong method type!")

    def _gather(self, x, kernels, weights):
        raise NotImplementedError()

    def _scatter(self, x, kernels, weights):
        raise NotImplementedError()

    def _gather_w(self, x, kernels, weights):
        raise NotImplementedError()

    def _scatter_w(self, x, kernels, weights):
        raise NotImplementedError()

    def forward(self, x, kernels=None, weights=None):
        """
        @param weights:
        @param kernels:
        @param x: input tensor of shape (batch_size, in_channels, height, width)
        @return: conv/splat image
        """
        assert len(x.shape) == 4
        # if self._method == 1:
        #     # check if the input x has same height width as kernel
        #     assert x.shape[-2] == self.hx and x.shape[-1] == self.wx

        x = x.to(self.device)
        return self._forward(x, kernels, weights)


class SVConv_Brute(SVConv):
    def __init__(self, weighted, image_shape, kernels, weights=None, method='scattering', *args, **kwargs):
        super().__init__(weighted, image_shape, kernels, weights, method, *args, **kwargs)
        # self.__pad = (self.wk // 2, self.wk // 2 + (self.wk % 2 - 1), self.hk // 2, self.hk // 2 + (self.hk % 2 - 1))
        # + part because if even, output sizes don't match

        # Case: build kernels from splat weights and psfs
        if weights is not None and method == 'scattering':
            # building full PSFs. MEMORY INTENSIVE!
            # (channels, height, width, kernel_size, kernel_size)
            npsf, hk, wk = kernels.shape[-3:]
            c = kernels.shape[0]
            _kernels = torch.zeros((c, image_shape[0], image_shape[1], hk, wk), device=self.device)

            for i in tqdm(range(image_shape[0]), desc="Building pixelwise kernels from"):
                for j in range(image_shape[1]):
                    kernel = (self.kernels[:, 0, :] * weights[:, :, i:i + 1, j:j + 1]).sum(1)
                    _kernels[:, i, j] = kernel
            self.kernels = _kernels.unsqueeze(1)

    def _gather(self, x, kernels=None, weights=None):
        if kernels is None:
            kernels = self.kernels
        if weights is None:
            weights = self.weights
        b, c, hx, wx = x.shape
        npsf, hk, wk = kernels.shape[-3:]
        __pad = (wk // 2, wk // 2 + (wk % 2 - 1), hk // 2, hk // 2 + (hk % 2 - 1))
        x = F.pad(x, __pad, mode='reflect')

        # extract patches around each pixel; unfold(dimension, size, step) -> Tensor
        # (batch_size, in_channels, height - kernel_size + 1 , width - kernel_size + 1 , kernel_size , kernel_size)
        patches = x.unfold(2, wk, 1).unfold(3, hk, 1)

        # reshape patches and kernels for element-wise multiplication
        patches = patches.reshape(b, c, -1)
        kernels = kernels.reshape(1, c, -1)

        # perform element-wise multiplication and sum along appropriate dimensions
        output = (patches * kernels).reshape(b, c, hx, wx, -1).sum(-1)
        return output

    def _gather_w(self, x, kernels=None, weights=None):
        if kernels is None:
            kernels = self.kernels
        if weights is None:
            weights = self.weights
        x, kernels, weights, __pad = initialize_kernels_and_weights(x, kernels, weights)
        b, c, hx, wx = x.shape
        npsf, hk, wk = kernels.shape[-3:]
        out = torch.zeros_like(x)
        _n = npsf - 1 if self.mean_ker else npsf
        for n in range(_n):
            _w = einops.rearrange(weights[:, 0, :, n], 'c b h w -> b c h w')
            _x = x.detach().clone()
            _x = _x * _w if self.mode == 0 else _x
            _x = F.pad(_x, __pad, mode='reflect')

            blurred = F.conv2d(_x, kernels[:, :, n], stride=1, padding='valid', groups=c)
            blurred = blurred * _w if self.mode == 1 else blurred
            out += blurred

        if self.mean_ker:
            _x = x.detach().clone()
            _x = F.pad(_x, __pad, mode='reflect')
            blurred = F.conv2d(_x, kernels[:, :, -1], stride=1, padding='valid', groups=c)
            out += blurred
        return out, kernels, weights

    def _scatter(self, x, kernels=None, weights=None):
        if kernels is None:
            kernels = self.kernels
        # if weights is None:
        #     weights = self.weights
        # x, kernels, weights, __pad = initialize_kernels_and_weights(x, kernels, weights)
        hk, wk = kernels.shape[-2:]
        pad = (wk // 2, wk // 2 + (wk % 2 - 1), hk // 2, hk // 2 + (hk % 2 - 1))
        b, c, hx, wx = x.shape
        npsf, hk, wk = kernels.shape[-3:]
        output = torch.zeros_like(x)
        output = F.pad(output, pad, mode='constant')
        for i in range(hx):
            for j in range(wx):
                output[:, :, i:i + hk, j:j + wk] += kernels[:, :, i, j] * x[:, :, i:i + 1, j:j + 1]
        output = output[:, :, hk // 2:hk // 2 + hx, wk // 2:wk // 2 + wx]
        return output, kernels, weights

    def _scatter_w(self, x, kernels, weights):
        return self._scatter(x, kernels, weights)


class SVConv_fast(SVConv):
    def __init__(self, weighted, image_shape=None, kernels=None, weights=None, method='scattering', *args, **kwargs):
        super().__init__(weighted, image_shape, kernels, weights, method, *args, **kwargs)
        self.H = None
        self.W = None
        if kernels is not None:
            # todo check parameter s. It can pad by zero inside fft function. (more efficient?)
            self.H = fft2(self.kernels, dim=(-2, -1))
        if weights is not None:
            self.W = self.weights

    def _scatter_w(self, x, kernels=None, weights=None):  # FFT convolution is a scattering operation!
        # weights : (batch_size, inc, outc, npsf, hw, ww)
        # kernels : (batch_size, inc, outc, npsf, hk, wk)
        # x : (batch_size, inc, h, w)
        h, w = x.shape[-2:]
        if self.H is not None:
            kernels = self.kernels
            H = self.H
        else:
            H = fft2(kernels, dim=(-2, -1))
        if self.W is not None:
            weights = self.W
        else:
            weights = weights

        H = H[..., 0, :, :, :]
        weights = weights[..., 0, :, :, :]

        if self.mode == 1:
            X = fft2(x, dim=(-2, -1))
            out = ifftshift(ifft2(X[:, :, None, None] * H[None], dim=(-2, -1)), dim=(-2, -1)).real
            out = (out * weights).sum(-3)[..., 0, :, :]
            print("This blur mode should not be used when blurring from FT!")
        else:
            X = fft2(x[..., None, :, :] * weights[None], dim=(-2, -1))
            out = ifftshift(ifft2(X * H[None], dim=(-2, -1)), dim=(-2, -1)).real.sum(-3)

        # ho, wo = out.shape[-2:]
        # if ho != h or wo != w:
        #     out = crop_arr(out, h, w)

        H = H[..., None, :, :, :]
        weights = weights[..., None, :, :, :]

        return out, kernels, weights


class SpatiallyVaryingCrossChannelConvolution(nn.Module):
    def __init__(self, kernels):
        # spatially varying kernels of shape (batch_size, outc, inc, height, width, kernel_size, kernel_size)
        assert kernels.shape[0] == 1  # batch size of kernels is 1
        super(SpatiallyVaryingCrossChannelConvolution, self).__init__()

        self.kernels = kernels
        self.outc = kernels.shape[1]
        self.inc = kernels.shape[2]
        self.hx, self.wx = kernels.shape[3], kernels.shape[4]
        self.hk, self.wk = kernels.shape[5], kernels.shape[6]

    def forward(self, x):
        # x: input tensor of shape (batch_size, in_channels, height, width)
        assert len(x.shape) == 4
        assert x.shape[-2] == self.hx and x.shape[-1] == self.wx  # check if the input x has same height width as kernel

        x = F.pad(x, (self.wk // 2, self.wk // 2, self.hk // 2, self.hk // 2), mode='reflect')

        # extract patches around each pixel; unfold(dimension, size, step) -> Tensor
        # (batch_size, in_channels, height - kernel_size + 1 , width - kernel_size + 1 , kernel_size , kernel_size)
        patches = x.unfold(2, self.wk, 1).unfold(3, self.hk, 1)

        # reshape patches and kernels for element-wise multiplication
        patches = patches.reshape(1, self.inc, -1).unsqueeze(1)
        kernels = self.kernels.reshape(1, self.outc, self.inc, -1)

        # perform element-wise multiplication and sum along appropriate dimensions
        output = (patches * kernels).mean(dim=2).reshape(1, self.outc, self.hx, self.wx, -1).sum(-1)
        return output
