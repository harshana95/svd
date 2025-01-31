import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import diffusers
import einops
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, repo_exists, delete_repo, whoami
from torch.fft import fft2, ifftshift, ifft2

from skimage.metrics import structural_similarity as ssim

def log_image(accelerator, formatted_images, name, step):
    name += '.png'
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(name, formatted_images, step, dataformats="NCHW")
        elif tracker.name == "wandb":
            tracker.log({"validation": wandb.Image(formatted_images, caption=name)})
        elif tracker.name == "comet_ml":
            tracker.writer.log_image(np.hstack(formatted_images), name=name, step=step, image_channels="first")
        else:
            raise Exception(f"image logging not implemented for {tracker.name}")

def log_metrics(img1, img2, metrics, accelerator, step):
    ret = {}
    img1 = einops.rearrange(img1, 'c h w -> h w c')
    img2 = einops.rearrange(img2, 'c h w -> h w c')
    for metric in metrics:
        if metric == 'psnr':
            out = calculate_psnr(img1, img2, metrics[metric]["crop"])
        elif metric == 'ssim':
            out = calculate_ssim(img1, img2, metrics[metric]["crop"])
        elif metric == 'mse':
            out = calculate_mse(img1, img2, metrics[metric]["crop"])
        else:
            raise Exception(f"Unknown metric {metric}")
        ret[metric]= out
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            raise NotImplementedError()
        elif tracker.name == "wandb":
            raise NotImplementedError()
        elif tracker.name == "comet_ml":
            tracker.writer.log_metrics(ret, step=step)
        else:
            raise Exception(f"image logging not implemented for {tracker.name}")

        

def calculate_psnr(img1, img2, crop=30):
    mse = calculate_mse(img1*255, img2*255, crop=crop)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def calculate_mse(img1, img2, crop=30):
    return np.mean((img1[crop:-crop, crop:-crop] - img2[crop:-crop, crop:-crop]) ** 2)

def calculate_ssim(img1, img2, crop=30):
    # Ensure the images are in grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256, 256))
    # Compute SSIM between the two images
    ssim_value, ssim_map = ssim(img1[crop:-crop, crop:-crop], img2[crop:-crop, crop:-crop], full=True, win_size=7)
    return ssim_value, ssim_map

def fftn(x):
    x_fft = torch.fft.fftn(x, dim=[-2, -1])
    return x_fft


def ifftn(x):
    return torch.fft.ifftn(x, dim=[-2, -1])


def rgb2yuv(x: torch.tensor, renormalize=True):
    if renormalize:
        x_scaled = ((x + 1) / 2) * 255
    else:
        x_scaled = x * 255

    R, G, B = x_scaled.chunk(3, dim=-3)
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
    U = -0.148 * R - 0.291 * G + 0.439 * B + 128
    V = 0.439 * R - 0.368 * G - 0.071 * B + 128
    return torch.stack([Y, U, V], dim=-3)


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


def merge_patches(patches, pos):
    """

    @param patches: patches with shape (n, c, ph, pw)
    @param pos: patch top-left position (n, 2)
    @return:
    """
    n, c, ph, pw = patches.shape
    window = torch.tensor(bartlett(ph, pw, color=c == 3), device=patches.device)

    # find the image size
    h, w = pos[:, 0].max() + ph, pos[:, 1].max() + pw
    out = torch.zeros((c, h, w), device=patches.device)
    out_weights = torch.zeros_like(out, device=patches.device)
    for patch, p in zip(patches, pos):
        i, j = p
        out[:, i:i + ph, j:j + pw] += patch * window
        out_weights[:, i:i + ph, j:j + pw] += window
    out /= (out_weights + 1e-6)
    return out


def patchify(arr, psz_h, psz_w, str_h, str_w):
    size_w = arr.shape[-1]
    exp_size_w = int(np.ceil((size_w - psz_w) / str_w)) * str_w + psz_w
    exp_size_w = exp_size_w + psz_w if exp_size_w < size_w else exp_size_w

    size_h = arr.shape[-2]
    exp_size_h = int(np.ceil((size_h - psz_h) / str_h)) * str_h + psz_h
    exp_size_h = exp_size_h + psz_h if exp_size_h < size_h else exp_size_h

    if exp_size_w > size_w or exp_size_h > size_h:
        arr = torch.nn.functional.pad(arr, (0, exp_size_w - size_w, 0, exp_size_h - size_h))

    # patch arr
    patched = arr.unfold(-2, psz_h, str_h).unfold(-2, psz_w, str_w)
    patched = einops.rearrange(patched, "... c n1 n2 ph pw -> ... (n1 n2) c ph pw")
    patched_pos = torch.tensor(np.mgrid[0:exp_size_h - psz_h + 1:str_h,
                               0:exp_size_w - psz_w + 1:str_w].reshape(2, -1).T)
    return patched, patched_pos


def convolve(images, kernels, weights, kernels_in_f=False, mode=0):
    # weights : (batch_size, inc, outc, npsf, hw, ww)
    # kernels : (batch_size, inc, outc, npsf, hk, wk)
    # images : (batch_size, inc, h, w)
    assert images.shape[-2:] == weights.shape[-2:]
    kernels = kernels[..., 0, :, :, :]
    weights = weights[..., 0, :, :, :]
    if not kernels_in_f:
        H = fft2(kernels, dim=(-2, -1))
    else:
        H = kernels

    # if mode == 1:
    #     X = fft2(images, dim=(-2, -1))
    #     out = ifftshift(ifft2(X[:, :, None] * H, dim=(-2, -1)), dim=(-2, -1)).real
    #     out = (out * weights).sum(2)
    #     print("This blur mode should not be used when blurring from FT!")
    # else:
    X = fft2(images[..., None, :, :] * weights, dim=(-2, -1))
    out = ifftshift(ifft2(X * H, dim=(-2, -1)), dim=(-2, -1)).real.sum(-3)

    kernels = kernels[..., None, :, :, :]
    weights = weights[..., None, :, :, :]

    return out


def normalized_rescale(arr, h, w):
    to_numpy = False
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
        to_numpy = True
    sum1 = arr.sum((-1, -2), keepdims=True)
    arr = torch.nn.functional.interpolate(arr, (h, w), mode='bicubic')
    sum2 = arr.sum((-1, -2), keepdims=True)
    sum1 += 1e-6
    sum2 += 1e-6
    arr = arr / sum2 * sum1
    if to_numpy:
        arr = arr.numpy()
    return arr


def crop_arr(arr, h, w, mode='constant'):
    hw, ww = arr.shape[-2:]
    do_pad = False
    istorch = isinstance(arr, torch.Tensor) or isinstance(arr, torch.nn.Parameter)
    if istorch:
        pad = [0, 0, 0, 0]
    else:
        pad = [[0, 0]] * (len(arr.shape))
    if h < hw:
        crop_height = min(h, hw)
        top = hw // 2 - crop_height // 2
        arr = arr[..., top:top + crop_height, :]
    elif h > hw:
        do_pad = True
        if istorch:
            pad[-2] = int(np.ceil((h - hw) / 2))
            pad[-1] = int(np.floor((h - hw) / 2))
        else:
            pad[-2] = [int(np.ceil((h - hw) / 2)), int(np.floor((h - hw) / 2))]
    if w < ww:
        crop_width = min(w, ww)
        left = ww // 2 - crop_width // 2
        arr = arr[..., :, left:left + crop_width]
    elif w > ww:
        do_pad = True
        if istorch:
            pad[0] = int(np.ceil((w - ww) / 2))
            pad[1] = int(np.floor((w - ww) / 2))
        else:
            pad[-1] = [int(np.ceil((w - ww) / 2)), int(np.floor((w - ww) / 2))]
    if do_pad:
        if istorch:
            arr = torch.nn.functional.pad(arr, pad, mode=mode)
        else:
            arr = np.pad(arr, pad, mode=mode)
    return arr


def fill_crop(arr, h, w):
    H, W = arr.shape[-2:]
    if h / w < H / W:  # width of the arr is smaller. fit width
        _w = w
        _h = round(w * (H / W))
    else:
        _h = h
        _w = round(h * (W / H))
    if H != _h or W != _w:  # do resize first to fill the image to given size
        _squeeze = False
        if len(arr.shape) == 3:
            arr = arr[None,]
            _squeeze = True
        arr = torch.nn.functional.interpolate(arr, (_h, _w))
        if _squeeze:
            arr = arr[0]
    return crop_arr(arr, h, w)


def divisible_crop(arr, n):
    h, w = arr.shape[-2:]
    if n > 1:
        H = h - h % n
        W = w - w % n
        a, c = h // 2 - H // 2, w // 2 - W // 2
        b, d = a + H, c + W
        assert  b > a and d > c
        return arr[..., a:b, c:d]
    return arr


def relative_crop(arr, rh, rw, u, v):
    """
    arr : array of images shaped NCHW
    rh : relative height of the crop (0<rh<1)
    rw : relative width of the crop (0<rw<1)
    u : relative top of the crop (0<u<1)
    v : relative left of the crop (0<v<1)
    """
    H, W = arr.shape[-2:]
    top, left = int(H * u), int(W * v)
    bottom, right = top+int(H * rh), left+int(W * rw)
    if bottom > H or right > W:
        raise ValueError(f"Crop size out of bounds. Image shape {H}x{W}. Crop {top}:{bottom}, {left}:{right}")
    return arr[..., top:bottom, left:right]


def initialize(args, logger):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    args.path.experiments_root = os.path.join(args.path.root, args.name)
    logging_dir = os.path.join(args.path.experiments_root, args.path.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.path.experiments_root,
                                                      logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # if args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is None:
        #     logger.warning(f"Deleting folder... {args.output_dir}")
        #     try:
        #         shutil.rmtree(args.output_dir)
        #     except FileNotFoundError:
        #         pass
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    return accelerator


def keep_last_checkpoints(output_dir, checkpoints_total_limit, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)

def generate_folder(p):
    if not os.path.exists(p):
        os.makedirs(p)

def log_metric(trackers, metrics, step):
    for tracker in trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_scaler('custom', metrics, global_step=step)
        elif tracker.name == "wandb":
            tracker.log(metrics, step=step)
        elif tracker.name == "comet_ml":
            tracker.writer.log_metrics(metrics, step=step)
        else:
            raise Warning(f"image logging not implemented for {tracker.name}")
