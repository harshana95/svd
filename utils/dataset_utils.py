import PIL
import cv2
import einops
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pynoise.noisemodule import Perlin
from pynoise.noiseutil import grayscale_gradient, RenderImage, noise_map_plane, noise_map_plane_gpu
from torchvision.transforms import transforms
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.functional as TF

class resize:
    def __init__(self, h, w, keys):
        self.h = h
        self.w = w
        self.keys = keys

    def _resize(self, x):
        resized = cv2.resize(x.transpose([1, 2, 0]), (self.w, self.h))
        if len(resized.shape) == 2:
            resized = resized[:, :, None]
        return resized.transpose([2, 0, 1])

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = self._resize(sample[key])
        return sample


class divisible_by:
    def __init__(self, n, keys, method='crop'):
        self.n = n
        self.keys = keys
        self.method = 0 if method == 'crop' else 1

    @staticmethod
    def _crop(x, n):
        h, w = x.shape[-2:]
        if n > 1:
            H = h - h % n
            W = w - w % n
            a, c = h // 2 - H // 2, w // 2 - W // 2
            b, d = a + H, c + W
            return x[..., a:b, c:d]
        return x

    @staticmethod
    def _resize(x, n):
        c, h, w = x.shape
        H = (h // n) * n
        W = (w // n) * n
        resized = cv2.resize(x.transpose([1, 2, 0]), (W, H))
        return resized.transpose([2, 0, 1])

    def __call__(self, sample):
        for key in self.keys:
            if self.method == 0:
                sample[key] = self._crop(sample[key], self.n)
            else:
                sample[key] = self._resize(sample[key], self.n)

        return sample


class crop2d:
    def __init__(self, crop_indices=None, keys=()):
        self.active_keys = keys
        self.ci = crop_indices

    def crop(self, image):
        # sample shape (..., h, w)
        image = image[..., self.ci[0]:self.ci[1], self.ci[2]:self.ci[3]]
        return image

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key] = self.crop(sample[key])
        return sample


class crop_best:
    def __init__(self, keys=()):
        self.active_keys = keys

    def crop(self, image):
        # sample shape (..., h, w)
        h, w = image.shape[-2], image.shape[-1]
        ch, cw = h // 2, w // 2
        r = min(ch, cw)
        image = image[..., ch - r:ch + r, cw - r:cw + r]
        return image

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key] = self.crop(sample[key])
        return sample


class crop_center:
    def __init__(self, h, w, keys=()):
        self.h = h
        self.w = w
        self.active_keys = keys

    def crop(self, image):
        return crop_arr(image, self.h, self.w)

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key] = self.crop(sample[key])
        return sample


class fix_image_shape:
    def fix(self, image):
        try:
            shape = image.shape
        except:
            return image
        if len(image.shape) == 4:
            raise Exception("too many dimensions. expected 3 dimensions")
            # if image.shape[1] == 3 or image.shape[1] == 1:
            #     pass
            # elif image.shape[-1] == 3 or image.shape[-1] == 1:
            #     image = image.transpose([0, 3, 1, 2])
            # else:
            #     raise Exception(f"unknown image shape {image.shape}")
        elif len(image.shape) == 3:
            if image.shape[-1] == 3 or image.shape[-1] == 1:
                image = image.transpose([2, 0, 1])

        elif len(image.shape) == 2:
            image = image[None, :]
        else:
            # not an image
            return image

        return image

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = self.fix(sample[key])
        return sample


class grayscale:
    def __init__(self, keys, channels):
        self.keys = keys
        self.channels = channels

    def convert2grayscale(self, image):
        # Convert to grayscale
        # Input should be a tensor
        # (N C H W) or (C H W) or (H W)
        try:
            shape = image.shape
        except:
            return image
        if len(image.shape) == 4:
            r, g, b = image.unbind(dim=1)
            l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(image.dtype)
            l_img = l_img.unsqueeze(dim=1)
            if self.channels != 1:
                l_img = l_img.repeat([1, self.channels, 1, 1])

        elif len(image.shape) == 3:
            r, g, b = image.unbind(dim=0)
            l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(image.dtype)
            l_img = l_img.unsqueeze(dim=0)
            if self.channels != 1:
                l_img = l_img.repeat([self.channels, 1, 1])
            # image = image.mean(0, keepdims=True)
        elif len(image.shape) == 2:
            l_img = image[None, :]
        else:
            return image
            # raise Exception("grayscale failed")

        return l_img

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = self.convert2grayscale(sample[key])
        return sample


class add_gaussian_noise:
    def __init__(self, max_sigma=0.02, sigma_dc=0.005, mu=0, keys=(), astype=torch.float32):
        self.max_sigma = max_sigma  # abit much maybe 0.04 best0.04+0.01
        self.sigma_dc = sigma_dc
        self.mu = mu
        self.active_keys = keys
        self.astype = astype

    def add_noise(self, sample):
        if type(sample) != torch.Tensor:
            sample = torch.from_numpy(sample)

        shape = sample.shape
        sigma = np.random.rand() * self.max_sigma + self.sigma_dc
        g_noise = torch.empty(shape).normal_(mean=self.mu, std=sigma).to(self.astype).to(sample.device)
        ret = sample + g_noise
        # ret = ret / torch.max(ret)
        # ret = torch.maximum(ret, torch.zeros_like(ret))
        ret = torch.clamp(ret, 0, 1)
        return ret, torch.tensor([sigma])

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key], sample[f'{key}_gauss_param'] = self.add_noise(sample[key])
        return sample


class add_poisson_noise:
    def __init__(self, peak=1000, peak_dc=50, keys=(), astype=torch.float32):
        super().__init__()
        self.PEAK = peak  # np.random.rand(1) * 1000 + 50
        self.PEAK_DC = peak_dc
        self.astype = astype
        self.active_keys = keys

    def add_noise(self, sample):
        if type(sample) != torch.Tensor:
            sample = torch.from_numpy(sample)
        peak = np.random.rand() * self.PEAK + self.PEAK_DC
        if peak < 0:
            return sample, torch.tensor([0])
        p_noise = torch.poisson(torch.clamp(sample, min=1e-6) * peak)  # poisson cannot take negative
        p_noise = p_noise.to(sample.device)
        # ret = p_noise
        ret = (p_noise.to(self.astype) / peak)  # poisson noise is not additive
        # ret = ret / torch.max(ret)
        # ret = torch.maximum(ret, torch.zeros_like(ret))
        ret = torch.clamp(ret, 0, 1)
        return ret, torch.tensor([peak])

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key], sample[f'{key}_poisson_param'] = self.add_noise(sample[key])
        return sample


class add_perlin_noise:
    def __init__(self, keys=(), refresh_noise_for_each=True, astype=torch.float32):
        super().__init__()
        self.active_keys = keys
        self.astype = astype
        self.refresh_noise_for_each = refresh_noise_for_each
        self.p = Perlin(frequency=6, octaves=10, persistence=0.6, lacunarity=2, seed=0)
        self.gradient = grayscale_gradient()
        self.render = RenderImage(light_enabled=True, light_contrast=3, light_brightness=2)
        self.lx, self.ux = 100, 200
        self.lz, self.uz = 100, 200
        self.noise_min = 0.8
        self.noise_max = 1.0

        self.nm = None
        self.tmp_h, self.tmp_w = -1, -1

    def add_noise(self, image):
        h, w = image.shape[-2:]
        if self.nm is None or h != self.tmp_h or w != self.tmp_w or self.refresh_noise_for_each:
            self.p = Perlin(frequency=6, octaves=10, persistence=0.6, lacunarity=2, seed=np.random.randint(1e6))
            self.nm = noise_map_plane_gpu(width=w, height=h,
                                          lower_x=self.lx, upper_x=self.ux,
                                          lower_z=self.lz, upper_z=self.uz,
                                          source=self.p)
            self.tmp_h = h
            self.tmp_w = w
        noise: PIL.Image = self.render.render(w, h, self.nm, 'remove_img_save_from_source.png', self.gradient)
        if noise is None:
            raise Exception("!!!!!!!!!! Return image in noiseutil.py in pynoise package !!!!!")
        noise = pil_to_tensor(noise)
        if image.max() <= 1.0:
            noise = noise.to(torch.float32).to(image.device)
            noise /= 255
            noisy = image * ((noise*(self.noise_max - self.noise_min)) + self.noise_min)
        else:
            noise = noise * (int(255 * self.noise_max) - int(255*self.noise_min)) + int(255*self.noise_min)
            noisy = image * noise // 255
            # noisy[noisy < noise//2] = 255  # avoid overflowing
        return noisy

    def __call__(self, sample):
        for key in self.active_keys:
            sample[key] = self.add_noise(sample[key])
        return sample


class to_tensor:
    def __call__(self, sample, astype=torch.float32):
        for key in sample.keys():
            if type(sample[key]) != torch.Tensor:
                if type(sample[key]) == np.ndarray:
                    sample[key] = torch.from_numpy(sample[key]).to(astype)
        return sample


class padding:
    def __init__(self, h, w, keys, mode='reflect'):
        super().__init__()
        self.h = h
        self.w = w
        self.keys = keys
        self.mode = mode

    def __call__(self, sample, astype=torch.float32):
        for key in self.keys:
            sample[key] = crop_arr(sample[key], self.h, self.w, mode=self.mode).to(astype)
        return sample


class normalize:
    def __call__(self, sample, astype=torch.float32):
        for key in sample.keys():
            sample[key] = (sample[key] / torch.max(sample[key])).to(astype)
        return sample


class rotate:
    def __init__(self, keys, angle):
        self.ops = []
        self.keys = keys
        self.angle = angle

    def rotate(self, image):
        return TF.rotate(image, self.angle)

    def __call__(self, sample, astype=torch.float32):
        for key in self.keys:
            sample[key] = self.rotate(sample[key]).to(astype)
        return sample


class augment:
    def __init__(self, keys, image_shape, horizontal_flip=True, resize_crop=True):
        self.ops = []
        self.keys = keys
        if horizontal_flip:
            self.ops.append(transforms.RandomHorizontalFlip())
        if resize_crop:
            self.ops.append(transforms.RandomResizedCrop(image_shape, antialias=True))

    def aug(self, image):
        for op in self.ops:
            image = op(image)
        return image

    def __call__(self, sample, astype=torch.float32):
        for key in self.keys:
            sample[key] = self.aug(sample[key]).to(astype)
        return sample


class translate_image:
    def __init__(self, translate_by, keys):
        self.translate_by = translate_by
        self.keys = keys

    def translate(self, image, idx):
        translate = self.translate_by[idx % len(self.translate_by)]
        translated = torch.zeros_like(image)
        c, h, w = image.shape
        hs, he, ws, we = max(0, translate[0]), min(h, translate[0] + h), max(0, translate[1]), min(w, translate[1] + w)
        ihs = abs(min(0, translate[0]))
        iws = abs(min(0, translate[1]))
        ihe = ihs + (he - hs)
        iwe = iws + (we - ws)
        translated[:, hs:he, ws:we] = image[:, ihs:ihe, iws:iwe]
        return translated

    def __call__(self, sample, astype=torch.float32):
        for key in self.keys:
            sample[key] = self.translate(sample[key], sample['idx']).to(astype)
        return sample


def crop_arr(arr, h, w, mode='constant'):  # todo: this is too slow
    hw, ww = arr.shape[-2:]
    do_pad = False
    istorch = type(arr) == torch.Tensor or type(arr) == torch.nn.Parameter
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

def crop_arr_new(arr, h, w, mode='constant'):
    hw, ww = arr.shape[-2:]
    istorch = type(arr) == torch.Tensor or type(arr) == torch.nn.Parameter
    if istorch:
        out = torch.zeros((*arr.shape[:-2], h, w), dtype=arr.dtype, device=arr.device)
    else:
        out = np.zeros((*arr.shape[:-2], h, w), dtype=arr.dtype)
    arr_idx = 0, hw, 0, ww
    top = h//2 - hw//2
    left = w//2 - ww//2
    bottom = top + hw
    right = left + ww
    if top < 0:
        arr_idx[0] = -top
        top = 0
    if left < 0:
        arr_idx[2] = -left
        left = 0
    if bottom > h:
        arr_idx[1] = hw - (bottom - h)
        bottom = h
    if right > w:
        arr_idx[3] = ww - (right - w)
        right = w
    out[..., top:bottom, left:right] = arr[..., arr_idx[0]:arr_idx[1], arr_idx[2]:arr_idx[3]]
    return out

def crop_from_center(sample: dict, ref_key: str, crop_keys: list):
    """
    Crop crop_key in sample wrt to ref_key from the center. h, w should be the last 2 dimensions
    @param sample: dict with arrays
    @param ref_key: reference array
    @param crop_keys: keys that should be cropped
    @return:
    """
    ref = sample[ref_key]
    hx, wx = ref.shape[-2:]

    for crop_key in crop_keys:
        crp = sample[crop_key]
        sample[crop_key] = crop_arr(crp, hx, wx)
    return sample


if __name__ == "__main__":
    img = cv2.imread("../dataset/test/0.png")
    plt.imshow(img)
    plt.show()
    for i in range(3):
        sam = {'gt': torch.from_numpy(img.transpose([2, 0, 1]) / 255)}
        transform = add_perlin_noise(keys=('gt',), )
        sam = transform(sam)
        plt.imshow((sam['gt'].numpy().transpose([1, 2, 0]) * 255).astype(np.uint8))
        plt.show()
