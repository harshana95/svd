import numpy as np
import torch
import torchvision.transforms
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from utils.dataset_utils import add_poisson_noise, add_gaussian_noise, crop_center


class HuggingFaceDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        dataset_name, split = opt['name'], opt['split']
        
        self.dataset = load_dataset(dataset_name, split=split)
        self.center_crop = crop_center(self.opt['gt_size'], self.opt['gt_size'])
        self.to_tensor = torchvision.transforms.ToTensor()
        gaussian_noise = opt.get('gaussian_noise', None)
        poisson_noise = opt.get('poisson_noise', None)
        resize = opt.get('resize', None)
        if gaussian_noise:
            print("Adding Gaussian Noise", gaussian_noise)
            self.noise_g = add_gaussian_noise(keys=['lq'], **gaussian_noise)
        else:
            self.noise_g = lambda x: x
        if poisson_noise:
            print("Adding Poisson Noise", poisson_noise)
            self.noise_p = add_poisson_noise(keys=['lq'], **poisson_noise)
        else:
            self.noise_p = lambda x: x
        if resize:
            print("Resizing to", resize)
            self.resize = torchvision.transforms.Resize(resize, antialias=True)
        else:
            self.resize = lambda x: x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        gt_path = f"gt-{idx}.png"
        lq_path = f"lq-{idx}.png"
        img_gt, img_lq = np.array(item['gt']), np.array(item['blur'])

        # Perform any necessary preprocessing here
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = self.to_tensor(img_gt).to(torch.float32), self.to_tensor(img_lq).to(torch.float32)
        # crop
        img_gt, img_lq = self.center_crop.crop(img_gt), self.center_crop.crop(img_lq)
        # scale
        img_gt, img_lq = self.resize(img_gt), self.resize(img_lq)

        item = {'gt_path': gt_path, 'lq_path': lq_path, 'gt': img_gt, 'lq': img_lq}
        item = self.noise_p(item)
        item = self.noise_g(item)
        return item
