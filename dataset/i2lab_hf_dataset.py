import numpy as np
import torch
import torchvision.transforms
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from utils.dataset_utils import crop_center


class HuggingFaceDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        dataset_name, split = opt['name'], opt['split']
        self.dataset = load_dataset(dataset_name, split=split)
        self.rand_crop = crop_center(self.opt['gt_size'], self.opt['gt_size'])
        self.to_tensor = torchvision.transforms.ToTensor()

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
        # random crop
        img_gt, img_lq = self.rand_crop.crop(img_gt), self.rand_crop.crop(img_lq)
        
        return {'gt_path': gt_path, 'lq_path': lq_path, 'gt': img_gt, 'lq': img_lq}
