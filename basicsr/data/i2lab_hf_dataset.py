import numpy as np
import torch
import torchvision.transforms
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


class HuggingFaceDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        dataset_name, split = opt['name'], opt['split']
        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        gt_path = f"gt-{idx}.png"
        lq_path = f"lq-{idx}.png"
        img_gt, img_lq = np.array(item['gt']), np.array(item['blur'])

        # Perform any necessary preprocessing here
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        # BGR to RGB, HWC to CHW, numpy to tensor
        to_tensor = torchvision.transforms.ToTensor()
        img_gt, img_lq = to_tensor(img_gt).to(torch.float32), to_tensor(img_lq).to(torch.float32)

        return {'gt_path': gt_path, 'lq_path': lq_path, 'gt': img_gt, 'lq': img_lq}
