import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from inspect import isfunction

import yaml

from utils.dataset_utils import crop_arr

class DictAsMember(dict):

    def __getattr__(self, name):
        if name not in self.keys():
            # print(f"'{name}' not in {list(self.keys())}")
            return None
        value = self[name]
        return value


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return DictAsMember(loader.construct_pairs(node))

    Dumper.add_representer(DictAsMember, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse_args(parser):
    args, kwargs = parser.parse_known_args()
    for i, arg in enumerate(kwargs):
        if arg.startswith(("-", "--")):
            _type = str
            if i < len(kwargs) - 1:
                try:
                    tmp = float(kwargs[i + 1])
                    _type = float
                    if float(kwargs[i + 1]) == int(kwargs[i + 1]):
                        _type = int
                except:
                    pass
            parser.add_argument(arg, type=_type)
    args = parser.parse_args()
    return args

def normalized_rescale(arr, h, w):
    to_numpy = False
    if type(arr) != torch.Tensor:
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

def generate_folder(p):
    if not os.path.exists(p):
        os.makedirs(p)



def copy_folder_contents(src_folder, dst_folder):
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Iterate over all files and folders in the source directory
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        # Copy files and directories
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def get_basename(p):
    tmp = os.path.basename(p)
    while tmp == '':
        p = os.path.dirname(p)
        tmp = os.path.basename(p)
    return tmp


def get_from_file(data, idx):
    return cv2.imread(data[idx]).astype('float32') / 255.


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def dict_add(d1, d2):
    if d1 == {}:
        d = d2
    elif d2 == {}:
        d = d1
    else:
        d = d1
        keys = set(d1.keys())
        keys = keys.union(set(d2.keys()))
        for key in keys:
            if key in d.keys():
                d[key] += d2[key]
            else:
                d[key] = d2[key]
    return d


def hstack_images(images, h_dim=0, w_dim=1):
    h, w = 0, 0
    for i in range(len(images)):
        h = max(h, images[i].shape[h_dim])
        w = max(w, images[i].shape[w_dim])
    for i in range(len(images)):
        images[i] = crop_arr(images[i].transpose([2, 0, 1]), h, w, mode='constant').transpose([1, 2, 0])
    return (np.clip(np.hstack(images), 0, 1) * 255).astype(np.uint8)


def display_images(d, same_plot=None, independent_colorbar=False, row_limit=-1, cols_per_plot=1, size=1):
    im = None
    if same_plot is None:
        same_plot = []
    max_rows = 0
    vmin = 1e10
    vmax = -1e10
    # find vmax, vmin, max rows
    for k in d.keys():
        # if max_rows > 0:
        #     assert max_rows == len(d[k]), "Lengths of each array should be same"
        max_rows = max(max_rows, len(d[k]))
        if max_rows == 0:
            print("Cannot display images! Zero sized array. Skipping...")
            return
        if len(d[k].shape) == 3:
            vmin = min(vmin, d[k].min())
            vmax = max(vmax, d[k].max())
    if independent_colorbar:
        vmin, vmax = None, None

    # set key map. if keys should be in the same plot, they will be grouped by a list
    keys = [key for key in d.keys()]
    plotmap = [same for same in same_plot]
    for same in same_plot:
        for key in same:
            keys.remove(key)
    for key in keys:
        plotmap.append([key])

    # start plotting
    rows, cols = max_rows, len(plotmap)
    cols *= cols_per_plot
    rows = int(np.ceil(rows / cols_per_plot))
    if row_limit > 0:
        rows = min(rows, row_limit)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * size, rows * size))
    show_legend = False
    for i in range(rows * cols):
        x, y = i // cols, i % cols
        if rows * cols == 1:
            ax = axes
        else:
            ax = axes.flat[i]
        for key in plotmap[y // cols_per_plot]:
            if x * cols_per_plot + (y % cols_per_plot) >= len(d[key]):
                ax.axis(False)
                continue
            data = d[key][x * cols_per_plot + (y % cols_per_plot)]
            if len(data.shape) == 2:
                im = ax.imshow(data, vmin=vmin, vmax=vmax)
                ax.axis(False)
            elif len(data.shape) == 3:
                im = ax.imshow(data)
                ax.axis(False)
            else:
                ax.plot(data[:], label=key)
                show_legend = True
            if x == 0:
                ax.set_title(f"{key}")
            if y == 0:
                ax.set_ylabel(f"{x}")
        if x == 0 and len(data.shape) != 2 and show_legend:
            ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if im is not None and not independent_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.35, 0.01, 0.4])
        fig.colorbar(im, cax=cbar_ax)
    return fig


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot_timesteps(imgs, image_ori=None, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image_ori] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


class AverageMeter(object):
    """
    computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
