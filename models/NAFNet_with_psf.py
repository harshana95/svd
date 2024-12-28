import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from models.NAFNet import prior_upsampling
from models.blocks import NAFBlock, SPADEResnetBlock, Up_ConvBlock
from models.encoder import ConvEncoder

class NAFKNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ad1_list = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.ad1_list.append(
                SPADEResnetBlock(chan, chan, "spectralspadesyncbatch3x3", label_nc=chan)
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, latent_list):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, ad, enc_skip, lat in zip(self.decoders, self.ups, self.ad1_list, encs[::-1], latent_list[::-1]):
            tmp = ad(enc_skip, lat) # added
            x = up(x)
            x = x + tmp #enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFKNetConfig(PretrainedConfig):
    model_type = "NAFKNet"

    def __init__(self, psfs=None, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], **kwargs):
        super().__init__(**kwargs)
        self.psfs = psfs
        self.img_channel = img_channel
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums


class NAFKNetModel(PreTrainedModel):
    config_class = NAFKNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.psfs = nn.Parameter(torch.from_numpy((np.array(config.psfs, dtype=np.float32)+1)/2), requires_grad=False)
        channels = [config.width * (2 ** i) for i in range(len(config.enc_blk_nums)+2)]
        # print(f"upsampling channels: {channels}")

        self.net_prior = ConvEncoder(ndf=config.width)  # output is ndf*32
        self.prior_upsampling = prior_upsampling(channels[::-1])
        
        self.net_prior_psfs = ConvEncoder(inc=self.psfs.shape[-3], ndf=config.width)  # output is ndf*32
        self.prior_upsampling_psfs = prior_upsampling(channels[::-1])
        
        self.model = NAFKNet(img_channel=config.img_channel, width=config.width, 
                            middle_blk_num=config.middle_blk_num, enc_blk_nums=config.enc_blk_nums, dec_blk_nums=config.dec_blk_nums)
        self.init_weights()

    def forward(self, x):
        prior_z = self.net_prior(x)
        latent_list_inverse = self.prior_upsampling(prior_z)
        prior_z_psfs = self.net_prior_psfs(self.psfs)
        latent_list_inverse_psfs = self.prior_upsampling_psfs(prior_z_psfs)

        latent_list_inverse = latent_list_inverse[:-1]
        latent_list_inverse_psfs = latent_list_inverse_psfs[:-1]
        for i in range(len(latent_list_inverse)):
            latent_list_inverse[i] = (latent_list_inverse[i] + latent_list_inverse_psfs[i])/2
        return self.model(x, latent_list_inverse)
        
