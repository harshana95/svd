import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from models.blocks import ResnetBlock, SPADEResnetBlock

from LD.models.archs.normalization import get_nonspade_norm_layer


class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, inc=3, kw=3, ndf=64):
        super().__init__()

        pw = int(np.ceil((kw - 1.0) / 2))
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(inc, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))
        self.res_0 = ResnetBlock(ndf*16, nn.LeakyReLU(0.2, False))
        self.res_1 = ResnetBlock(ndf*16, nn.LeakyReLU(0.2, False))
        self.res_2 = ResnetBlock(ndf*16, nn.LeakyReLU(0.2, False))
        activation=nn.LeakyReLU(0.2,False)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ndf * 16, ndf * 32, kernel_size=kw)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ndf * 32, ndf * 32, kernel_size=kw)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(ndf * 32, ndf * 32, kernel_size=kw)),
            activation
        )
        self.down = nn.AvgPool2d(2,2)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        
    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x) # 128
        x = self.conv_7x7(self.pad_3(self.actvn(x)))
        x = self.layer2(self.actvn(x)) # 64
        x = self.layer3(self.actvn(x)) # 32
        x = self.layer4(self.actvn(x)) # 16 
        x = self.layer5(self.actvn(x)) # 8
        
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        mu = self.out(x)

        return mu
