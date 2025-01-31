from models.modules.Unets import DecodeCell, EncodeCell
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
from models.blocks import ResnetBlock, SPADEResnetBlock, Up_ConvBlock
from models.blocks import UNetConvBlock, UNetUpBlock, conv3x3
from models.encoder import ConvEncoder


class HINet(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(HINet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.ad1_list = nn.ModuleList()

        prev_channels = self.get_input_chn(wf)
        norm_G = "spectralspadesyncbatch3x3"
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.ad1_list = self.ad1_list[0:-1]
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, in_chn)
        

    def forward(self, x, latent_list):
        image = x
        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        # print("x1", x1.shape)
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                # print("i--spade", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i--spade output", i, x1.shape)
                x1, x1_up = down(x1) # 64, 128, 128 -- 64, 256, 256
                # print("i", i, x1.shape, x1_up.shape)
                
                encs.append(x1_up)
            else:
                # print("i", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i spade", i, x1.shape)
                x1 = down(x1) # 2048, 8, 8
                # print("i - nodown", i, x1.shape)
                # x1 = self.ad1_list[-1](x1, latent_list[-1])
                

        for i, up in enumerate(self.up_path_1):
            # temps = self.skip_conv_1[i](encs[-i-1])
            # (8,8) ---- (1024,16,16) --- (16,16)
            # print("i temps2 input", i, encs[-i-1].shape, latent_list[-2-i].shape)
            temps2 = self.ad1_list[-1-i](encs[-i-1], latent_list[-1-i])
            # print("i, temps shape", i, x1.shape, encs[-i-1].shape, temps.shape, temps2.shape)
            x1 = up(x1, temps2)
            decs.append(x1)
        out = self.last(x1)
        out = out + image
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class prior_upsampling(nn.Module):
    def __init__(self, wf=64):
        super(prior_upsampling, self).__init__()
        # self.conv_latent_init = Up_ConvBlock(4 * wf, 32 * wf)
        self.conv_latent_up2 = Up_ConvBlock(32 * wf, 16 * wf) 
        self.conv_latent_up3 = Up_ConvBlock(16 * wf, 8 * wf)
        self.conv_latent_up4 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up5 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up6 = Up_ConvBlock(2 * wf, 1 * wf)
    def forward(self, z):
        # latent_1 = self.conv_latent_init(z) # 8, 8
        latent_2 = self.conv_latent_up2(z) # 16
        latent_3 = self.conv_latent_up3(latent_2) # 32
        latent_4 = self.conv_latent_up4(latent_3) # 64
        latent_5 = self.conv_latent_up5(latent_4) # 128
        latent_6 = self.conv_latent_up6(latent_5) # 256
        latent_list = [latent_6,latent_5,latent_4,latent_3]
        return latent_list
    
class MSDI3(nn.Module):
    def __init__(self):
        super(MSDI3, self).__init__()        
        wf = 64
        self.net_prior = ConvEncoder()
        self.prior_upsampling = prior_upsampling(wf=wf)
        # self.net_prior_psfs = ConvEncoder(inc=self.psfs.shape[-3])
        # self.prior_upsampling_psfs = prior_upsampling()                   
        self.inverse_generator = HINet(wf=wf)
        self.generator = HINet(wf=wf)


    def forward(self, x, y, additional_latent_list=None):
        prior_z = self.net_prior(x)
        latent_list_inverse = self.prior_upsampling(prior_z)
        
        # prior_z_psfs = self.net_prior_psfs(self.psfs)
        # latent_list_inverse_psfs = self.prior_upsampling_psfs(prior_z_psfs)
        if additional_latent_list:
            for i in range(len(latent_list_inverse)):
                latent_list_inverse[i] = (latent_list_inverse[i] + additional_latent_list[i])

        out = self.generator(x, latent_list_inverse)
        if y is None:
            out_inverse = None
        else:
            out_inverse = self.inverse_generator(y, latent_list_inverse)    
        return out, out_inverse


    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class MSDI3Config(PretrainedConfig):
    model_type = "MSDI3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.psfs = kwargs.get('psfs')  # pass the psfs as a list
        self.weights = kwargs.get('weights')

class MSDI3Model(PreTrainedModel):
    config_class = MSDI3Config

    def __init__(self, config):
        super().__init__(config)
        self.psfs = nn.Parameter(torch.from_numpy((np.array(self.config.psfs, dtype=np.float32)+1)/2), requires_grad=False)
        self.weights = nn.Parameter(torch.from_numpy(np.array(self.config.weights, dtype=np.float32)), requires_grad=False)
        print(f"psfs: {self.psfs.shape} {self.psfs.min()}, {self.psfs.max()}")
        print(f"weights: {self.weights.shape} {self.weights.min()}, {self.weights.max()}")

        self.model = MSDI3()
        n_psfs = self.psfs.shape[-3]  # 3 x N
        n_features = 64
        self.psfenc = EncodeCell(n_features, in_channel=n_psfs)
        self.psfdec = DecodeCell(n_features, out_dim=n_psfs)
        self.weienc = EncodeCell(n_features, in_channel=n_psfs)
        self.weidec = DecodeCell(n_features, out_dim=n_psfs)
        self.init_weights()

    def forward(self, x, y=None):
        out1, out2, out3, h1 = self.psfenc(self.psfs)
        psfs_hat = self.psfdec(h1, out3, out2, out1)
        out4, out5, out6, h2 = self.weienc(self.weights)
        weights_hat = self.weidec(h2, out6, out5, out4)
        # n*1, n*2, n*4, n*8
        x_hat, y_hat = self.model(x, y, [out1+out4,out2+out5,out3+out6,h1+h2])
        return x_hat, y_hat, psfs_hat, weights_hat
        
