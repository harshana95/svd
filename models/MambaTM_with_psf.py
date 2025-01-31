import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import argparse, math
from transformers import PreTrainedModel, PretrainedConfig

from models.MambaTM import MambaTMNetwork
from models.WienerDeconv import WienerDeconvolutionConfig, WienerDeconvolutionModel
from models.modules.Unets import DecodeCell, EncodeCell

class MambaTMConfig(PretrainedConfig):
    model_type = "MambaTM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.psfs = kwargs.get("psfs", None)
        self.weights = kwargs.get("weights", None)
        self.n_features = kwargs.get("n_features", 16)
        self.n_blocks  = kwargs.get("n_blocks", 6)
        self.input_size = kwargs.get("input_size",(192, 192, 16))
        self.output_last_only = kwargs.get("output_last_only", False)
    
class MambaTMModel(PreTrainedModel):
    config_class = MambaTMConfig

    def __init__(self, config):
        super().__init__(config)
        
        psfs = np.array(config.psfs, dtype=np.float32)
        weights = np.array(config.weights, dtype=np.float32)
        psfs = einops.rearrange(psfs, '1 c t h w -> 1 t c h w')
        weights = einops.rearrange(weights, '1 c t h w -> 1 t c h w')
        print(f"psfs {psfs.shape} weights {weights.shape}")

        self.psfs = nn.Parameter(torch.from_numpy(psfs).contiguous(), requires_grad=False)#.contiguous()
        self.weights = nn.Parameter(torch.from_numpy(weights).contiguous(), requires_grad=False)#.contiguous()
        self.weight_scaling = nn.Parameter(torch.ones([1, weights.shape[1], 1, 1, 1]), requires_grad=True)
        n_psfs = self.psfs.shape[1]
        n_channels = self.psfs.shape[2]
        n_features = 64

        self.model = MambaTMNetwork(**config.to_dict(), ref_channels=n_channels, ref_steps=n_psfs)

        # self.psfenc = EncodeCell(n_features, in_channel=n_channels)
        # self.psfdec = DecodeCell(n_features, out_dim=n_channels)
        # self.weienc = EncodeCell(n_features, in_channel=n_channels)
        # self.weidec = DecodeCell(n_features, out_dim=n_channels)
        self.init_weights()

    def forward(self, y, Y):
        # B = y.shape[0]
        # psfs = self.psfs
        # weights = self.weights
        # if self.psfs.shape[0] != B:
        #     psfs = einops.repeat(psfs, '1 t c h w -> b t c h w', b=B)
        #     weights = einops.repeat(weights, '1 t c h w -> b t c h w', b=B)
        # psfs = einops.rearrange(psfs, 'b t c h w -> (b t) c h w')
        # weights = einops.rearrange(weights, 'b t c h w -> (b t) c h w')
        
        # out1, out2, out3, h1 = self.psfenc(psfs)
        # psfs_hat = self.psfdec(h1, out3, out2, out1)
        # out4, out5, out6, h2 = self.weienc(weights)
        # weights_hat = self.weidec(h2, out6, out5, out4)
        # latent = [out1+out4,out2+out5,out3+out6,h1+h2]

        X, LPD = self.model(torch.cat([y[:, None], Y], dim=1))  # if return last, b c h w
        return X, LPD 
        