import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import argparse, math
from transformers import PreTrainedModel, PretrainedConfig

from models.modules.Unets import DecodeCell, EncodeCell
from models.modules.Hilbert3d import Hilbert3d
from models.modules.mambablock import MambaLayerglobal, MambaLayerglobalRef, MambaLayerlocal, MambaLayerlocalRef


class MambaTM_noLPD(nn.Module):
    def __init__(self, n_features=16, n_blocks=6, input_size=(192, 192, 16), output_last_only=True, **kwargs):
        super(MambaTM_noLPD, self).__init__()
        self.n_feats = n_features
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.output_last_only = output_last_only
        
        self.encoder = EncodeCell(n_features)
        H, W, T = input_size
        self.set_h_curve(H, W, T, "cuda")
        self.SGlobalMambaBlocks = nn.ModuleList()
        self.TGlobalMambaBlocks = nn.ModuleList()
        self.LocalMambaBlocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.SGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8))
            self.TGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8, spatial_first=False))
            self.LocalMambaBlocks.append(MambaLayerlocal(dim=self.n_feats*8))
        
        self.decoder = DecodeCell(n_features)
    
    def set_h_curve(self, H, W, T, device):
        SH = math.ceil(H/8)
        SW = math.ceil(W/8)
        h_curve_small_list = list(Hilbert3d(width=SW, height=SH, depth=T))
        h_curve_small = torch.tensor(h_curve_small_list).long().to(device)
        self.h_curve = h_curve_small[:, 0] * SW * T + h_curve_small[:, 1] * T + h_curve_small[:, 2]
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        if (H, W, T) != self.input_size or self.h_curve.device != x.device:
            self.set_h_curve(H, W, T, x.device)
            self.input_size = (H, W, T)

        enc1, enc2, enc3, h = self.encoder(x.contiguous().view(-1, C, H, W))
        h = rearrange(h, '(b t) c h w -> b t c h w', t=T)
        for i in range(self.n_blocks):
            h = self.SGlobalMambaBlocks[i](h)
            h = self.TGlobalMambaBlocks[i](h)
            h = self.LocalMambaBlocks[i](h, self.h_curve)
        h = rearrange(h, 'b t c h w -> (b t) c h w')
        output = self.decoder(h, enc3, enc2, enc1)
        output = output.view(B, T, C, H, W)
        if self.output_last_only:
            return output[:, -1]
        else:
            return output


class MambaTMNetwork(nn.Module):
    def __init__(self, n_features=16, n_blocks=6, input_size=(192, 192, 16), output_last_only=True, ref_channels=3, ref_steps=16, **kwargs):
        super(MambaTMNetwork, self).__init__()
        self.n_feats = n_features
        self.input_size = input_size
        self.n_blocks = n_blocks
        self.output_last_only = output_last_only
        self.n_ref_block = 1
        self.ref_channels = ref_channels  # number of channels in PSFs
        self.ref_steps = ref_steps  # number of PSFs
        assert self.n_ref_block < self.n_blocks
        
        self.encoder = EncodeCell(n_features)
        self.ref_encoder = EncodeCell(n_features, ref_channels)
        self.convert_ref = nn.Conv2d(n_features*8, n_features*8, 1)
        H, W, T = input_size
        self.set_h_curve(H, W, T, "cuda")
        self.SGlobalMambaBlocks = nn.ModuleList()
        self.TGlobalMambaBlocks = nn.ModuleList()
        self.LocalMambaBlocks = nn.ModuleList()
        for _ in range(self.n_blocks-self.n_ref_block):
            self.SGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8))
            self.TGlobalMambaBlocks.append(MambaLayerglobal(dim=self.n_feats*8, spatial_first=False))
            self.LocalMambaBlocks.append(MambaLayerlocal(dim=self.n_feats*8))
        for _ in range(self.n_ref_block):
            self.SGlobalMambaBlocks.append(MambaLayerglobalRef(dim=self.n_feats*8))
            self.TGlobalMambaBlocks.append(MambaLayerglobalRef(dim=self.n_feats*8, spatial_first=False))
            self.LocalMambaBlocks.append(MambaLayerlocalRef(dim=self.n_feats*8))    
        self.ref_decoder = DecodeCell(n_features, ref_channels)
        self.decoder = DecodeCell(n_features)
    
    def set_h_curve(self, H, W, T, device):
        SH = math.ceil(H/8)
        SW = math.ceil(W/8)
        h_curve = list(Hilbert3d(width=SW, height=SH, depth=T))
        h_curve = torch.tensor(h_curve).long().to(device)
        self.h_curve = h_curve[:, 0] * SW * T + h_curve[:, 1] * T + h_curve[:, 2]
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        if (H, W, T) != self.input_size or self.h_curve.device != x.device:
            self.set_h_curve(H, W, T, x.device)
            self.input_size = (H, W, T)
        enc1, enc2, enc3, h = self.encoder(x.contiguous().view(-1, C, H, W))
        h = rearrange(h, '(b t) c h w -> b t c h w', t=T)

        for i in range(self.n_blocks-self.n_ref_block):
            h = self.SGlobalMambaBlocks[i](h)
            h = self.TGlobalMambaBlocks[i](h)
            h = self.LocalMambaBlocks[i](h, self.h_curve)
            
        h_ref = self.convert_ref(h.flatten(0,1))
        LPD_hat = self.ref_decoder(h_ref, enc3, enc2, enc1)
        _, _, _, ref = self.ref_encoder(LPD_hat)
        ref = rearrange(ref, '(b t) c h w -> b t c h w', t=T)
        for i in range(self.n_blocks-self.n_ref_block, self.n_blocks):
            h = self.SGlobalMambaBlocks[i](h, ref)
            h = self.TGlobalMambaBlocks[i](h, ref)
            h = self.LocalMambaBlocks[i](h, ref, self.h_curve)
        h = rearrange(h, 'b t c h w -> (b t) c h w')  
        output = self.decoder(h, enc3, enc2, enc1)
        output = output.view(B, T, C, H, W)
        if self.output_last_only:
            return output[:, -1], LPD_hat.view(-1, self.ref_steps, self.ref_channels, H, W)
        else:
            return output, LPD_hat.view(-1, self.ref_steps, self.ref_channels, H, W)
        

class MambaTM_noLPDConfig(PretrainedConfig):
    model_type = "MambaTM_noLPD"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_features = kwargs.get("n_features", 16)
        self.n_blocks  = kwargs.get("n_blocks", 6)
        self.input_size = kwargs.get("input_size",(192, 192, 16))
        self.output_last_only = kwargs.get("output_last_only", True)
    
class MambaTM_noLPDModel(PreTrainedModel):
    config_class = MambaTM_noLPDConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MambaTM_noLPD(**config.to_dict())
        self.init_weights()

    def forward(self, x):
        # repeat the input images t times
        x = x[:, None]
        if self.config.input_size[-1] != 1:
            x = einops.repeat(x, 'b 1 c h w -> b t c h w', t=self.config.input_size[-1])
        
        return self.model(x)
        
