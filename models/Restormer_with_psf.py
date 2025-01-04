import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from Restormer.models.archs.restormer_arch import Downsample, OverlapPatchEmbed, TransformerBlock, Upsample
from models.NAFNet import prior_upsampling
from models.blocks import SPADEResnetBlock
from models.encoder import ConvEncoder


##########################################################################
##---------- Restormer -----------------------
class RestormerK(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        **kwargs
    ):

        super(RestormerK, self).__init__()

        self.ad1_list = nn.ModuleList()
        self.ad1_list.append(SPADEResnetBlock(dim, dim, "spectralspadesyncbatch3x3", label_nc=dim))
        self.ad1_list.append(SPADEResnetBlock(dim*2**1, dim*2**1, "spectralspadesyncbatch3x3", label_nc=dim*2**1))
        self.ad1_list.append(SPADEResnetBlock(dim*2**2, dim*2**2, "spectralspadesyncbatch3x3", label_nc=dim*2**2))
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, latent_list):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        out_enc_level3_new = self.ad1_list[-1](latent_list[-1], out_enc_level3)
        # print(f"out_enc_level3_new: {out_enc_level3_new.shape} <= latent_list[-1]: {latent_list[-1].shape}, out_enc_level3: {out_enc_level3.shape}")
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3_new], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_enc_level2_new = self.ad1_list[-2](latent_list[-2], out_enc_level2)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2_new], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_enc_level1_new = self.ad1_list[-3](latent_list[-3], out_enc_level1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_new], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1


class RestormerKConfig(PretrainedConfig):
    model_type = "RestormerK"

    def __init__(self, psfs=None, **kwargs):
        super().__init__(**kwargs)
        self.psfs = psfs
        

class RestormerKModel(PreTrainedModel):
    config_class = RestormerKConfig

    def __init__(self, config):
        super().__init__(config)
        self.psfs = nn.Parameter(torch.from_numpy((np.array(config.psfs, dtype=np.float32)+1)/2), requires_grad=False)

        ndf = config.dim*3//(2**3)
        self.net_prior_psfs = ConvEncoder(inc=self.psfs.shape[-3]*self.psfs.shape[0], ndf=ndf)  # output is ndf*32
        self.prior_upsampling_psfs = prior_upsampling([ndf*32, ndf*32, ndf*32, 192, 96, 48]) # need 2 more upsampling layers before 192

        self.model = RestormerK(**config.to_dict()) 
        self.init_weights()

    def forward(self, x):
        # print(f"x: {x.shape} {self.psfs.shape}" )
        psfs = einops.rearrange(self.psfs, 'b c h w -> 1 (b c) h w')
        psfs = F.interpolate(psfs, (x.shape[-2], x.shape[-1]))
        prior_z_psfs = self.net_prior_psfs(psfs)
        latent_list_inverse_psfs = self.prior_upsampling_psfs(prior_z_psfs)[:-2] #  drop last 2 layers
        # print(f"latent_list_inverse_psfs: {[s.shape for s in latent_list_inverse_psfs]}")
        return self.model(x, latent_list_inverse_psfs)
