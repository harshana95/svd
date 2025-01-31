import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import actFunc, conv1x1, conv3x3, conv5x5
from models.modules.CAB import CAB
from models.modules.RDB import RDB

    
class EncodeCell(nn.Module):
    def __init__(self, n_features, in_channel=3):
        super(EncodeCell, self).__init__()
        self.n_feats = n_features
        self.conv = conv5x5(in_channel, self.n_feats, stride=1)
        self.down1 = conv5x5(self.n_feats, 2*self.n_feats, stride=2)
        self.down2 = conv5x5(2*self.n_feats, 4*self.n_feats, stride=2)
        self.down3 = conv5x5(4*self.n_feats, 8*self.n_feats, stride=2)
        self.enc_l1 = RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3)
        self.enc_l2 = RDB(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3)
        self.enc_l3 = RDB(in_channels=4 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3)
        self.enc_h = RDB(in_channels=8 * self.n_feats, growthRate=self.n_feats * 2, num_layer=3)

    def forward(self, x):
        '''
        out1: torch.Size([B, 16, 256, 256])
        out2: torch.Size([B, 32, 128, 128])
        out3: torch.Size([B, 64, 64, 64])
        h: torch.Size([B, 128, 32, 32])
        '''
        b,c,h,w = x.shape
        
        out1 = self.enc_l1(self.conv(x))
        out2 = self.enc_l2(self.down1(out1))
        out3 = self.enc_l3(self.down2(out2))
        h = self.enc_h(self.down3(out3))
        
        return out1, out2, out3, h
        
class DecodeCell(nn.Module):
    def __init__(self, n_features, out_dim=3):
        super(DecodeCell, self).__init__()
        self.n_feats = n_features
        self.uph = nn.ConvTranspose2d(8 * self.n_feats, 4 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        self.fusion3 = CAB(8*self.n_feats, [1,3])
        
        self.up3 = nn.ConvTranspose2d(8*self.n_feats, 2*self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)    
        self.fusion2 = CAB(4*self.n_feats, [1,3]) 
        
        self.up2 = nn.ConvTranspose2d(4*self.n_feats, self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1)                
        self.fusion1 = CAB(2*self.n_feats, [1,3])   
        
        self.output = nn.Sequential(
            conv3x3(2*self.n_feats, self.n_feats, stride=1),
            conv3x3(self.n_feats, out_dim, stride=1)
        )

    def forward(self, h, x3, x2, x1):
        # channel: 8, 4, 2 * n_feat
        
        h_decode = self.uph(h)
        x3 = self.fusion3(torch.cat([h_decode, x3], dim=1))
        
        x3_up = self.up3(x3)
        x2 = self.fusion2(torch.cat([x3_up, x2], dim=1))
        
        x2_up = self.up2(x2)
        x1 = self.fusion1(torch.cat([x2_up, x1], dim=1))
        
        return self.output(x1)