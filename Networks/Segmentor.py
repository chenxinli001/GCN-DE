# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:53:19 2020

@author: Sunly
"""

from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
import torch
import torch.nn.functional as F

from Networks.UNET3D_Query import UNet3DQ
from Networks.UNET3D_Support import UNet3DS

class SegMenTor(Module):

    def __init__(self, num_channels=1, feat_channels=[16, 32, 64]):
        
        # Initialize Support Image Semgnetor and Query Image Segmentor

        super(SegMenTor, self).__init__()
        
        # Encoder downsamplers
        self.SuppSeg = UNet3DS()
        self.SuppQuery = UNet3DQ()
        


    def forward(self, support_image, query_image, support_label):
        
        _,SA_enc1,SA_enc2,SA_enc3,SA_dec2,SA_dec1 = self.SuppSeg(support_image,support_label)
        SuppAttention = [SA_enc1,SA_enc2,SA_enc3,SA_dec2,SA_dec1]
        seg = self.SuppQuery(query_image,SuppAttention)
        
        return seg
