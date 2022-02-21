# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:01:02 2020

@author: Sunly
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:00:53 2020

@author: Sunly
"""

from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from torch.nn import Parameter
import torch.nn.functional as F
import torch

class UNet3DQ(Module):
    # __                          __
    #  1|__   ______________   __|1
    #     2|__  __________  __|2
    #        3|__  ____  __|3
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity 

    def __init__(self, num_channels=1, feat_channels=[16, 32, 64], residual=None):
        
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3DQ, self).__init__()
        
        # Encoder downsamplers
        self.pool1 = MaxPool3d((2,2,2))
        self.pool2 = MaxPool3d((2,2,2))
        
        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers

        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Activation function
        self.sigmoid = Sigmoid()
        
        # Query Attention Module
        
        self.QAM_enc1 = QueryAttentionModule(feat_channels[0])
        self.QAM_enc2 = QueryAttentionModule(feat_channels[1])
        self.QAM_enc3 = QueryAttentionModule(feat_channels[2])
        self.QAM_dec2 = QueryAttentionModule(feat_channels[1])
        self.QAM_dec1 = QueryAttentionModule(feat_channels[0])
        
        # Weight Fusion Module
        
        self.WFM_enc1 = WeightFusionModule(feat_channels[0])
        self.WFM_enc2 = WeightFusionModule(feat_channels[1])
        self.WFM_enc3 = WeightFusionModule(feat_channels[2])
        self.WFM_dec2 = WeightFusionModule(feat_channels[1])
        self.WFM_dec1 = WeightFusionModule(feat_channels[0])

    def forward(self, x, SuppAttention):
        
        # Encoder part
        
        x1 = self.conv_blk1(x)
        QA_enc1 = self.QAM_enc1(x1)
        FW_enc1 = self.WFM_enc1(SuppAttention[0],QA_enc1)
        x1w = x1*FW_enc1
        
        x_low1 = self.pool1(x1w)
        x2 = self.conv_blk2(x_low1)
        QA_enc2 = self.QAM_enc2(x2)
        FW_enc2 = self.WFM_enc2(SuppAttention[1],QA_enc2)
        x2w = x2*FW_enc2
        
        x_low2 = self.pool2(x2w)
        x3 = self.conv_blk3(x_low2)
        QA_enc3 = self.QAM_enc3(x3)
        FW_enc3 = self.WFM_enc3(SuppAttention[2],QA_enc3)
        x3w = x3*FW_enc3

        # Decoder part
        
        d2 = torch.cat([self.deconv_blk2(x3w), x2w], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        QA_dec2 = self.QAM_dec2(d_high2)
        FW_dec2 = self.WFM_dec2(SuppAttention[3],QA_dec2)
        d_high2w = d_high2*FW_dec2

        d1 = torch.cat([self.deconv_blk1(d_high2w), x1w], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        QA_dec1 = self.QAM_dec1(d_high1)
        FW_dec1 = self.WFM_dec1(SuppAttention[4],QA_dec1)
        d_high1w = d_high1*FW_dec1
        
        seg = self.sigmoid(self.one_conv(d_high1w))

        return seg

class Conv3D_Block(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())
        
        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv3D_Block(Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel,kernel,kernel), 
                                    stride=(stride,stride,stride), padding=(padding, padding, padding), output_padding=0, bias=True),
                        ReLU())
    
    def forward(self, x):
        
        return self.deconv(x)



class QueryAttentionModule(Module):
    
    def __init__(self,num_channel):
        super(QueryAttentionModule, self).__init__()
        
        self.conv3D = Conv3d(num_channel, num_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = Sigmoid()
        
    def forward(self,QueryFeat_input):
        
        QAtten = self.sigmoid(self.conv3D(QueryFeat_input))
        return QAtten 
    
class WeightFusionModule(Module):
    
    # def __init__(self,SuppFeat,QueryFeat):
    def __init__(self,num_channel):
        
        super(WeightFusionModule, self).__init__()
        self.num_channel = num_channel

        self.SuppFeatWeight = torch.sigmoid(Parameter(torch.zeros(self.num_channel,1,1,1,1)).cuda())         
        self.QueryFeatWeight = (1 - self.SuppFeatWeight).cuda()
        
    def forward(self,SuppAttention,QueryAttention):
        SuppFeat_weighted = F.conv3d(SuppAttention, self.SuppFeatWeight, bias=None, stride=1, padding=0, dilation=1, groups=self.num_channel)
        QueryFeat_weighted = F.conv3d(QueryAttention, self.QueryFeatWeight, bias=None, stride=1, padding=0, dilation=1, groups=self.num_channel)
        FusedWeight = SuppFeat_weighted + QueryFeat_weighted
        return FusedWeight



















