# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:00:53 2020

@author: Sunly
"""

from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
import torch
import torch.nn.functional as F

class UNet3DS(Module):
    # __                          __
    #  1|__   ______________   __|1
    #     2|__  __________  __|2
    #        3|__  ____  __|3
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity 

    def __init__(self, num_channels=1, feat_channels=[16, 32, 64], residual=None):
        
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3DS, self).__init__()
        
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

    def forward(self, x, support_label):
        
        # Encoder part
        
        x1 = self.conv_blk1(x)
        
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        # Decoder part
        
        d2 = torch.cat([self.deconv_blk2(x3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)
        
        seg = self.sigmoid(self.one_conv(d_high1))
        
        label_enc1 = support_label
        label_enc2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        label_enc3 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        label_dec2 = label_enc2
        label_dec1 = label_enc1
        
        SA_enc1 = self.sigmoid(x1*label_enc1)
        SA_enc2 = self.sigmoid(x2*label_enc2)
        SA_enc3 = self.sigmoid(x3*label_enc3)
        SA_dec2 = self.sigmoid(d_high2*label_dec2)
        SA_dec1 = self.sigmoid(d_high1*label_dec1)

        # return seg,x1,x2,x3,d_high2,d_high1
    
        return seg,SA_enc1,SA_enc2,SA_enc3,SA_dec2,SA_dec1

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

# class ChannelPool3d(AvgPool1d):
    
#     def __init__(self, kernel_size, stride, padding):
        
#         super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
#         self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

#     def forward(self, inp):
#         n, c, d, w, h = inp.size()
#         inp = inp.view(n,c,d*w*h).permute(0,2,1)
#         pooled = self.pool_1d(inp)
#         c = int(c/self.kernel_size[0])
#         return inp.view(n,c,d,w,h)
        
    
    
# inp = torch.randn(8,1,64,64,64).cuda()
# label = torch.ones(8,1,64,64,64).cuda()
# unet3ds = UNet3DS().cuda()

# _,SA_enc1,SA_enc2,SA_enc3,SA_dec2,SA_dec1 = unet3ds(inp,label)  
    
    
    
    
    
    
    
    

