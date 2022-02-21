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
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, MaxPool2d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

#from nn_common_modules import modules as sm
#from squeeze_and_excitation import squeeze_and_excitation as se

class UNet(Module):
    # __                          __
    #  1|__   ______________   __|1
    #     2|__  __________  __|2
    #        3|__  ____  __|3
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity 

    def __init__(self, num_channels=1, feat_channels=[16, 32, 64], residual=None):
        
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()
        
        # Encoder downsamplers
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)
        
        # Encoder convolutions
        self.conv_blk1 = Conv2D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv2D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv2D_Block(feat_channels[1], feat_channels[2], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk2 = Conv2D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv2D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers

        self.deconv_blk2 = Deconv2D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv2D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv2d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Activation function
        self.sigmoid = Sigmoid()
        

    def forward(self, x):
        
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

        return seg


class FewShotSegmentorDoubleSDnet(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner(params)
        self.segmentor = SDnetSegmentor(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class SDnetConditioner(nn.Module):
    """
    A conditional branch of few shot learning regressing the parameters for the segmentor
    """

    def __init__(self, params={}):
        super(SDnetConditioner, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 2
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        e1, _, ind1 = self.encode1(input)
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentor(nn.Module):
    """
    Segmentor Code

    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':1
        'se_block': True,
        'drop_out':0
    }

    """

    def __init__(self, params={}):
        super(SDnetSegmentor, self).__init__()
        params ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_class':1,
        'se_block': 'None',
        'drop_out':0
        }

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        if cls_w is not None:
            logit = torch.mul(logit, cls_w)
        
        #logit = F.softmax(logit)
        logit = torch.sigmoid(logit)

        return logit

# ffs_ms
class FewShotSegmentorDoubleSDnetms(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnetms, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditionerms(params)
        self.segmentor = SDnetSegmentorms(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment,seg_fea = self.segmentor(input2, weights)
        return segment,seg_fea

class SDnetConditionerms(nn.Module):
    """
    A conditional branch of few shot learning regressing the parameters for the segmentor
    """

    def __init__(self, params={}):
        super(SDnetConditionerms, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 2
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        e1, _, ind1 = self.encode1(input)
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentorms(nn.Module):
    """
    Segmentor Code

    param ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':1
        'se_block': True,
        'drop_out':0
    }

    """

    def __init__(self, params={}):
        super(SDnetSegmentorms, self).__init__()
        params ={
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_class':1,
        'se_block': 'None',
        'drop_out':0
        }

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        if cls_w is not None:
            logit = torch.mul(logit, cls_w)
        #logit = self.soft_max(logit)
        logit = torch.sigmoid(logit)

        return logit,bn

#### 将孙博代码改为2D版本，网络结构同源码 ####
### ours1: conditioner用Support Label对特征图mask,而不是并行输入### 
class FewShotSegmentorDoubleSDnet_ours1(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet_ours1, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner_ours1(params)
        self.segmentor = SDnetSegmentor_ours1(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class SDnetConditioner_ours1(nn.Module):

    def __init__(self, params={}):
        super(SDnetConditioner_ours1, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 1
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        support_image = input[:,0:1]
        support_label = input[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        e1, _, ind1 = self.encode1(support_image)
        e1 = e1*mask_down2 
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e2 = e2*mask_down4
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e3 = e3*mask_down8
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)      
        e4 = e4*mask_down16
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn = bn*mask_down16
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d4 = d4*mask_down8
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d3 = d3*mask_down4
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d2 = d2*mask_down2
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d1 = d1*mask
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

     

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentor_ours1(nn.Module):

    def __init__(self, params={}):
        super(SDnetSegmentor_ours1, self).__init__()

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        if cls_w is not None:
            logit = torch.mul(logit, cls_w)
        #logit = self.soft_max(logit)
        logit = torch.sigmoid(logit)

        return logit

### ours1: conditioner用Support Label对特征图mask,仍并行输入### 
class FewShotSegmentorDoubleSDnet_ours1a(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet_ours1a, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner_ours1a(params)
        self.segmentor = SDnetSegmentor_ours1(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class SDnetConditioner_ours1a(nn.Module):

    def __init__(self, params={}):
        super(SDnetConditioner_ours1a, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 2
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        support_image = input[:,0:1]
        support_label = input[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        e1, _, ind1 = self.encode1(input)
        e1 = e1*mask_down2 
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e2 = e2*mask_down4
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e3 = e3*mask_down8
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)      
        e4 = e4*mask_down16
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn = bn*mask_down16
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d4 = d4*mask_down8
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d3 = d3*mask_down4
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d2 = d2*mask_down2
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d1 = d1*mask
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

     

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights


### ours2: 加上自注意力机制### 
class FewShotSegmentorDoubleSDnet_ours2(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet_ours2, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner_ours2(params)
        self.segmentor = SDnetSegmentor_ours2(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class SDnetConditioner_ours2(nn.Module):

    def __init__(self, params={}):
        super(SDnetConditioner_ours2, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 2
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        e1, _, ind1 = self.encode1(input)
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentor_ours2(nn.Module):

    def __init__(self, params={}):
        super(SDnetSegmentor_ours2, self).__init__()

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

        # Query Attention Module
        self.QAM_e1 = QueryAttentionModule(params['num_channels'])
        self.QAM_e2 = QueryAttentionModule(params['num_channels'])
        self.QAM_e3 = QueryAttentionModule(params['num_channels'])
        self.QAM_e4 = QueryAttentionModule(params['num_channels'])
        self.QAM_bn = QueryAttentionModule(params['num_channels'])
        self.QAM_d1 = QueryAttentionModule(params['num_channels'])
        self.QAM_d2 = QueryAttentionModule(params['num_channels'])
        self.QAM_d3 = QueryAttentionModule(params['num_channels'])
        self.QAM_d4 = QueryAttentionModule(params['num_channels'])
        
        # Weight Fusion Module
        self.WFM_e1 = WeightFusionModule()
        self.WFM_e2 = WeightFusionModule()
        self.WFM_e3 = WeightFusionModule()
        self.WFM_e4 = WeightFusionModule()
        self.WFM_bn = WeightFusionModule()
        self.WFM_d1 = WeightFusionModule()
        self.WFM_d2 = WeightFusionModule()
        self.WFM_d3 = WeightFusionModule()
        self.WFM_d4 = WeightFusionModule()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e_qw1 = self.QAM_e1(e1)
            e_w1 = self.WFM_e1(e_w1,e_qw1)
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e_qw2 = self.QAM_e2(e2)
            e_w2 = self.WFM_e1(e_w2,e_qw2)
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e_qw3 = self.QAM_e3(e3)
            e_w3 = self.WFM_e3(e_w3,e_qw3)
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e_qw4 = self.QAM_e4(e4)
            e_w4 = self.WFM_e4(e_w4,e_qw4)
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn_qw = self.QAM_bn(bn)
            bn_w = self.WFM_e1(bn_w,bn_qw)
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d_qw4 = self.QAM_d4(d4)
            d_w4 = self.WFM_d4(d_w4,d_qw4)
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d_qw3 = self.QAM_d3(d3)
            d_w3 = self.WFM_d3(d_w3,d_qw3)
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d_qw2 = self.QAM_d2(d2)
            d_w2 = self.WFM_d2(d_w2,d_qw2)
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d_qw1 = self.QAM_d1(d1)
            d_w1 = self.WFM_d1(d_w1,d_qw1)
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        # if cls_w is not None:
        #     d_qw4 = self.QAM_d4(d4)
        #     d_w4 = self.WFM_d4(d_w4,d_qw4)
        #     logit = torch.mul(logit, cls_w)
        #logit = self.soft_max(logit)
        logit = torch.sigmoid(logit)

        return logit

### ours3: ours_1+ours_2### 
class FewShotSegmentorDoubleSDnet_ours3(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet_ours3, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner_ours3(params)
        self.segmentor = SDnetSegmentor_ours3(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment
    
class SDnetConditioner_ours3(nn.Module):

    def __init__(self, params={}):
        super(SDnetConditioner_ours3, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 1
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        support_image = input[:,0:1]
        support_label = input[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        e1, _, ind1 = self.encode1(support_image)
        e1 = e1*mask_down2 
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e2 = e2*mask_down4
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e3 = e3*mask_down8
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)      
        e4 = e4*mask_down16
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn = bn*mask_down16
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d4 = d4*mask_down8
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d3 = d3*mask_down4
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d2 = d2*mask_down2
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d1 = d1*mask
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentor_ours3(nn.Module):
 
    def __init__(self, params={}):
        super(SDnetSegmentor_ours3, self).__init__()

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

        # Query Attention Module
        self.QAM_e1 = QueryAttentionModule(params['num_channels'])
        self.QAM_e2 = QueryAttentionModule(params['num_channels'])
        self.QAM_e3 = QueryAttentionModule(params['num_channels'])
        self.QAM_e4 = QueryAttentionModule(params['num_channels'])
        self.QAM_bn = QueryAttentionModule(params['num_channels'])
        self.QAM_d1 = QueryAttentionModule(params['num_channels'])
        self.QAM_d2 = QueryAttentionModule(params['num_channels'])
        self.QAM_d3 = QueryAttentionModule(params['num_channels'])
        self.QAM_d4 = QueryAttentionModule(params['num_channels'])
        
        # Weight Fusion Module
        self.WFM_e1 = WeightFusionModule()
        self.WFM_e2 = WeightFusionModule()
        self.WFM_e3 = WeightFusionModule()
        self.WFM_e4 = WeightFusionModule()
        self.WFM_bn = WeightFusionModule()
        self.WFM_d1 = WeightFusionModule()
        self.WFM_d2 = WeightFusionModule()
        self.WFM_d3 = WeightFusionModule()
        self.WFM_d4 = WeightFusionModule()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e_qw1 = self.QAM_e1(e1)
            e_w1 = self.WFM_e1(e_w1,e_qw1)
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e_qw2 = self.QAM_e2(e2)
            e_w2 = self.WFM_e1(e_w2,e_qw2)
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e_qw3 = self.QAM_e3(e3)
            e_w3 = self.WFM_e3(e_w3,e_qw3)
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e_qw4 = self.QAM_e4(e4)
            e_w4 = self.WFM_e4(e_w4,e_qw4)
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn_qw = self.QAM_bn(bn)
            bn_w = self.WFM_e1(bn_w,bn_qw)
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d_qw4 = self.QAM_d4(d4)
            d_w4 = self.WFM_d4(d_w4,d_qw4)
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d_qw3 = self.QAM_d3(d3)
            d_w3 = self.WFM_d3(d_w3,d_qw3)
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d_qw2 = self.QAM_d2(d2)
            d_w2 = self.WFM_d2(d_w2,d_qw2)
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d_qw1 = self.QAM_d1(d1)
            d_w1 = self.WFM_d1(d_w1,d_qw1)
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        # if cls_w is not None:
        #     d_qw4 = self.QAM_d4(d4)
        #     d_w4 = self.WFM_d4(d_w4,d_qw4)
        #     logit = torch.mul(logit, cls_w)
        #logit = self.soft_max(logit)
        logit = torch.sigmoid(logit)

        return logit

### ours3ms: ours_1+ours_2+ms### 
class FewShotSegmentorDoubleSDnet_ours3ms(nn.Module):

    def __init__(self, params={}):
        
        super(FewShotSegmentorDoubleSDnet_ours3ms, self).__init__()
        params ={
            'num_channels':1,
            'num_filters':64,
            'kernel_h':5,
            'kernel_w':5,
            'kernel_c':1,
            'stride_conv':1,
            'pool':2,
            'stride_pool':2,
            'num_class':1,
            'se_block': 'None',
            'drop_out':0
            }
        self.conditioner = SDnetConditioner_ours3ms(params)
        self.segmentor = SDnetSegmentor_ours3ms(params)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment,seg_fea = self.segmentor(input2, weights)
        return segment,seg_fea
    
class SDnetConditioner_ours3ms(nn.Module):

    def __init__(self, params={}):
        super(SDnetConditioner_ours3ms, self).__init__()
        se_block_type = se.SELayer.SSE
        params['num_channels'] = 1
        params['num_filters'] = 16
        self.encode1 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.squeeze_conv_e4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.bottleneck = sm.GenericBlock(params)
        self.squeeze_conv_bn = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.decode1 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d1 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d2 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode3 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d3 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        self.decode4 = sm.SDnetDecoderBlock(params)
        self.squeeze_conv_d4 = nn.Conv2d(in_channels=params['num_filters'], out_channels=1,
                                         kernel_size=(1, 1),
                                         padding=(0, 0),
                                         stride=1)
        params['num_channels'] = 16
        self.classifier = sm.ClassifierBlock(params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        support_image = input[:,0:1]
        support_label = input[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        e1, _, ind1 = self.encode1(support_image)
        e1 = e1*mask_down2 
        e_w1 = self.sigmoid(self.squeeze_conv_e1(e1))
        e2, out2, ind2 = self.encode2(e1)
        e2 = e2*mask_down4
        e_w2 = self.sigmoid(self.squeeze_conv_e2(e2))
        e3, _, ind3 = self.encode3(e2)
        e3 = e3*mask_down8
        e_w3 = self.sigmoid(self.squeeze_conv_e3(e3))
        e4, _, ind4 = self.encode3(e3)      
        e4 = e4*mask_down16
        e_w4 = self.sigmoid(self.squeeze_conv_e4(e4))

        bn = self.bottleneck(e4)
        bn = bn*mask_down16
        bn_w4 = self.sigmoid(self.squeeze_conv_bn(bn))
        d4 = self.decode4(bn, None, ind4)
        d4 = d4*mask_down8
        d_w4 = self.sigmoid(self.squeeze_conv_d4(d4))
        d3 = self.decode3(d4, None, ind3)
        d3 = d3*mask_down4
        d_w3 = self.sigmoid(self.squeeze_conv_d3(d3))
        d2 = self.decode2(d3, None, ind2)
        d2 = d2*mask_down2
        d_w2 = self.sigmoid(self.squeeze_conv_d2(d2))
        d1 = self.decode1(d2, None, ind1)
        d1 = d1*mask
        d_w1 = self.sigmoid(self.squeeze_conv_d1(d1))

        space_weights = (e_w1, e_w2, e_w3, e_w4, bn_w4,
                         d_w4, d_w3, d_w2, d_w1, None)
        channel_weights = (None, None, None, None)

        return space_weights, channel_weights

class SDnetSegmentor_ours3ms(nn.Module):
 
    def __init__(self, params={}):
        super(SDnetSegmentor_ours3ms, self).__init__()

        params['num_channels'] = 1
        params['num_filters'] = 64
        self.encode1 = sm.SDnetEncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.SDnetEncoderBlock(params)
        self.encode3 = sm.SDnetEncoderBlock(params)
        self.encode4 = sm.SDnetEncoderBlock(params)
        self.bottleneck = sm.GenericBlock(params)

        self.decode1 = sm.SDnetDecoderBlock(params)
        self.decode2 = sm.SDnetDecoderBlock(params)
        self.decode3 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.decode4 = sm.SDnetDecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        #self.soft_max = nn.Softmax2d()

        # Query Attention Module
        self.QAM_e1 = QueryAttentionModule(params['num_channels'])
        self.QAM_e2 = QueryAttentionModule(params['num_channels'])
        self.QAM_e3 = QueryAttentionModule(params['num_channels'])
        self.QAM_e4 = QueryAttentionModule(params['num_channels'])
        self.QAM_bn = QueryAttentionModule(params['num_channels'])
        self.QAM_d1 = QueryAttentionModule(params['num_channels'])
        self.QAM_d2 = QueryAttentionModule(params['num_channels'])
        self.QAM_d3 = QueryAttentionModule(params['num_channels'])
        self.QAM_d4 = QueryAttentionModule(params['num_channels'])
        
        # Weight Fusion Module
        self.WFM_e1 = WeightFusionModule()
        self.WFM_e2 = WeightFusionModule()
        self.WFM_e3 = WeightFusionModule()
        self.WFM_e4 = WeightFusionModule()
        self.WFM_bn = WeightFusionModule()
        self.WFM_d1 = WeightFusionModule()
        self.WFM_d2 = WeightFusionModule()
        self.WFM_d3 = WeightFusionModule()
        self.WFM_d4 = WeightFusionModule()

    def forward(self, inpt, weights=None):
        if weights != None:
            space_weights, channel_weights = weights
        else:
            space_weights = None
        e_w1, e_w2, e_w3, e_w4, bn_w, d_w4, d_w3, d_w2, d_w1, cls_w = space_weights if space_weights is not None else (
                None, None, None, None, None, None, None, None, None, None)

        e1, _, ind1 = self.encode1(inpt)
        if e_w1 is not None:
            e_qw1 = self.QAM_e1(e1)
            e_w1 = self.WFM_e1(e_w1,e_qw1)
            e1 = torch.mul(e1, e_w1)

        e2, _, ind2 = self.encode2(e1)
        if e_w2 is not None:
            e_qw2 = self.QAM_e2(e2)
            e_w2 = self.WFM_e1(e_w2,e_qw2)
            e2 = torch.mul(e2, e_w2)

        e3, _, ind3 = self.encode3(e2)
        if e_w3 is not None:
            e_qw3 = self.QAM_e3(e3)
            e_w3 = self.WFM_e3(e_w3,e_qw3)
            e3 = torch.mul(e3, e_w3)

        e4, out4, ind4 = self.encode4(e3)
        if e_w4 is not None:
            e_qw4 = self.QAM_e4(e4)
            e_w4 = self.WFM_e4(e_w4,e_qw4)
            e4 = torch.mul(e4, e_w4)

        bn = self.bottleneck(e4)
        if bn_w is not None:
            bn_qw = self.QAM_bn(bn)
            bn_w = self.WFM_e1(bn_w,bn_qw)
            bn = torch.mul(bn, bn_w)

        d4 = self.decode4(bn, None, ind4)
        if d_w4 is not None:
            d_qw4 = self.QAM_d4(d4)
            d_w4 = self.WFM_d4(d_w4,d_qw4)
            d4 = torch.mul(d4, d_w4)

        d3 = self.decode3(d4, None, ind3)
        if d_w3 is not None:
            d_qw3 = self.QAM_d3(d3)
            d_w3 = self.WFM_d3(d_w3,d_qw3)
            d3 = torch.mul(d3, d_w3)

        d2 = self.decode2(d3, None, ind2)
        if d_w2 is not None:
            d_qw2 = self.QAM_d2(d2)
            d_w2 = self.WFM_d2(d_w2,d_qw2)
            d2 = torch.mul(d2, d_w2)

        d1 = self.decode1(d2, None, ind1)
        if d_w1 is not None:
            d_qw1 = self.QAM_d1(d1)
            d_w1 = self.WFM_d1(d_w1,d_qw1)
            d1 = torch.mul(d1, d_w1)

        logit = self.classifier.forward(d1)
        # if cls_w is not None:
        #     d_qw4 = self.QAM_d4(d4)
        #     d_w4 = self.WFM_d4(d_w4,d_qw4)
        #     logit = torch.mul(logit, cls_w)
        #logit = self.soft_max(logit)
        logit = torch.sigmoid(logit)

        return logit,bn


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

class QueryAttentionModule(Module):
    
    def __init__(self,num_channel):
        super(QueryAttentionModule, self).__init__()
        
        self.conv3D = nn.Conv2d(num_channel, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self,QueryFeat_input):
        
        QAtten = torch.sigmoid(self.conv3D(QueryFeat_input))
        return QAtten 
    
class WeightFusionModule(Module):

    def __init__(self):
        
        super(WeightFusionModule, self).__init__()
        #self.num_channel = num_channel

        self.SuppFeatWeight = torch.sigmoid(nn.Parameter(torch.zeros(1,1,1,1)).cuda())         
        self.QueryFeatWeight = (1 - self.SuppFeatWeight).cuda()
        
    def forward(self,SuppAttention,QueryAttention):
        FusedWeight = SuppAttention*self.SuppFeatWeight + QueryAttention*self.QueryFeatWeight
        return FusedWeight


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


class Conv2D_Block(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
        super(Conv2D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv2d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm2d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv2d(out_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm2d(out_feat),
                        ReLU())
        
        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv2d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv2D_Block(Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv2D_Block, self).__init__()
        
        self.deconv = Sequential(
                        ConvTranspose2d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, output_padding=0, bias=True),
                        ReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

import resnet
### 预训练res unet wenao###
### only 分割 ###
class baseline(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.out = output_block(64,1)

    def forward(self, x):
        x = self.ec_block(x)       
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out(x_out)
        x_out = torch.sigmoid(x_out)

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 

class autoencoder(nn.Module):

    def __init__(self,eval=False):
        super().__init__()
        self.eval = eval
        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn_noskip(256, 128)
        self.up3 = UnetBlock_tran10_bn_noskip(128, 64)
        self.up4 = UnetBlock_tran10_bn_noskip(64, 64)
    
        self.out = output_block(64,1)

    def forward(self, x):
        x = torch.cat((x,x,x),1)  
        x = self.rn(x)
        if self.eval:
            return x
        
        x = F.relu(x)

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out(x_out)
        x_out = torch.sigmoid(x_out)

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 


### 预训练res unet ==> my_fss ### 
### error: conditioner 的加权没加sigmoid,后已改正
class my_fss(nn.Module):
    def __init__(self,pretrain=True):
        super(my_fss, self).__init__()

        self.conditioner = my_conditioner(pretrain)
        self.segmentor = my_segmentor(pretrain)

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class my_conditioner(nn.Module):

    def __init__(self,pretrain):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(2)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):

        x = self.ec_block(x)
        w1 = torch.sigmoid(self.se1(x))       
        x = F.relu(self.rn(x))
        w5 = torch.sigmoid(self.se5(x))

        x = self.up2(x, self.sfs[2].features)
        w6 = torch.sigmoid(self.se6(x))
        x = self.up3(x, self.sfs[1].features)
        w7 = torch.sigmoid(self.se7(x))
        x = self.up4(x, self.sfs[0].features)
        w8 = torch.sigmoid(self.se8(x))

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        w9 = torch.sigmoid(self.se9(x_out))

        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))
 

        return (w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class my_segmentor(nn.Module):

    def __init__(self,pretrain):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  


    def forward(self, x, weight):
        x = self.ec_block(x)  
        x = x*weight[0]     
        x = F.relu(self.rn(x))
        x = x*weight[4]

        x = self.up2(x, self.sfs[2].features)
        x = x*weight[5]
        x = self.up3(x, self.sfs[1].features)
        x = x*weight[6]
        x = self.up4(x, self.sfs[0].features)
        x = x*weight[7]

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        x = x*weight[8]

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 


### 预训练res unet ==> my_fss 网络结构调参###
### error: conditioner 的加权没加sigmoid
class my_fss_ex(nn.Module):
    def __init__(self):
        super(my_fss_ex, self).__init__()

        self.conditioner = my_conditioner_ex()
        self.segmentor = my_segmentor_ex()

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class my_conditioner_ex(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        ratio = 1

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))
        
        # self.up2 = UnetBlock_tran10_bn(256, 128)
        # self.up3 = UnetBlock_tran10_bn(128, 64)
        # self.up4 = UnetBlock_tran10_bn(64, 64)

        # self.up2 = UnetBlock_tran10_bn_noskip(256, 128)
        # self.up3 = UnetBlock_tran10_bn_noskip(128, 64)
        # self.up4 = UnetBlock_tran10_bn_noskip(64, 64)

        self.up2 = UnetBlock_tran10_bn(256//ratio, 128//ratio)
        self.up3 = UnetBlock_tran10_bn(128//ratio, 64//ratio)
        self.up4 = UnetBlock_tran10_bn(64//ratio, 64//ratio)

        self.ec_block = expand_compress_block3(2)

        # self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(64)
        self.conv = nn.Conv2d(64//ratio, 64//ratio, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64//ratio)

        #weight
        # self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        # self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        # self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64//ratio,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64//ratio,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128//ratio,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256//ratio,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128//ratio,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64//ratio,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64//ratio,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64//ratio,1,kernel_size=1, padding=0)


    def forward(self, x):

        x = self.ec_block(x)
        w1 = self.se1(x)       
        x = F.relu(self.rn(x))
        w5 = self.se5(x)

        x = self.up2(x, self.sfs[2].features)
        w6 = self.se6(x)
        x = self.up3(x, self.sfs[1].features)
        w7 = self.se7(x)
        x = self.up4(x, self.sfs[0].features)
        w8 = self.se8(x)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        w9 = self.se9(x_out)

        w2 = self.se2(self.sfs[0].features)
        w3 = self.se3(self.sfs[1].features)
        w4 = self.se4(self.sfs[2].features)
 

        return (w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 

class my_segmentor_ex(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        # self.up2 = UnetBlock_tran10_bn_noskip(256, 128)
        # self.up3 = UnetBlock_tran10_bn_noskip(128, 64)
        # self.up4 = UnetBlock_tran10_bn_noskip(64, 64)
        
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out1 = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self._bn1 = nn.BatchNorm2d(64)
        self.bn_out1 = nn.BatchNorm2d(1)  

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self._bn2 = nn.BatchNorm2d(64)
        self.bn_out2 = nn.BatchNorm2d(1)  

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out3 = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self._bn3 = nn.BatchNorm2d(64)
        self.bn_out3 = nn.BatchNorm2d(1)  


    
        # self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(1)      
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(32)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.bn6 = nn.BatchNorm2d(32)
        # self.bn7 = nn.BatchNorm2d(32)


    def forward(self, x, weight):
        x = self.ec_block(x)  
        x = x*weight[0]     
        x = F.relu(self.rn(x))
        x = x*weight[4]

        x = self.up2(x, self.sfs[2].features)
        x = x*weight[5]
        x = self.up3(x, self.sfs[1].features)
        x = x*weight[6]
        x = self.up4(x, self.sfs[0].features)
        x = x*weight[7]

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        x = x*weight[8]

        x_out = self.bn_out(self.conv_out(x))


        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]

        # ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        # ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        # ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        # ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        # ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        # ds1 = F.relu(self._bn1(self.conv1(ds1)))
        # ds1 = self.bn_out1(self.conv_out1(ds1))

        # ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        # ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        # ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        # ds2 = F.relu(self._bn2(self.conv2(ds2)))
        # ds2 = self.bn_out2(self.conv_out2(ds2))
        
        # ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        # ds3 = F.relu(self._bn3(self.conv3(ds3)))
        # ds3 = self.bn_out3(self.conv_out3(ds3))

        return torch.sigmoid(x_out)#,torch.sigmoid(ds1),torch.sigmoid(ds2),torch.sigmoid(ds3)

    def close(self):
        for sf in self.sfs: sf.remove() 


class fss_fusion(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = fss_fusion_conditioner(pretrain)
        self.segmentor = fss_fusion_segmentor(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class fss_fusion_conditioner(nn.Module):

    def __init__(self,pretrain):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(2)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        fea = []
        x = self.ec_block(x)
        fea.append(x)     
        x = F.relu(self.rn(x))
        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        fea.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        fea.append(x_out)

        return fea

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class fss_fusion_segmentor(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            self.weight.append(fusion_module(in_channels[i]).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


class fss_vis(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = my_conditioner_fea_nlc(pretrain)
        self.segmentor = segmentor_vis(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,attention,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,attention,q_feature,qw_feature

class segmentor_vis(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC_vis(in_channels[i]).cuda())
            else:
                self.weight.append(NONLocalBlock2D(in_channels[i],sub_sample=sub_sample,
                    bn_layer=bn_layer,shortcut=shortcut,cos_dis=cos_dis).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        attention = []
        x = self.ec_block(x)
        fea.append(x)
        x,att = self.weight[0](s_fea[0],x)
        fea_w.append(x)
        attention.append(att)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features,att = self.weight[1](s_fea[1],self.sfs[0].features)
        attention.append(att)
        self.sfs[1].features,att = self.weight[2](s_fea[2],self.sfs[1].features)
        attention.append(att)
        self.sfs[2].features,att = self.weight[3](s_fea[3],self.sfs[2].features)
        attention.append(att)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x,att = self.weight[4](s_fea[4],x)
        attention.append(att)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x,att = self.weight[5](s_fea[5],x)
        attention.append(att)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x,att = self.weight[6](s_fea[6],x)
        attention.append(att)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x,att = self.weight[7](s_fea[7],x)
        attention.append(att)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x,att = self.weight[8](s_fea[8],x)
        attention.append(att)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,attention,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


### 预训练res unet ==> my_fss 中间层输出 ###
class my_fss_fea(nn.Module):
    def __init__(self):
        super(my_fss_fea, self).__init__()

        self.conditioner = my_conditioner_fea()
        self.segmentor = my_segmentor_fea()

    def forward(self, input1, input2):
        s_feature,weights = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, weights)
        return segment,s_feature,q_feature,qw_feature

class my_conditioner_fea(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(2)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        # self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        # self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        # self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        k = 3
        pad = 1
        self.se1 = nn.Conv2d(3,1,kernel_size=k, padding=pad)
        self.se2 = nn.Conv2d(64,1,kernel_size=k, padding=pad)
        self.se3 = nn.Conv2d(64,1,kernel_size=k, padding=pad)
        self.se4 = nn.Conv2d(128,1,kernel_size=k, padding=pad)
        self.se5 = nn.Conv2d(256,1,kernel_size=k, padding=pad)
        self.se6 = nn.Conv2d(128,1,kernel_size=k, padding=pad)
        self.se7 = nn.Conv2d(64,1,kernel_size=k, padding=pad)
        self.se8 = nn.Conv2d(64,1,kernel_size=k, padding=pad)
        self.se9 = nn.Conv2d(64,1,kernel_size=k, padding=pad)

    def forward(self, x):
        fea = []
        x = self.ec_block(x)
        fea.append(x)
        w1 = torch.sigmoid(self.se1(x))       
        x = F.relu(self.rn(x))
        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))
        fea.append(x)
        w5 = torch.sigmoid(self.se5(x))

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        w6 = torch.sigmoid(self.se6(x))
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        w7 = torch.sigmoid(self.se7(x))
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        w8 = torch.sigmoid(self.se8(x))

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        fea.append(x_out)
        w9 = torch.sigmoid(self.se9(x_out))

        return fea,(w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class my_segmentor_fea(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  


    def forward(self, x, weight):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = x*weight[0]
        fea_w.append(x)

        x = F.relu(self.rn(x))
        

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = x*weight[4]
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = x*weight[5]
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = x*weight[6]
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = x*weight[7]
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = x*weight[8]
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 

### 预训练res unet ==> my_fss 中间层输出 ###
### S和Q concat起来SE
class fss_fea2(nn.Module):
    def __init__(self,pretrain=True):
        super().__init__()

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = segmentor_fea2(pretrain)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea2(nn.Module):

    def __init__(self,pretrain):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):           
            self.weight.append(SE_AC2(in_channels[i]).cuda())


    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 

### SE 和 PANet结合
class fss_fea3(nn.Module):
    def __init__(self,pretrain=True,fuse_type=None):
        super().__init__()

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = segmentor_fea3(pretrain,fuse_type)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        support_mask = input1[:,1:2]
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature,support_mask)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea3(nn.Module):

    def __init__(self,pretrain,fuse_type):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):           
            self.weight.append(SE_AC3(fuse_type,in_channels[i]).cuda())


    def forward(self, x, s_fea,support_mask):
        fea = []
        fea_w = []
        mask = support_mask
        mask_down2 = F.interpolate(support_mask,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_mask,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_mask,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_mask,scale_factor=0.125*0.5,mode='nearest')

        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x,mask)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features,mask_down2)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features,mask_down4)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features,mask_down8)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x,mask_down16)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x,mask_down8)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x,mask_down4)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x,mask_down2)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x,mask)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


#### 预训练res unet ==> my_fss 中间层输出 + nonlocal注意力###
#### SE + non-local交互 输出Q的加权和+1*1卷积 add Q
class my_fss_fea_nlc(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super(my_fss_fea_nlc, self).__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class conditioner_fea(nn.Module):

    def __init__(self,pretrain):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(2)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        # self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        # self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        # self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        # self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        # self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):
        fea = []
        x = self.ec_block(x)
        fea.append(x)
        # w1 = torch.sigmoid(self.se1(x))       
        x = F.relu(self.rn(x))
        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        # w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        # w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        # w4 = torch.sigmoid(self.se4(self.sfs[2].features))
        fea.append(x)
        #w5 = torch.sigmoid(self.se5(x))

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        #w6 = torch.sigmoid(self.se6(x))
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        #w7 = torch.sigmoid(self.se7(x))
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        #w8 = torch.sigmoid(self.se8(x))

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        fea.append(x_out)
        #w9 = torch.sigmoid(self.se9(x_out))

        return fea#,(w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class my_segmentor_fea_nlc(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(NONLocalBlock2D(in_channels[i],sub_sample=sub_sample,
                    bn_layer=bn_layer,shortcut=shortcut,cos_dis=cos_dis).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


#### 预训练res unet ==> my_fss 中间层输出 + nonlocal注意力2###
#### SE + non-local交互 输出S的加权和+1*1卷积+sigmoid
class my_fss_fea_nlc2(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super(my_fss_fea_nlc2, self).__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc2(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc2(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(NLC_AC2(in_channels[i],sub_sample=sub_sample).cuda())
        # for i in range(9):
        #     if i+1 not in nlc_layer:
        #         self.weight.append(SE_AC(in_channels[i]).cuda())
        #     else:
        #         self.weight.append(SE_AC_nlc(in_channels[i],sub_sample=sub_sample).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


class my_fss_fea_nlc2_(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc2_(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc2_(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        if 1 not in nlc_layer:
            self.w1 = SE_AC(3)
        else:
            self.w1 = NLC_AC2(3)
        if 2 not in nlc_layer:
            self.w2 = SE_AC(64)
        else:
            self.w2 = NLC_AC2(64)
        if 3 not in nlc_layer:
            self.w3 = SE_AC(64)
        else:
            self.w3 = NLC_AC2(64)
        if 4 not in nlc_layer:
            self.w4 = SE_AC(128)
        else:
            self.w4 = NLC_AC2(128)
        if 5 not in nlc_layer:
            self.w5 = SE_AC(256)
        else:
            self.w5 = NLC_AC2(256)
        if 6 not in nlc_layer:
            self.w6 = SE_AC(128)
        else:
            self.w6 = NLC_AC2(128)
        if 7 not in nlc_layer:
            self.w7 = SE_AC(64)
        else:
            self.w7 = NLC_AC2(64)
        if 8 not in nlc_layer:
            self.w8 = SE_AC(64)
        else:
            self.w8 = NLC_AC2(64)
        if 9 not in nlc_layer:
            self.w9 = SE_AC(64)
        else:
            self.w9 = NLC_AC2(64)


        # self.weight = []
        # for i in range(9):
        #     if i+1 not in nlc_layer:
        #         if i==0:
        #         self.weight = [SE_AC(in_channels[i])]
        #     else:
        #         self.weight.append(nlc_module1(in_channels[i]))
        # for i in range(9):
        #     if i==0:
        #         self.weight = [SE_AC(in_channels[i]).cuda()]
        #     else:
        #         self.weight.append(SE_AC(in_channels[i]).cuda())


    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.w1(s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.w2(s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.w3(s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.w4(s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.w5(s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.w6(s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.w7(s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.w8(s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.w9(s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


### SE_AC_nlc
### SE + non-local-S 输出S的加权和+1*1卷积+sigmoid
class my_fss_fea_nlc21(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc21(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc21(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        # for i in range(9):
        #     if i+1 not in nlc_layer:
        #         self.weight.append(SE_AC(in_channels[i]).cuda())
        #     else:
        #         self.weight.append(NLC_AC2(in_channels[i],sub_sample=sub_sample).cuda())
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(SE_AC_nlc(in_channels[i],sub_sample=sub_sample).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 

### SE + non-local交互 输出S的加权和+1*1卷积 add S + se
class my_fss_fea_nlc22(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc22(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc22(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(NLC_AC22(in_channels[i],sub_sample=sub_sample).cuda())
        # for i in range(9):
        #     if i+1 not in nlc_layer:
        #         self.weight.append(SE_AC(in_channels[i]).cuda())
        #     else:
        #         self.weight.append(SE_AC_nlc(in_channels[i],sub_sample=sub_sample).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 

### SE + non-local-S 输出S的加权和+1*1卷积 add S + se
class my_fss_fea_nlc23(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super().__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc23(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc23(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        # for i in range(9):
        #     if i+1 not in nlc_layer:
        #         self.weight.append(SE_AC(in_channels[i]).cuda())
        #     else:
        #         self.weight.append(NLC_AC2(in_channels[i],sub_sample=sub_sample).cuda())
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(SE_AC_nlc2(in_channels[i],sub_sample=sub_sample).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 

#### 预训练res unet ==> my_fss 中间层输出 + nonlocal注意力3###
class my_fss_fea_nlc3(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super(my_fss_fea_nlc3, self).__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_nlc3(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_nlc3(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer
        #print(nlc_layer)

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 in nlc_layer:
                self.weight.append(NLC_AC3(in_channels[i],sub_sample=sub_sample).cuda())
            else:
                self.weight.append(None)

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        if 1 in self.nlc_layer:
            x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        if 2 in self.nlc_layer:
            self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        if 3 in self.nlc_layer:
            self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        if 4 in self.nlc_layer:
            self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        if 5 in self.nlc_layer:
            x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        if 6 in self.nlc_layer:
            x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        if 7 in self.nlc_layer:
            x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        if 8 in self.nlc_layer:
            x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        if 9 in self.nlc_layer:
            x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


#### 预训练res unet ==> my_fss 中间层输出 + 按类别nonlocal注意力###
class my_fss_fea_cnlc(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False):
        super(my_fss_fea_cnlc, self).__init__()

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = my_segmentor_fea_cnlc(pretrain,nlc_layer,sub_sample,bn_layer,shortcut,cos_dis)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class my_segmentor_fea_cnlc(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut,cos_dis):
        super().__init__()
        self.nlc_layer = nlc_layer

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=pretrain).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        #weight
        in_channels = [3,64,64,128,256,128,64,64,64]
        self.weight = []
        for i in range(9):
            if i+1 not in nlc_layer:
                self.weight.append(SE_AC(in_channels[i]).cuda())
            else:
                self.weight.append(CNONLocalBlock2D(in_channels[i],sub_sample=sub_sample,
                    bn_layer=bn_layer,shortcut=shortcut,cos_dis=cos_dis).cuda())

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = self.weight[0](s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.weight[1](s_fea[1],self.sfs[0].features)
        self.sfs[1].features = self.weight[2](s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.weight[3](s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = self.weight[4](s_fea[4],x)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = self.weight[5](s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = self.weight[6](s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = self.weight[7](s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = self.weight[8](s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 



### 预训练res unet ==> my_fss ours1a ### 
class my_fss_ours1a(nn.Module):
    def __init__(self):
        super().__init__()

        self.conditioner = my_conditioner_ours1a()
        self.segmentor = my_segmentor_ours1a()

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class my_conditioner_ours1a(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):
        support_image = x[:,0:1]
        support_label = x[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        x = self.ec_block(support_image) 
        x = x*mask
        w1 = torch.sigmoid(self.se1(x))
 
        x = F.relu(self.rn(x))

        self.sfs[0].features = self.sfs[0].features*mask_down2
        self.sfs[1].features = self.sfs[1].features*mask_down4
        self.sfs[2].features = self.sfs[2].features*mask_down8
        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))

        x = x*mask_down16
        w5 = torch.sigmoid(self.se5(x))

        x = self.up2(x, self.sfs[2].features)
        x = x*mask_down8
        w6 = torch.sigmoid(self.se6(x))

        x = self.up3(x, self.sfs[1].features)
        x = x*mask_down4
        w7 = torch.sigmoid(self.se7(x))

        x = self.up4(x, self.sfs[0].features)
        x = x*mask_down2
        w8 = torch.sigmoid(self.se8(x))

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        x = x*mask
        w9 = torch.sigmoid(self.se9(x))



        return (w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 

class my_segmentor_ours1a(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  


    def forward(self, x, weight):
        x = self.ec_block(x)  
        x = x*weight[0]     
        x = F.relu(self.rn(x))
        x = x*weight[4]

        x = self.up2(x, self.sfs[2].features)
        x = x*weight[5]
        x = self.up3(x, self.sfs[1].features)
        x = x*weight[6]
        x = self.up4(x, self.sfs[0].features)
        x = x*weight[7]

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        x = x*weight[8]

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 

### 预训练res unet ==> my_fss ours1b ### 
class my_fss_ours1b(nn.Module):
    def __init__(self):
        super().__init__()

        self.conditioner = my_conditioner_ours1b()
        self.segmentor = my_segmentor_ours1b()

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class my_conditioner_ours1b(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):
        support_image = x[:,0:1]
        support_label = x[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        x = self.ec_block(support_image)
        w1 = torch.sigmoid(self.se1(x))
        x = x*mask   
        
        x = F.relu(self.rn(x))
        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))
        self.sfs[0].features = self.sfs[0].features*mask_down2
        self.sfs[1].features = self.sfs[1].features*mask_down4
        self.sfs[2].features = self.sfs[2].features*mask_down8

        w5 = torch.sigmoid(self.se5(x))
        x = x*mask_down16

        x = self.up2(x, self.sfs[2].features)
        w6 = torch.sigmoid(self.se6(x))
        x = x*mask_down8

        x = self.up3(x, self.sfs[1].features)
        w7 = torch.sigmoid(self.se7(x))
        x = x*mask_down4

        x = self.up4(x, self.sfs[0].features)
        w8 = torch.sigmoid(self.se8(x))
        x = x*mask_down2

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        w9 = torch.sigmoid(self.se9(x))
        x = x*mask



        return (w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 

class my_segmentor_ours1b(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  


    def forward(self, x, weight):
        x = self.ec_block(x)  
        x = x*weight[0]     
        x = F.relu(self.rn(x))
        x = x*weight[4]

        x = self.up2(x, self.sfs[2].features)
        x = x*weight[5]
        x = self.up3(x, self.sfs[1].features)
        x = x*weight[6]
        x = self.up4(x, self.sfs[0].features)
        x = x*weight[7]

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        x = x*weight[8]

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 

### 预训练res unet ==> my_fss ours1b 中间层输出 ###
class my_fss_ours1b_fea(nn.Module):
    def __init__(self):
        super(my_fss_ours1b_fea, self).__init__()

        self.conditioner = my_conditioner_ours1b_fea()
        self.segmentor = my_segmentor_fea()

    def forward(self, input1, input2):
        s_feature,sm_feature,weights = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, weights)
        return segment,s_feature,sm_feature,q_feature,qw_feature

class my_conditioner_ours1b_fea(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):
        fea = []
        fea_m = []

        support_image = x[:,0:1]
        support_label = x[:,1:2]

        mask = support_label
        mask_down2 = F.interpolate(support_label,scale_factor=0.5,mode='nearest')
        mask_down4 = F.interpolate(support_label,scale_factor=0.25,mode='nearest')
        mask_down8 = F.interpolate(support_label,scale_factor=0.125,mode='nearest')
        mask_down16 = F.interpolate(support_label,scale_factor=0.125*0.5,mode='nearest')

        x = self.ec_block(support_image)
        fea.append(x)
        w1 = torch.sigmoid(self.se1(x))
        x = x*mask
        fea_m.append(x)       
        x = F.relu(self.rn(x))
        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))
        self.sfs[0].features = self.sfs[0].features*mask_down2
        self.sfs[1].features = self.sfs[1].features*mask_down4
        self.sfs[2].features = self.sfs[2].features*mask_down8
        fea_m.append(self.sfs[0].features)
        fea_m.append(self.sfs[1].features)
        fea_m.append(self.sfs[2].features)
        fea.append(x)
        w5 = torch.sigmoid(self.se5(x))
        x = x*mask_down16
        fea_m.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        w6 = torch.sigmoid(self.se6(x))
        x = x*mask_down8
        fea_m.append(x)
        
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        w7 = torch.sigmoid(self.se7(x))
        x = x*mask_down4
        fea_m.append(x)


        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        w8 = torch.sigmoid(self.se8(x))
        x = x*mask_down2
        fea_m.append(x)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        fea.append(x_out)
        w9 = torch.sigmoid(self.se9(x_out))
        x_out = x_out*mask
        fea_m.append(x)

        return fea,fea_m,(w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class my_segmentor_ours1b_fea(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)  


    def forward(self, x, weight):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x = x*weight[0]
        fea_w.append(x)

        x = F.relu(self.rn(x))
        

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features = self.sfs[0].features*weight[1]
        self.sfs[1].features = self.sfs[1].features*weight[2]
        self.sfs[2].features = self.sfs[2].features*weight[3]
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        x = x*weight[4]
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x = x*weight[5]
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x = x*weight[6]
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x = x*weight[7]
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x = x*weight[8]
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)
        return x_out,fea,fea_w

    def close(self):
        for sf in self.sfs: sf.remove() 


### 预训练res unet ==> my_fss ours2 ### 
class my_fss_ours2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conditioner = my_conditioner_ours2()
        self.segmentor = my_segmentor_ours2()

    def forward(self, input1, input2):
        weights = self.conditioner(input1)
        segment = self.segmentor(input2, weights)
        return segment

class my_conditioner_ours2(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(2)

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        #weight
        self.se1 = nn.Conv2d(3,1,kernel_size=1, padding=0)
        self.se2 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se3 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se4 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se5 = nn.Conv2d(256,1,kernel_size=1, padding=0)
        self.se6 = nn.Conv2d(128,1,kernel_size=1, padding=0)
        self.se7 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se8 = nn.Conv2d(64,1,kernel_size=1, padding=0)
        self.se9 = nn.Conv2d(64,1,kernel_size=1, padding=0)

    def forward(self, x):

        x = self.ec_block(x)
        w1 = torch.sigmoid(self.se1(x))       
        x = F.relu(self.rn(x))
        w5 = torch.sigmoid(self.se5(x))

        x = self.up2(x, self.sfs[2].features)
        w6 = torch.sigmoid(self.se6(x))
        x = self.up3(x, self.sfs[1].features)
        w7 = torch.sigmoid(self.se7(x))
        x = self.up4(x, self.sfs[0].features)
        w8 = torch.sigmoid(self.se8(x))

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = F.relu(self.bn(self.conv(x_out)))
        w9 = torch.sigmoid(self.se9(x_out))

        w2 = torch.sigmoid(self.se2(self.sfs[0].features))
        w3 = torch.sigmoid(self.se3(self.sfs[1].features))
        w4 = torch.sigmoid(self.se4(self.sfs[2].features))
 

        return (w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class my_segmentor_ours2(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 7

        base_model = resnet.resnet18

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        #self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(1)
    
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(1)

        self.QAM_e1 = QueryAttentionModule(3)
        self.QAM_e2 = QueryAttentionModule(64)
        self.QAM_e3 = QueryAttentionModule(64)
        self.QAM_e4 = QueryAttentionModule(128)
        self.QAM_e5 = QueryAttentionModule(256)
        self.QAM_e6 = QueryAttentionModule(128)
        self.QAM_e7 = QueryAttentionModule(64)
        self.QAM_e8 = QueryAttentionModule(64)
        self.QAM_e9 = QueryAttentionModule(64)  

        # Weight Fusion Module
        self.WFM_e1 = WeightFusionModule()
        self.WFM_e2 = WeightFusionModule()
        self.WFM_e3 = WeightFusionModule()
        self.WFM_e4 = WeightFusionModule()
        self.WFM_e5 = WeightFusionModule()
        self.WFM_e6 = WeightFusionModule()
        self.WFM_e7 = WeightFusionModule()
        self.WFM_e8 = WeightFusionModule()
        self.WFM_e9 = WeightFusionModule()


    def forward(self, x, weight):
        x = self.ec_block(x) 
        w_q1 = self.QAM_e1(x) 
        w1 = self.WFM_e1(weight[0],w_q1)
        x = x*w1   
        x = F.relu(self.rn(x))
        w_q5 = self.QAM_e5(x) 
        w5 = self.WFM_e5(weight[4],w_q5)
        x = x*w5  

        x = self.up2(x, self.sfs[2].features)
        w_q6 = self.QAM_e6(x) 
        w6 = self.WFM_e6(weight[5],w_q6)
        x = x*w6  
        x = self.up3(x, self.sfs[1].features)
        w_q7 = self.QAM_e7(x) 
        w7 = self.WFM_e7(weight[6],w_q7)
        x = x*w7   
        x = self.up4(x, self.sfs[0].features)
        w_q8 = self.QAM_e8(x) 
        w8 = self.WFM_e8(weight[7],w_q8)
        x = x*w8   

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        w_q9 = self.QAM_e9(x) 
        w9 = self.WFM_e9(weight[8],w_q9)
        x = x*w9  

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        w_q2 = self.QAM_e2(self.sfs[0].features) 
        w2 = self.WFM_e2(weight[1],w_q2)
        self.sfs[0].features = self.sfs[0].features*w2  

        w_q3 = self.QAM_e3(self.sfs[1].features) 
        w3= self.WFM_e3(weight[2],w_q3)
        self.sfs[1].features = self.sfs[1].features*w3  

        w_q4 = self.QAM_e4(self.sfs[2].features) 
        w4 = self.WFM_e4(weight[3],w_q4)
        self.sfs[2].features = self.sfs[2].features*w4 
        

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 


class UnetBlock_tran10_bn(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out*2, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.bn3 = nn.BatchNorm2d(up_out)
        #self.scSE = scSEBlock(up_out)

    def forward(self, up_p, x_p):

        up_p = F.upsample(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = F.relu(self.bn3(self.x_conv3(up_p)))
        
        cat_p = torch.cat([up_p, x_p], dim=1)
        
        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return cat_p

class UnetBlock_tran10_bn_noskip(nn.Module):
    def __init__(self, up_in,up_out):
        super().__init__()

        self.x_conv3 = nn.Conv2d(up_in, up_out, kernel_size=3, padding=1)
        self.x_conv4 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)
        self.x_conv5 = nn.Conv2d(up_out, up_out, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(up_out)
        self.bn2 = nn.BatchNorm2d(up_out)
        self.bn3 = nn.BatchNorm2d(up_out)
        #self.scSE = scSEBlock(up_out)

    def forward(self, up_p, x_p):

        up_p = F.upsample(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        up_p = F.relu(self.bn3(self.x_conv3(up_p)))
        
        #cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = up_p

        cat_p = self.x_conv4(cat_p)
        cat_p = F.relu(self.bn(cat_p))
        
        cat_p = self.x_conv5(cat_p)
        cat_p = F.relu(self.bn2(cat_p))
        
        return cat_p

class expand_compress_block3(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.conv_expand = nn.Conv2d(n_in, 32, kernel_size=3, padding=1)
        self.conv_compress = nn.Conv2d(32, 3, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(32)
        self.bn_out = nn.BatchNorm2d(3)


    def forward(self, x_in):
        x = self.conv_expand(x_in)
        x = F.relu(self.bn(x))
        x = self.conv_compress(x)
        return F.relu(self.bn_out(x))  

class output_block(nn.Module):
    def __init__(self, n_in,n_out):
        super().__init__()

        self.conv = nn.Conv2d(n_in, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64,n_out, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(n_out)  

    def forward(self, x_in):
        
        x = self.conv(x_in)
        x = F.relu(self.bn(x))
        output = self.bn_out(self.conv_out(x))
        return output 

class SaveFeatures():
    features = None 

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class _CNonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, 
        bn_layer=True,shortcut =True,cos_dis=False):
        super(_CNonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        self.shortcut = shortcut
        self.cos_dis = cos_dis

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        bs = s_x.size(0)
        n_class = bs //2
        batch_size = 2
        _sx = s_x
        _qx = q_x

        z = []
        for i in range(n_class):
            q_x = _qx[i*batch_size:(i+1)*batch_size]
            s_x = _sx[i*batch_size:(i+1)*batch_size] 

            g_x = self.g(q_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(s_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(q_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)

            if self.shortcut:
                z.append(W_y + q_x)
            else:
                z.append(W_y)

        z = torch.cat(z)    
        #print(z.shape,_sx.shape)
        return z

class CNONLocalBlock2D(_CNonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True,
        shortcut=True,cos_dis=False):
        super(CNONLocalBlock2D, self).__init__(in_channels, inter_channels=None, sub_sample=True, 
            bn_layer=True,shortcut=True,cos_dis=False)


class fusion_module(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2,in_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        assert s_x.shape[1] == q_x.shape[1]
        x = torch.cat([s_x,q_x],dim=1)
        x_out = self.conv(x)
        return x_out

class SE_AC(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        self.attention = torch.sigmoid(self.conv(s_x))
        x_out = q_x * self.attention
        return x_out

class SE_AC_vis(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        self.attention = torch.sigmoid(self.conv(s_x))
        x_out = q_x * self.attention
        return x_out,self.attention

class SE_AC2(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2,out_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        self.attention = torch.sigmoid(self.conv(c_x))
        x_out = q_x * self.attention
        return x_out

class SE_AC3(nn.Module):
    def __init__(self,fuse_type,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.fuse_type = fuse_type
        if fuse_type == 1:
            self.fuse_conv = nn.Conv2d(in_channels*2,in_channels,kernel_size=kernel_size, padding=padding)
        elif fuse_type == 2:
            self.fuse_ratio = torch.sigmoid(Parameter(torch.zeros(1,1,1,1)).cuda())
        elif fuse_type == 3:
            self.fuse_ratio = torch.sigmoid(Parameter(torch.zeros(1,in_channels,1,1)).cuda())
        
    def forward(self,support_feature,query_feature,support_mask):
        pos_mask = support_mask
        spatial_attention = torch.sigmoid(self.conv(support_feature))
        # print((support_feature * pos_mask).shape,pos_mask.shape)
        # print(torch.sum(support_feature * pos_mask,dim=[-2,-1]).shape,torch.sum(pos_mask,dim=[-2,-1]).shape)
        vec_pos = torch.sum(support_feature * pos_mask,dim=[-2,-1])/(torch.sum(pos_mask,dim=[-2,-1]) + 1e-9)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        #print(query_feature.shape,vec_pos.shape)
        similar_attention = F.cosine_similarity(query_feature,vec_pos,dim=1).unsqueeze(1)
        if self.fuse_type == 1:
            x_spw = query_feature * spatial_attention
            x_sgw = query_feature * similar_attention
            x_w = torch.cat([x_spw,x_sgw],dim=1)
            x_w = self.fuse_conv(x_w)
        elif self.fuse_type == 2 or self.fuse_type == 3:
            fuse_attention = spatial_attention*self.fuse_ratio + similar_attention*(1-self.fuse_ratio)
            x_w = query_feature * fuse_attention 
        
        return x_w

class SE_AC_nlc(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super().__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)


    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(s_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        z = self.W(y)
        #z = z + s_x
        z = torch.sigmoid(z)
        
        result = q_x*z
 
        return result

class SE_AC_nlc2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super().__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        self.se = SE_AC(in_channels=in_channels)


    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(s_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        z = self.W(y)
        z = z + s_x
        result = self.se(z,q_x)

        return result


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, 
        bn_layer=True,shortcut =True,cos_dis=False):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # if dimension == 3:
        #     conv_nd = nn.Conv3d
        #     max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        #     bn = nn.BatchNorm3d
        # elif dimension == 2:
            #conv_nd = nn.Conv2d
            #max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            #bn = nn.BatchNorm2d
        # else:
        #     conv_nd = nn.Conv1d
        #     max_pool_layer = nn.MaxPool1d(kernel_size=(2))
        #     bn = nn.BatchNorm1d

        # self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #  kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        self.shortcut = shortcut
        self.cos_dis = cos_dis

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(q_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(s_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(q_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        W_y = self.W(y)
        if self.shortcut:
            z = W_y + q_x
        else:
            z = W_y

        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True,
        shortcut=True,cos_dis=False):
        super(NONLocalBlock2D, self).__init__(in_channels, inter_channels=None, sub_sample=True, 
            bn_layer=True,shortcut=True,cos_dis=False)

class NLC_AC2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super().__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)


    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        z = self.W(y)
        z = torch.sigmoid(z)
        
        result = q_x*z
 
        return result

class NLC_AC22(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super().__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        self.se = SE_AC(in_channels)

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        z = self.W(y) + s_x 
    
        #z = torch.sigmoid(z)
        
        result = self.se(z,q_x)
 
        return result

class NLC_AC3(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True):
        super().__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        # self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        #    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(2, 2)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)


    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        
        g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()
        #print(np.unique(self.matrix.cpu().numpy()))

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        z = self.W(y)
        z = torch.sigmoid(z)
        
        result = q_x*z
 
        return result






