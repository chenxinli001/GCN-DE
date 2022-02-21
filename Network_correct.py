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

import resnet

class uni_conditioner_fea(nn.Module):

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
 

### 预训练res unet ==> my_fss 中间层输出 ###
class fss_fea(nn.Module):
    def __init__(self,pretrain,SE_type=None):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea(pretrain,SE_type)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea(nn.Module):

    def __init__(self,pretrain,SE_type):
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
        # if SE_type == 1:
        self.w1 = SE_AC(3)
        self.w2 = SE_AC(64)
        self.w3 = SE_AC(64)
        self.w4 = SE_AC(128)
        self.w5 = SE_AC(256)
        self.w6 = SE_AC(128)
        self.w7 = SE_AC(64)
        self.w8 = SE_AC(64)
        self.w9 = SE_AC(64)
        # elif SE_type == 2:
        #     self.w1 = SE_AC2(3)
        #     self.w2 = SE_AC2(64)
        #     self.w3 = SE_AC2(64)
        #     self.w4 = SE_AC2(128)
        #     self.w5 = SE_AC2(256)
        #     self.w6 = SE_AC2(128)
        #     self.w7 = SE_AC2(64)
        #     self.w8 = SE_AC2(64)
        #     self.w9 = SE_AC2(64)
        # elif SE_type == 3:
        #     self.w1 = SE_AC3(3)
        #     self.w2 = SE_AC3(64)
        #     self.w3 = SE_AC3(64)
        #     self.w4 = SE_AC3(128)
        #     self.w5 = SE_AC3(256)
        #     self.w6 = SE_AC3(128)
        #     self.w7 = SE_AC3(64)
        #     self.w8 = SE_AC3(64)
        #     self.w9 = SE_AC3(64)
        # elif SE_type == 4:
        #     self.w1 = SE_AC4(3)
        #     self.w2 = SE_AC4(64)
        #     self.w3 = SE_AC4(64)
        #     self.w4 = SE_AC4(128)
        #     self.w5 = SE_AC4(256)
        #     self.w6 = SE_AC4(128)
        #     self.w7 = SE_AC4(64)
        #     self.w8 = SE_AC4(64)
        #     self.w9 = SE_AC4(64)
        # elif SE_type == 5:
        #     self.w1 = SE_AC5(256,3)
        #     self.w2 = SE_AC5(128,64)
        #     self.w3 = SE_AC5(64,64)
        #     self.w4 = SE_AC5(32,128)
        #     self.w5 = SE_AC5(16,256)
        #     self.w6 = SE_AC5(32,128)
        #     self.w7 = SE_AC5(64,64)
        #     self.w8 = SE_AC5(128,64)
        #     self.w9 = SE_AC5(256,64)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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

class fss_fea_ac(nn.Module):
    def __init__(self,pretrain):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_ac(pretrain)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature,ac = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature,ac

class segmentor_fea_ac(nn.Module):

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
        self.w1 = SE_AC(3,ac_out=True)
        self.w2 = SE_AC(64,ac_out=True)
        self.w3 = SE_AC(64,ac_out=True)
        self.w4 = SE_AC(128,ac_out=True)
        self.w5 = SE_AC(256,ac_out=True)
        self.w6 = SE_AC(128,ac_out=True)
        self.w7 = SE_AC(64,ac_out=True)
        self.w8 = SE_AC(64,ac_out=True)
        self.w9 = SE_AC(64,ac_out=True)
        

    def forward(self, x, s_fea):
        fea = []
        fea_w = []
        x = self.ec_block(x)
        fea.append(x)
        x,ac1 = self.w1(s_fea[0],x)
        fea_w.append(x)

        x = F.relu(self.rn(x))

        fea.append(self.sfs[0].features)
        fea.append(self.sfs[1].features)
        fea.append(self.sfs[2].features)
        self.sfs[0].features,ac2 = self.w2(s_fea[1],self.sfs[0].features)
        self.sfs[1].features,ac3 = self.w3(s_fea[2],self.sfs[1].features)
        self.sfs[2].features,ac4 = self.w4(s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        #print(x.shape)
        x,ac5 = self.w5(s_fea[4],x)
        #print(x.shape)
        fea_w.append(x)

        x = self.up2(x, self.sfs[2].features)
        fea.append(x)
        x,ac6 = self.w6(s_fea[5],x)
        fea_w.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea.append(x)
        x,ac7 = self.w7(s_fea[6],x)
        fea_w.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea.append(x)
        x,ac8 = self.w8(s_fea[7],x)
        fea_w.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea.append(x)
        x,ac9 = self.w9(s_fea[8],x)
        fea_w.append(x)

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        assert len(fea) == len(fea_w)

        ac2 = F.upsample(ac2, scale_factor=2, mode='bilinear', align_corners=True)
        ac3 = F.upsample(ac3, scale_factor=4, mode='bilinear', align_corners=True)
        ac4 = F.upsample(ac4, scale_factor=8, mode='bilinear', align_corners=True)
        ac5 = F.upsample(ac5, scale_factor=16, mode='bilinear', align_corners=True)
        ac6 = F.upsample(ac6, scale_factor=8, mode='bilinear', align_corners=True)
        ac7 = F.upsample(ac7, scale_factor=4, mode='bilinear', align_corners=True)
        ac8 = F.upsample(ac8, scale_factor=2, mode='bilinear', align_corners=True)

        return x_out,fea,fea_w,[ac1,ac2,ac3,ac4,ac5,ac6,ac7,ac8,ac9]

    def close(self):
        for sf in self.sfs: sf.remove() 

class fss_lastfea(nn.Module):
    def __init__(self,pretrain,SE_type=None):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea(pretrain,SE_type)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature[-1],q_feature[-1],qw_feature[-1]



#### 预训练res unet ==> my_fss 中间层输出 + nonlocal注意力###
#### SE + non-local交互 输出Q的加权和+1*1卷积 add Q
class fss_fea_nlc(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_nlc(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = nlc_module(nlc_type,3,in_spatial=256*256)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = nlc_module(nlc_type,64,in_spatial=128*128)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = nlc_module(nlc_type,64,in_spatial=64*64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = nlc_module(nlc_type,128,in_spatial=32*32)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = nlc_module(nlc_type,256,in_spatial=16*16)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = nlc_module(nlc_type,128,in_spatial=32*32)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = nlc_module(nlc_type,64,in_spatial=64*64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = nlc_module(nlc_type,64,in_spatial=128*128) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = nlc_module(nlc_type,64,in_spatial=256*256)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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

class nlc_module(nn.Module):
    def __init__(self, nlc_type,in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        self.nlc_type = nlc_type

        assert dimension in [1, 2, 3]

        if nlc_type != 7:
            self.dimension = dimension
            self.sub_sample = sub_sample

            if nlc_type in [4,5,6]:
                self.in_channels = in_channels*2
            else:
                self.in_channels = in_channels

            self.inter_channels = inter_channels

            if self.inter_channels is None:
                if nlc_type == 4 or nlc_type == 5 or nlc_type == 6:
                    self.inter_channels = in_channels 
                else:
                    self.inter_channels = in_channels // 2
                if self.inter_channels == 0:
                    self.inter_channels = 1

            assert nlc_type in [1,2,3,4,5,6]
            if nlc_type in [1,6]:
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
            elif nlc_type in [2,3,4,5]:
                self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                                kernel_size=1, stride=1, padding=0)
                nn.init.constant_(self.W.weight, 0)
                nn.init.constant_(self.W.bias, 0)

            if nlc_type != 5:
                self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)
            else:
                self.theta = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
                self.down_rate = down_rate

            if sub_sample:
                self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
                self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            else:
                self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
                self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
            
            self.shortcut = shortcut
            if nlc_type == 6:
                self.se_module = SE_AC(in_channels*2)

        else:
            in_channel = in_channels*2
            self.in_channel = in_channel
            self.inter_channel = in_channel//8
            self.in_spatial = in_spatial
            self.inter_spatial = in_spatial//8
            self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
            self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
            self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
            self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//8,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//8),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//8, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        print(self.nlc_type)
        if self.nlc_type == 1:
            g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)
            if self.shortcut:
                z = W_y + q_x
            else:
                z = W_y
            self.W_y = W_y
            self.q_x = q_x

        elif self.nlc_type == 2:
            g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)
            z = q_x * torch.sigmoid(W_y)

        elif self.nlc_type == 3:
            g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(s_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)
            z = q_x * torch.sigmoid(W_y)
        
        elif self.nlc_type == 4:
            c_x = torch.cat([s_x,q_x],dim=1)
            g_x = self.g(c_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(c_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(c_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)
            attention = torch.sigmoid(W_y)
            z = q_x * attention 
            self.W_y = W_y
            self.q_x = q_x
            self.am = attention
        
        elif self.nlc_type == 5:
            c_x = torch.cat([s_x,q_x],dim=1)
            
            g_x = self.g(c_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(c_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(c_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, c_x.shape[-2]//self.down_rate, c_x.shape[-1]//self.down_rate)
            W_y = self.W(y)
            W_y = F.upsample(W_y, scale_factor=self.down_rate, mode='bilinear', align_corners=True)
            z = q_x * torch.sigmoid(W_y)
            self.W_y = W_y
            self.q_x = q_x
        
        elif self.nlc_type == 6:
            c_x = torch.cat([s_x,q_x],dim=1)
            #print(c_x.shape)
            g_x = self.g(c_x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            theta_x = self.theta(c_x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(c_x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)
            f_div_C = F.softmax(f, dim=-1)
            self.matrix = f_div_C.detach()

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
            W_y = self.W(y)
            if self.shortcut:
                z = W_y + c_x
            else:
                z = W_y
            self.W_y = W_y
            self.q_x = q_x
            #print(z.shape)
            #print(q_x.shape)

            z = self.se_module(z,q_x)

        elif self.nlc_type == 7:
            x = torch.cat([s_x,q_x],dim=1)
            b, c, h, w = x.size()
		
            theta_xs = self.theta_spatial(x)	
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)
            Gs_out = Gs.view(b, h*w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)
        
            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            z = F.sigmoid(W_ys.expand_as(q_x)) * q_x
                
        return z


class fss_fea_nlc1(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)
        assert SE_type == 1
        assert nlc_type == 1

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc1(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature,matrix = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature,matrix

class segmentor_fea_nlc1(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = nlc_module1(3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = nlc_module1(64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = nlc_module1(64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = nlc_module1(128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = nlc_module1(256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = nlc_module1(128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = nlc_module1(64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = nlc_module1(64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = nlc_module1(64)


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
        #x = self.w5(s_fea[4],x)
        x,f_affn = self.w5(s_fea[4],x)
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
        return x_out,fea,fea_w,f_affn

    def close(self):
        for sf in self.sfs: sf.remove() 

class nlc_module1(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=False, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        #self.nlc_type = nlc_type

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

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
    
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
        theta_x_norm = nn.functional.normalize(theta_x,dim=1)
        theta_x_norm = theta_x_norm.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        phi_x_norm = nn.functional.normalize(phi_x,dim=1)
        
        f_cosdist = torch.matmul(theta_x_norm, phi_x_norm)
        
        f_affn = 0.5*(f_cosdist+1)  #   f_affn 和 由Support和Query构成的关联矩阵的GroundTruth 之间 做L2 损失
        
        f_div_C = F.softmax(f_affn, dim=-1)
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

        return z,f_affn

    # def forward(self, s_x,q_x):
    #     '''
    #     :param x: (b, c, t, h, w)
    #     :return:
    #     '''
    #     assert(s_x.shape ==q_x.shape)
    #     batch_size = s_x.size(0)
    #     # print(self.nlc_type)
  
    #     g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
    #     g_x = g_x.permute(0, 2, 1)

    #     theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
    #     theta_x = theta_x.permute(0, 2, 1)
    #     phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
    #     f = torch.matmul(theta_x, phi_x)
    #     f_div_C = F.softmax(f, dim=-1)
    #     self.matrix = f_div_C.detach()

    #     y = torch.matmul(f_div_C, g_x)
    #     y = y.permute(0, 2, 1).contiguous()
    #     y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
    #     W_y = self.W(y)
    #     if self.shortcut:
    #         z = W_y + q_x
    #     else:
    #         z = W_y
    #     self.W_y = W_y
    #     self.q_x = q_x
    #     return z


class fss_fea_nlc4(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)

        assert SE_type == 1 and nlc_type==4

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc4(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_nlc4(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:  
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = nlc_module4(3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = nlc_module4(64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = nlc_module4(64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = nlc_module4(128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = nlc_module4(256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = nlc_module4(128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = nlc_module4(64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = nlc_module4(64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = nlc_module4(64)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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

class nlc_module4(nn.Module):
    def __init__(self,in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()


        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels*2
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels 


        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0)


        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
        
        self.shortcut = shortcut

  
    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)

        c_x = torch.cat([s_x,q_x],dim=1)
        g_x = self.g(c_x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(c_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(c_x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        self.matrix = f_div_C.detach()

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *s_x.size()[2:])
        W_y = self.W(y)
        attention = torch.sigmoid(W_y)
        z = q_x * attention 
        self.W_y = W_y
        self.q_x = q_x
        self.am = attention
        
        return z


class fss_fea_nlc8(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)
        assert SE_type == 1
        assert nlc_type == 8

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc8(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        s_mask = input1[:,1]
        segment,q_feature,qw_feature,weight = self.segmentor(input2, s_feature,s_mask)
        return segment,s_feature,q_feature,qw_feature,weight

class segmentor_fea_nlc8(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = nlc_module8(3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = nlc_module8(64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = nlc_module8(64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = nlc_module8(128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = nlc_module8(256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = nlc_module8(128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = nlc_module8(64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = nlc_module8(64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = nlc_module8(64)


    def forward(self, x, s_fea,s_mask):
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
        #print(x.shape)
        #print(s_mask.shape)
        s_mask_down4 = F.interpolate(s_mask.unsqueeze(1), size=[16, 16],mode='bilinear')
        #print(s_mask_down4.shape)self.x_conv3(up_p)
        #weight = None
        #x = 0*x
        x,weight = self.w5(s_fea[4],x,s_mask_down4)

        #print(x.shape)
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
        return x_out,fea,fea_w,weight

    def close(self):
        for sf in self.sfs: sf.remove() 

class nlc_module8(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        #self.nlc_type = nlc_type

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

        # if sub_sample:
        #     self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        #     self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        # else:
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
     

    def forward(self, s_x,q_x,s_mask):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        # print(self.nlc_type)
  
        # g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        #theta_x = q_x.view(batch_size,self.in_channels,-1).permute(0,2,1)
        #phi_x = s_x.view(batch_size,self.in_channels,-1)

        f = torch.matmul(theta_x, phi_x)
        
        #f_div_C = F.softmax(f, dim=-1) #b,HW,hw
        f_div_C = f
        
        
        #self.matrix = f_div_C.detach()

        #s_mask = F.interpolate(s_mask.unsqueeze(-1), size=[16,16])
        #print(s_mask.shape)
        s_mask = s_mask.view(batch_size,-1).unsqueeze(-1) #b,hw,1
        #print(s_mask.shape)
        #print(s_mask.detach().cpu().numpy().min(),s_mask.detach().cpu().numpy().max())
        
        fg = torch.matmul(f_div_C,s_mask) / s_mask.sum(dim=1).unsqueeze(1) #b,HW,1
        bg = torch.matmul(f_div_C,1-s_mask) /(1-s_mask).sum(dim=1).unsqueeze(1) #b,HW,1
        self.fg = fg.detach()
        self.bg = bg.detach()
        # matrix = torch.cat([fg,bg],dim=-1) #b,hw,2
        # attention = torch.softmax(matrix,dim=-1)[:,:,0] #b,hw
        attention = torch.sigmoid(fg-bg)
        

        # print(fg.shape)
        # #print(torch.sum(fg,dim=-1))
        #print("#")
        #print(fg[0].detach().cpu().numpy().min(),fg[0].detach().cpu().numpy().max())
        # fg = F.softmax(fg/0.005,dim=-1)
        # print(torch.min(fg,dim=-1))
        # min_fg = torch.min(fg,dim=-1)[0].view(batch_size,1)
        # max_fg = torch.max(fg,dim=-1)[0].view(batch_size,1)
        # fg = (fg - min_fg +1e-12) / (max_fg - min_fg+1e-12) 
        # print(fg.shape)
        #print(fg[0].detach().cpu().numpy().min(),fg[0].detach().cpu().numpy().max())
        # print("##")

        # fg = fg.view(batch_size,*s_x.size()[2:]).unsqueeze(1) #b,1,h,w
        attention = attention.view(batch_size,*s_x.size()[2:]).unsqueeze(1)
        #attention = torch.sigmoid(fg) 

        self.attention = attention.detach()
        #print(self.attention.cpu().numpy().min(),self.attention.cpu().numpy().max())

        # if self.shortcut:
        #     z = (1+fg) * q_x
        # else:
        #     z = fg*q_x

        z = attention * q_x
        #z = q_x

        self.q_x = q_x.detach()
        self.z = z.detach()

        return z,attention


class fss_fea_nlc9(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)
        assert SE_type == 1
        assert nlc_type == 9

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc9(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        s_mask = input1[:,1]
        segment,q_feature,qw_feature,f_affn = self.segmentor(input2, s_feature,s_mask)
        return segment,s_feature,q_feature,qw_feature,f_affn

class segmentor_fea_nlc9(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = nlc_module9(3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = nlc_module9(64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = nlc_module9(64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = nlc_module9(128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = nlc_module9(256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = nlc_module9(128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = nlc_module9(64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = nlc_module9(64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = nlc_module9(64)


    def forward(self, x, s_fea,s_mask):
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
        #print(x.shape)
        #print(s_mask.shape)
        s_mask_down4 = F.interpolate(s_mask.unsqueeze(1), size=[16, 16],mode='bilinear')
        #print(s_mask_down4.shape)
        x,f_affn = self.w5(s_fea[4],x,s_mask_down4)
        #print(x.shape)
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
        return x_out,fea,fea_w,f_affn

    def close(self):
        for sf in self.sfs: sf.remove() 

class nlc_module9(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        #self.nlc_type = nlc_type

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

        # if sub_sample:
        #     self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        #     self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        # else:
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
     

    def forward(self, s_x,q_x,s_mask):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        # print(self.nlc_type)
  
        # g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x_norm = nn.functional.normalize(theta_x,dim=1)
        theta_x_norm = theta_x_norm.permute(0, 2, 1)
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        phi_x_norm = nn.functional.normalize(phi_x,dim=1)
        
        f_cosdist = torch.matmul(theta_x_norm, phi_x_norm)
        
        f_affn = 0.5*(f_cosdist+1)  #   f_affn 和 由Support和Query构成的关联矩阵的GroundTruth 之间 做L2 损失
        
        #f_div_C = F.softmax(f_affn, dim=-1)
        f_affn = f_cosdist
        f_div_C = f_affn
        #self.matrix = f_div_C.detach()

        # theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        
        # f_div_C = F.softmax(f, dim=-1) #b,HW,hw
        # #f_div_C = f
        # self.matrix = f_div_C.detach()

        #s_mask = F.interpolate(s_mask.unsqueeze(-1), size=[16,16])
        #print(s_mask.shape)
        
        s_mask = s_mask.view(batch_size,-1).unsqueeze(-1) #b,hw,1
        #print(s_mask.shape)
        #print(s_mask.detach().cpu().numpy().min(),s_mask.detach().cpu().numpy().max())
        
        # fg = torch.matmul(f_div_C,s_mask).squeeze(-1) #b,HW
        # fg = fg.view(batch_size,*s_x.size()[2:]).unsqueeze(1) #b,1,h,w
        fg = torch.matmul(f_div_C,s_mask) /s_mask.sum(dim=1).unsqueeze(1) #b,HW,1
        bg = torch.matmul(f_div_C,1-s_mask) /(1-s_mask).sum(dim=1).unsqueeze(1) #b,HW,1
        attention = torch.sigmoid(fg-bg).view(batch_size,1,*s_x.size()[2:])
        self.attention = attention.detach()

        
        #self.attention = fg.detach()
        #print(self.attention.cpu().numpy().min(),self.attention.cpu().numpy().max())

        # if self.shortcut:
        #     z = (1+fg) * q_x
        # else:
        #     z = fg*q_x
        z = attention * q_x
        self.q_x = q_x.detach()
        self.z = z.detach()

        return z,f_affn


class fss_fea_nlc10(nn.Module):
    def __init__(self,pretrain=True,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_type, nlc_layer)

        assert SE_type == 1 and nlc_type==10

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc10(pretrain,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_nlc10(nn.Module):

    def __init__(self,pretrain,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:  
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = GCNet2(3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = GCNet2(64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = GCNet2(64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = GCNet2(128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = GCNet2(256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = GCNet2(128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = GCNet2(64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = GCNet2(64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = GCNet2(64)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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

class fss_fea_GC(nn.Module):
    def __init__(self,pretrain=True,input_type=None):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_GC(pretrain,input_type)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_GC(nn.Module):

    def __init__(self,pretrain,input_type):
        super().__init__()
        print('Input type is {}'.format(input_type))
      
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        self.w1 = GCNet5(3,input_type)
        self.w2 = GCNet5(64,input_type)
        self.w3 = GCNet5(64,input_type)
        self.w4 = GCNet5(128,input_type)
        self.w5 = GCNet5(256,input_type)
        self.w6 = GCNet5(128,input_type)
        self.w7 = GCNet5(64,input_type)
        self.w8 = GCNet5(64,input_type)
        self.w9 = GCNet5(64,input_type)



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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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


class GCNet1(nn.Module):
    def __init__(self, channel, r=2):
        '''
        :param channel: 输入通道
        :param r: transform 模块中显著降低参数量的通道降低率
        '''
        super().__init__()
        self.context_modeling = nn.Sequential(nn.Conv2d(channel*2, 1, 1), )
        self.transform = nn.Sequential(nn.Conv2d(channel*2, channel // r, 1),
                                       nn.LayerNorm([channel // r, 1, 1], elementwise_affine=True),
                                       nn.ReLU(),
                                       nn.Conv2d(channel // r, channel, 1))

    def forward(self, s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        batch, channel, height, width = c_x.size()

        xin_ = self.context_modeling(c_x)
        xin_ = xin_.view(batch, 1, height * width)  # N, 1, HXW
        xin_ = F.softmax(xin_, dim=2).unsqueeze(-1)  # N, 1, HXW, 1
        xin_ = torch.matmul(c_x.view(batch, channel, height * width).unsqueeze(1),
                            xin_)  # N,1,C,HXW N,1,HXW,1 -> N,1,C,1
        xin_ = xin_.permute(0, 2, 1, 3)  # N,C,1,1
        xin_ = self.transform(xin_)
        return q_x + xin_

class GCNet2(nn.Module):
    def __init__(self, channel, r=2):
        '''
        :param channel: 输入通道
        :param r: transform 模块中显著降低参数量的通道降低率
        '''
        super().__init__()
        self.context_modeling = nn.Sequential(nn.Conv2d(channel, 1, 1), )
        self.transform = nn.Sequential(nn.Conv2d(channel, channel // r, 1),
                                       nn.LayerNorm([channel // r, 1, 1], elementwise_affine=True),
                                       nn.ReLU(),
                                       nn.Conv2d(channel // r, channel, 1))

    def forward(self, s_x,q_x):
        #c_x = torch.cat([s_x,q_x],dim=1)
        batch, channel, height, width = s_x.size()

        xin_ = self.context_modeling(s_x)
        xin_ = xin_.view(batch, 1, height * width)  # N, 1, HXW
        xin_ = F.softmax(xin_, dim=2).unsqueeze(-1)  # N, 1, HXW, 1
        xin_ = torch.matmul(q_x.view(batch, channel, height * width).unsqueeze(1),
                            xin_)  # N,1,C,HXW N,1,HXW,1 -> N,1,C,1
        xin_ = xin_.permute(0, 2, 1, 3)  # N,C,1,1
        xin_ = self.transform(xin_)
        return q_x + xin_

class GCNet3(nn.Module):
    def __init__(self, channel, input_type,r=2):
        '''
        :param channel: 输入通道
        :param r: transform 模块中显著降低参数量的通道降低率
        '''
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['S', 'C']
        if self.input_type == 'C':
            in_channel = channel*2
        else:
            in_channel = channel
            
        self.context_modeling = nn.Sequential(nn.Conv2d(in_channel, 1, 1), )
        self.transform = nn.Sequential(nn.Conv2d(in_channel, channel // r, 1),
                                    nn.LayerNorm([channel // r, 1, 1], elementwise_affine=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channel // r, channel, 1))
        self.conv = nn.Conv2d(channel,1,kernel_size=1, padding=0)
       

    def forward(self, s_x,q_x):
        #c_x = torch.cat([s_x,q_x],dim=1
        if self.input_type == 'S':
            x = s_x
        elif self.input_type == 'C':
            x = torch.cat([s_x,q_x],dim=1)

        batch, channel, height, width = x.size()

        xin_ = self.context_modeling(x)
        xin_ = xin_.view(batch, 1, height * width)  # N, 1, HXW
        xin_ = F.softmax(xin_, dim=2).unsqueeze(-1)  # N, 1, HXW, 1
        xin_ = torch.matmul(x.view(batch, channel, height * width).unsqueeze(1),
                            xin_)  # N,1,C,HXW N,1,HXW,1 -> N,1,C,1
        xin_ = xin_.permute(0, 2, 1, 3)  # N,C,1,1
        xin_ = self.transform(xin_)
        enhance_s = s_x + xin_
        enhance_s = self.conv(enhance_s)
        self.attention = torch.sigmoid(enhance_s)
        out = q_x * self.attention
        return out

class GCNet4(nn.Module):
    def __init__(self, channel, input_type,r=2):
        '''
        :param channel: 输入通道
        :param r: transform 模块中显著降低参数量的通道降低率
        '''
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['S', 'C']
        if self.input_type == 'C':
            in_channel = channel*2
        else:
            in_channel = channel
            
        self.context_modeling = nn.Sequential(nn.Conv2d(in_channel, 1, 1), )
        self.transform = nn.Sequential(nn.Conv2d(in_channel, channel // r, 1),
                                    nn.LayerNorm([channel // r, 1, 1], elementwise_affine=True),
                                    nn.ReLU(),
                                    nn.Conv2d(channel // r, channel, 1))
        self.conv = nn.Conv2d(channel,1,kernel_size=1, padding=0)
       

    def forward(self, s_x,q_x):
        #c_x = torch.cat([s_x,q_x],dim=1
        if self.input_type == 'S':
            x = s_x
        elif self.input_type == 'C':
            x = torch.cat([s_x,q_x],dim=1)

        batch, channel, height, width = x.size()

        xin_ = self.context_modeling(x)
        xin_ = xin_.view(batch, 1, height * width)  # N, 1, HXW
        xin_ = F.softmax(xin_, dim=2).unsqueeze(-1)  # N, 1, HXW, 1
        xin_ = torch.matmul(x.view(batch, channel, height * width).unsqueeze(1),
                            xin_)  # N,1,C,HXW N,1,HXW,1 -> N,1,C,1
        xin_ = xin_.permute(0, 2, 1, 3)  # N,C,1,1
        xin_ = self.transform(xin_)
        out = q_x + xin_
        #enhance_s = self.conv(enhance_s)
        #self.attention = torch.sigmoid(enhance_s)
        #out = q_x * self.attention
        return out

class GCNet5(nn.Module):
    def __init__(self, channel, input_type,r=2):
        '''
        :param channel: 输入通道
        :param r: transform 模块中显著降低参数量的通道降低率
        '''
        super().__init__()
        self.input_type = input_type
        assert self.input_type in ['S', 'C']
        if self.input_type == 'C':
            in_channel = channel*2
        else:
            in_channel = channel
            
        self.context_modeling = nn.Sequential(nn.Conv2d(in_channel, 1, 1), )
        self.transform = nn.Sequential(nn.Conv2d(in_channel, in_channel // r, 1),
                                    nn.LayerNorm([in_channel // r, 1, 1], elementwise_affine=True),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channel // r, in_channel, 1))
        self.conv = nn.Conv2d(in_channel,1,kernel_size=1, padding=0)
       

    def forward(self, s_x,q_x):
        #c_x = torch.cat([s_x,q_x],dim=1
        if self.input_type == 'S':
            x = s_x
        elif self.input_type == 'C':
            x = torch.cat([s_x,q_x],dim=1)

        batch, channel, height, width = x.size()

        xin_ = self.context_modeling(x)
        xin_ = xin_.view(batch, 1, height * width)  # N, 1, HXW
        xin_ = F.softmax(xin_, dim=2).unsqueeze(-1)  # N, 1, HXW, 1
        xin_ = torch.matmul(x.view(batch, channel, height * width).unsqueeze(1),
                            xin_)  # N,1,C,HXW N,1,HXW,1 -> N,1,C,1
        xin_ = xin_.permute(0, 2, 1, 3)  # N,C,1,1
        xin_ = self.transform(xin_)
        enhance_s = x + xin_
        enhance_s = self.conv(enhance_s)
        self.attention = torch.sigmoid(enhance_s)
        out = q_x * self.attention
        return out



class fuse_module(nn.Module):
    def __init__(self, fuse_type,SE_type,nlc_type,channels):
        super().__init__()

        if SE_type == 1:
            self.se_module = SE_AC(channels)
        elif SE_type == 2:
            self.se_module = SE_AC2(channels)
        
        self.nlc_module = nlc_module(nlc_type,channels)
        self.fuse_type = fuse_type

        if self.fuse_type == 2:
            self.gate = nn.Parameter(torch.Tensor(1,1,1,1))
            self.gate.data.fill_(1)
            setattr(self.gate, 'bin_gate', True)
            # self.fw = torch.sigmoid(nn.Parameter(torch.zeros(1,1,1,1)).cuda())      
        elif self.fuse_type == 3:
            # self.fw = torch.sigmoid(nn.Parameter(torch.zeros(1,channels,1,1)).cuda()) 
            self.gate = nn.Parameter(torch.Tensor(1,channels,1,1))
            self.gate.data.fill_(1)
            setattr(self.gate, 'bin_gate', True)
        elif self.fuse_type == 4:
            # self.fw = torch.sigmoid(nn.Parameter(torch.zeros(1,channels,1,1)).cuda()) 
            self.gate = nn.Parameter(torch.Tensor(1,channels,1,1))
            self.gate.data.fill_(1)
            setattr(self.gate, 'bin_gate', True)


    def forward(self, s_x,q_x):
        se_q = self.se_module(s_x,q_x)
        nlc_q = self.nlc_module(s_x,q_x)
        if self.fuse_type == 1:
            result = se_q + nlc_q
        elif self.fuse_type == 2:
            se_w = self.gate.clamp(0,1)
            nlc_w = 1 - self.gate
            result = se_q*se_w + nlc_q*nlc_w
        elif self.fuse_type == 3:
            se_w = self.gate
            nlc_w = 1 - self.gate
            result = se_q*se_w + nlc_q*nlc_w
        
        return result


#### 预训练res unet ==> my_fss 中间层输出 + nonlocal注意力###
#### SE + non-local交互 输出Q的加权和+1*1卷积 add Q
class fss_fea_fuse(nn.Module):
    def __init__(self,pretrain=True,fuse_type=None,SE_type=None,nlc_type=None,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(fuse_type,SE_type,nlc_type, nlc_layer)

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_fuse(pretrain,fuse_type,SE_type,nlc_type,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_fuse(nn.Module):

    def __init__(self,pretrain,fuse_type,SE_type,nlc_type,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

    
        if 1 not in nlc_layer:
            if SE_type == 1:
                self.w1 = SE_AC(3) 
            elif SE_type == 2:
                self.w1 = SE_AC2(3)
        else:
            self.w1 = fuse_module(fuse_type,SE_type,nlc_type,3)
        if 2 not in nlc_layer:
            if SE_type == 1:
                self.w2 = SE_AC(64) 
            elif SE_type == 2:
                self.w2 = SE_AC2(64)
        else:
            self.w2 = fuse_module(fuse_type,SE_type,nlc_type,64)
        if 3 not in nlc_layer:
            if SE_type == 1:
                self.w3 = SE_AC(64) 
            elif SE_type == 2:
                self.w3 = SE_AC2(64)
        else:
            self.w3 = fuse_module(fuse_type,SE_type,nlc_type,64)
        if 4 not in nlc_layer:
            if SE_type == 1:
                self.w4 = SE_AC(128) 
            elif SE_type == 2:
                self.w4 = SE_AC2(128)
        else:
            self.w4 = fuse_module(fuse_type,SE_type,nlc_type,128)
        if 5 not in nlc_layer:
            if SE_type == 1:
                self.w5 = SE_AC(256) 
            elif SE_type == 2:
                self.w5 = SE_AC2(256)
        else:
            self.w5 = fuse_module(fuse_type,SE_type,nlc_type,256)
        if 6 not in nlc_layer:
            if SE_type == 1:
                self.w6 = SE_AC(128) 
            elif SE_type == 2:
                self.w6 = SE_AC2(128)
        else:
            self.w6 = fuse_module(fuse_type,SE_type,nlc_type,128)
        if 7 not in nlc_layer:
            if SE_type == 1:
                self.w7 = SE_AC(64) 
            elif SE_type == 2:
                self.w7 = SE_AC2(64)
        else:
            self.w7 = fuse_module(fuse_type,SE_type,nlc_type,64)
        if 8 not in nlc_layer:
            if SE_type == 1:
                self.w8 = SE_AC(64) 
            elif SE_type == 2:
                self.w8 = SE_AC2(64)
        else:
            self.w8 = fuse_module(fuse_type,SE_type,nlc_type,64) 
        if 9 not in nlc_layer:
            if SE_type == 1:
                self.w9 = SE_AC(64) 
            elif SE_type == 2:
                self.w9 = SE_AC2(64)
        else:
            self.w9 = fuse_module(fuse_type,SE_type,nlc_type,64)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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


class SE_AC(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0,ac_out=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.ac_out = ac_out
    
    def forward(self,s_x,q_x):
        self.attention = torch.sigmoid(self.conv(s_x))
        x_out = q_x * self.attention
        if self.ac_out:
            return x_out,self.attention
        else:
            return x_out

class SE_AC2(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0,ac_out=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2,out_channels,kernel_size=kernel_size, padding=padding)
        self.ac_out = ac_out

    def forward(self,s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        self.attention = torch.sigmoid(self.conv(c_x))
        x_out = q_x * self.attention
        if self.ac_out:
            return x_out,self.attention
        else:
            return x_out

class SE_AC3(nn.Module):
    def __init__(self,in_channels,out_channels=None,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2,in_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        self.attention = torch.sigmoid(self.conv(c_x))
        x_out = q_x * self.attention
        return x_out


class SE_AC4(nn.Module):
    def __init__(self,in_channels,out_channels=None,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        self.attention = torch.sigmoid(self.conv(s_x))
        x_out = q_x * self.attention
        return x_out

class SE_AC5(nn.Module):
    def __init__(self,size,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.conv_h1 = nn.Conv2d(size,size//4,kernel_size=1,padding=0)
        self.conv_h2 = nn.Conv2d(size//4,size,kernel_size=1,padding=0)
        self.conv_w1 = nn.Conv2d(size,size//4,kernel_size=1,padding=0)
        self.conv_w2 = nn.Conv2d(size//4,size,kernel_size=1,padding=0)
    
    def forward(self,s_x,q_x):
        feature = self.conv(s_x) #b,1,h,w
        feature_h = torch.mean(feature,dim=-1,keepdim=True).permute(0,2,1,3) #b,size,1,1
        feature_w = torch.mean(feature,dim=-2,keepdim=True).permute(0,3,1,2)
        attention_h = torch.sigmoid(self.conv_h2(self.conv_h1(feature_h))).permute(0,2,1,3) #b,1,size,1
        attention_w = torch.sigmoid(self.conv_w2(self.conv_w1(feature_w))).permute(0,2,3,1) #b,1,1,size

        x_out = q_x * attention_h * attention_w
        return x_out

class fss_fea_fuseac(nn.Module):
    def __init__(self,pretrain):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_fuseac(pretrain)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_fuseac(nn.Module):

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
        self.w1 = SE_fuse_AC2(3)
        self.w2 = SE_fuse_AC2(64)
        self.w3 = SE_fuse_AC2(64)
        self.w4 = SE_fuse_AC2(128)
        self.w5 = SE_fuse_AC2(256)
        self.w6 = SE_fuse_AC2(128)
        self.w7 = SE_fuse_AC2(64)
        self.w8 = SE_fuse_AC2(64)
        self.w9 = SE_fuse_AC2(64)


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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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


class SE_fuse_AC1(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.s_conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.q_conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.gate = Parameter(torch.Tensor(1))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)
    
    def forward(self,s_x,q_x):
        self.s_attention = torch.sigmoid(self.s_conv(s_x))
        self.q_attention = torch.sigmoid(self.q_conv(q_x))
        self.fuse_attention = self.gate * self.s_attention + (1-self.gate) * self.q_attention
        x_out = q_x * self.fuse_attention 
        return x_out

class SE_fuse_AC2(nn.Module):
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.s_conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
        self.q_conv = nn.Conv2d(in_channels*2,out_channels,kernel_size=kernel_size, padding=padding)
        self.gate = Parameter(torch.Tensor(1))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)
    
    def forward(self,s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        self.s_attention = torch.sigmoid(self.s_conv(s_x))
        self.q_attention = torch.sigmoid(self.q_conv(c_x))
        self.fuse_attention = self.gate * self.s_attention + (1-self.gate) * self.q_attention
        x_out = q_x * self.fuse_attention 
        return x_out

class fss_fea_nlc_mc(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_layer)
        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc_mc(pretrain,nlc_layer,sub_sample,bn_layer,shortcut)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        s_mask = input1[:,1]
        segment,q_feature,qw_feature,f_affn = self.segmentor(input2, s_feature,s_mask)
        return segment,s_feature,q_feature,qw_feature,f_affn

class segmentor_fea_nlc_mc(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut):
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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        if 1 not in nlc_layer:
            self.w1 = SE_AC(3) 
        else:
            self.w1 = nlc_mc_module2(3)
        if 2 not in nlc_layer:
            self.w2 = SE_AC(64) 
        else:
            self.w2 = nlc_mc_module2(64)
        if 3 not in nlc_layer:
            self.w3 = SE_AC(64) 
        else:
            self.w3 = nlc_mc_module2(64)
        if 4 not in nlc_layer:
            self.w4 = SE_AC(128) 
        else:
            self.w4 = nlc_mc_module2(128)
        if 5 not in nlc_layer:
            self.w5 = SE_AC(256) 
        else:
            self.w5 = nlc_mc_module2(256)
        if 6 not in nlc_layer:
            self.w6 = SE_AC(128) 
        else:
            self.w6 = nlc_mc_module2(128)
        if 7 not in nlc_layer:
            self.w7 = SE_AC(64) 
        else:
            self.w7 = nlc_mc_module2(64)
        if 8 not in nlc_layer:
            self.w8 = SE_AC(64) 
        else:
            self.w8 = nlc_mc_module2(64) 
        if 9 not in nlc_layer:
            self.w9 = SE_AC(64) 
        else:
            self.w9 = nlc_mc_module2(64)


    def forward(self, x, s_fea,s_mask):
        # s_mask_down2 = F.interpolate(s_mask.unsqueeze(1),scale_factor=0.5,mode='nearest')
        # s_mask_down4 = F.interpolate(s_mask_down2,scale_factor=0.5,mode='nearest')
        # s_mask_down8 = F.interpolate(s_mask_down4,scale_factor=0.5,mode='nearest')
        # s_mask_down16 = F.interpolate(s_mask_down8,scale_factor=0.5,mode='nearest')
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
        self.sfs[0].features,f_affn = self.w2(s_fea[1],self.sfs[0].features,s_mask)
        self.sfs[1].features = self.w3(s_fea[2],self.sfs[1].features)
        self.sfs[2].features = self.w4(s_fea[3],self.sfs[2].features)
        fea_w.append(self.sfs[0].features)
        fea_w.append(self.sfs[1].features)
        fea_w.append(self.sfs[2].features)

        fea.append(x)
        

        x = self.w5(s_fea[4],x)
        #print(x.shape)
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
        return x_out,fea,fea_w,f_affn

    def close(self):
        for sf in self.sfs: sf.remove() 

class nlc_mc_module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        #self.nlc_type = nlc_type

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

        # if sub_sample:
        #     self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        #     self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        # else:

        # self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        #             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
     

    def forward(self, s_x,q_x,_s_mask):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)
        # print(self.nlc_type)
  
        # g_x = self.g(s_x).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x_norm = nn.functional.normalize(theta_x,dim=1)
        theta_x_norm = theta_x_norm.permute(0, 2, 1) #b,hw,c
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        phi_x_norm = nn.functional.normalize(phi_x,dim=1) #b,c,hw

        f_cosdist = torch.matmul(theta_x_norm, phi_x_norm) #b,hw,hw
        
        f_affn = 0.5*(f_cosdist+1)  #   f_affn 和 由Support和Query构成的关联矩阵的GroundTruth 之间 做L2 损失
        
        #f_div_C = F.softmax(f_affn, dim=-1)
        #f_affn = f_cosdist
        f_div_C = f_affn
        #self.matrix = f_div_C.detach()

        # theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        
        # f_div_C = F.softmax(f, dim=-1) #b,HW,hw
        # #f_div_C = f
        # self.matrix = f_div_C.detach()

        s_mask = F.interpolate(_s_mask.unsqueeze(1), size=[16,16],mode='nearest')
        s_mask = s_mask.view(batch_size,-1).unsqueeze(-1) #b,hw,1

        # fg = torch.matmul(f_div_C,s_mask).squeeze(-1) #b,HW
        #fg = fg.view(batch_size,*s_x.size()[2:]).unsqueeze(1) #b,1,h,w
        fg = torch.matmul(f_div_C,s_mask) /s_mask.sum(dim=1).unsqueeze(1) #b,HW,1
        bg = torch.matmul(f_div_C,1-s_mask) /(1-s_mask).sum(dim=1).unsqueeze(1) #b,HW,1
        # attention = torch.sigmoid(fg-bg).view(batch_size,1,*s_x.size()[2:])
        attention = (fg-bg).squeeze(-1) #b,HW
        self.origin_attention = attention.detach()
        min_ = torch.min(attention,dim=-1)[0].view(batch_size,1)
        max_ = torch.max(attention,dim=-1)[0].view(batch_size,1)
        attention = (attention - min_) / (max_ - min_) 
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        self.attention = attention.detach()

        assert( self.attention.cpu().numpy().max() == 1 )

        z = attention * q_x
        self.q_x = q_x.detach()
        self.z = z.detach()

        return z,f_affn


class nlc_mc_module2(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()

        #self.nlc_type = nlc_type

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

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
     

    def forward(self, s_x,q_x,_s_mask):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        assert(s_x.shape ==q_x.shape)
        batch_size = s_x.size(0)

        theta_x = self.theta(q_x).view(batch_size, self.inter_channels, -1)
        theta_x_norm = nn.functional.normalize(theta_x,dim=1)
        theta_x_norm = theta_x_norm.permute(0, 2, 1) #b,hw,c
        phi_x = self.phi(s_x).view(batch_size, self.inter_channels, -1)
        phi_x_norm = nn.functional.normalize(phi_x,dim=1) #b,c,hw
        
        s_mask = F.interpolate(_s_mask.unsqueeze(1), scale_factor = 0.25,mode='nearest')
        s_mask = s_mask.view(batch_size,-1).unsqueeze(-1) #b,hw,1


        f_cosdist = torch.matmul(theta_x_norm, phi_x_norm)
        
        f_affn = 0.5*(f_cosdist+1)  #   f_affn 和 由Support和Query构成的关联矩阵的GroundTruth 之间 做L2 损失
        
        f_div_C = f_affn #b,hw,nf

        # fg = torch.matmul(f_div_C,s_mask).squeeze(-1) #b,HW
        #fg = fg.view(batch_size,*s_x.size()[2:]).unsqueeze(1) #b,1,h,w
        fg = torch.matmul(f_div_C,s_mask) /s_mask.sum(dim=1).unsqueeze(1) #b,HW,1
        bg = torch.matmul(f_div_C,1-s_mask) /(1-s_mask).sum(dim=1).unsqueeze(1) #b,HW,1
        self.fg = fg.detach()
        self.bg = bg.detach()
        #attention = torch.sigmoid(fg-bg).view(batch_size,1,*s_x.size()[2:])
        # attention = (fg-bg).squeeze(-1) #b,HW
        attention = F.softmax(torch.cat([fg,bg],dim=-1),dim=-1) #b,HW,1
        #attention = torch.sum(f_div_C,dim=-1) #b,hw
        # self.origin_attention = attention.detach()
        # min_ = torch.min(attention,dim=-1)[0].view(batch_size,1)
        # max_ = torch.max(attention,dim=-1)[0].view(batch_size,1)
        # attention = (attention - min_) / (max_ - min_) 
        # attention = attention.unsqueeze(-1).unsqueeze(-1)
        attention = attention[:,:,0].view(batch_size,*s_x.size()[2:]).unsqueeze(1)
        self.attention = attention.detach()

        # assert( self.attention.cpu().numpy().max() == 1 )

        z = attention * q_x
        self.q_x = q_x.detach()
        self.z = z.detach()

        return z,f_affn

class uni_conditioner_fea_mask(nn.Module):

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

        #self.out = output_block(64,1)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.conv_out = nn.Conv2d(64,1, kernel_size=1, padding=0)
        self.bn_out = nn.BatchNorm2d(1)

        self.mask_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.mask_conv2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mask_conv3= nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mask_conv4 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.mask_conv5 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.mask_conv6 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.mask_conv7 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mask_conv8 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.mask_conv9 = nn.Conv2d(1, 64, kernel_size=3, padding=1)


    def forward(self, x):
        fea_list = []
        w_fea_list = []
        mask_list = []
        mask = x[:,1:2]
        x = x[:,0:1]

        mask1 = torch.sigmoid(self.mask_conv1(mask))

        mask2 = torch.sigmoid(self.mask_conv2(mask)) 
        mask2 = F.interpolate(mask2,scale_factor=0.5,mode='nearest')

        mask3 = torch.sigmoid(self.mask_conv3(mask)) 
        mask3 = F.interpolate(mask3,scale_factor=0.25,mode='nearest')

        mask4 = torch.sigmoid(self.mask_conv4(mask)) 
        mask4 = F.interpolate(mask4,scale_factor=0.125,mode='nearest')

        mask5 = torch.sigmoid(self.mask_conv5(mask)) 
        mask5 = F.interpolate(mask5,scale_factor=0.125*0.5,mode='nearest')

        mask6 = torch.sigmoid(self.mask_conv6(mask)) 
        mask6 = F.interpolate(mask6,scale_factor=0.125,mode='nearest')

        mask7 = torch.sigmoid(self.mask_conv7(mask)) 
        mask7 = F.interpolate(mask7,scale_factor=0.25,mode='nearest')

        mask8 = torch.sigmoid(self.mask_conv8(mask)) 
        mask8 = F.interpolate(mask8,scale_factor=0.5,mode='nearest')

        mask9 = torch.sigmoid(self.mask_conv9(mask)) 

        x = self.ec_block(x)
        fea_list.append(x)
        x = x * mask1
        w_fea_list.append(x)
 
        x = F.relu(self.rn(x))
        fea_list.append(self.sfs[0].features)
        self.sfs[0].features = self.sfs[0].features * mask2
        w_fea_list.append(self.sfs[0].features)
        fea_list.append(self.sfs[1].features)
        self.sfs[1].features = self.sfs[1].features * mask3
        w_fea_list.append(self.sfs[1].features)
        fea_list.append(self.sfs[2].features)
        self.sfs[2].features = self.sfs[2].features * mask4
        w_fea_list.append(self.sfs[2].features)
        fea_list.append(x)
        x = x * mask5
        w_fea_list.append(x)
        

        x = self.up2(x, self.sfs[2].features)
        fea_list.append(x)
        x = x*mask6
        w_fea_list.append(x)
        x = self.up3(x, self.sfs[1].features)
        fea_list.append(x)
        x = x*mask7
        w_fea_list.append(x)
        x = self.up4(x, self.sfs[0].features)
        fea_list.append(x)
        x = x*mask8
        w_fea_list.append(x)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))
        fea_list.append(x)
        x = x*mask9
        w_fea_list.append(x)

        x = self.bn_out(self.conv_out(x))
        x_out = torch.sigmoid(x)

        self.mask_list = [mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,mask9]
        self.fea_list = fea_list
        self.w_fea_list = w_fea_list
        
        return x_out,fea_list,mask_list#,(w1,w2,w3,w4,w5,w6,w7,w8,w9)

    def close(self):
        for sf in self.sfs: sf.remove() 
 

class fss_fea_cross(nn.Module):
    def __init__(self,pretrain):
        super().__init__()

        self.conditioner = uni_conditioner_fea_mask(pretrain)
        self.segmentor = segmentor_fea_cross(pretrain)

    def forward(self, input1, input2):
        s_seg,s_feature,s_mask = self.conditioner(input1)
        q_seg,q_feature,qw_feature = self.segmentor(input2, s_feature)

        return q_seg,s_seg,s_feature,q_feature,qw_feature

class segmentor_fea_cross(nn.Module):

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
        
        self.w1 = cross_fuse_module(3)
        self.w2 = cross_fuse_module(64)
        self.w3 = cross_fuse_module(64)
        self.w4 = cross_fuse_module(128)
        self.w5 = cross_fuse_module(256)
        self.w6 = cross_fuse_module(128)
        self.w7 = cross_fuse_module(64)
        self.w8 = cross_fuse_module(64)
        self.w9 = cross_fuse_module(64)

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
        #print(x.shape)
        x = self.w5(s_fea[4],x)
        #print(x.shape)
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

class cross_fuse_module(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.c_conv = nn.Conv2d(in_channels*2,2,kernel_size=1, padding=0)
        
       
    def forward(self,s_x,q_x):
        c_x = torch.cat([s_x,q_x],dim=1)
        ac = F.softmax(self.c_conv(c_x),dim=1)
        x_out = s_x * ac[:,0:1] + q_x * ac[:,1:2]
        self.attention = ac[:,0:1]
        return x_out
    

class fss_fea_lnlc(nn.Module):
    def __init__(self,pretrain=True):
        super().__init__()

        self.conditioner = uni_conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_lnlc(pretrain)

    def forward(self, input1, input2):
        s_feature = self.conditioner(input1)
        segment,q_feature,qw_feature = self.segmentor(input2, s_feature)
        return segment,s_feature,q_feature,qw_feature

class segmentor_fea_lnlc(nn.Module):

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
        #in_channels = [3,64,64,128,256,128,64,64,64]

        # self.w1 = SE_AC(3)
        self.w1 = SE_AC(3)
        self.w2 = lnlc_module2_ls(64)
        self.w3 = lnlc_module2_ls(64)
        self.w4 = SE_AC(128)
        self.w5 = SE_AC(256)
        self.w6 = SE_AC(128)
        self.w7 = SE_AC(64)
        self.w8 = SE_AC(64)
        self.w9 = SE_AC(64)
        

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

class lnlc_module1_s(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels*2
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels #// 2
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

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        QH = QW = 4
        assert(s_x.shape ==q_x.shape)
        x = torch.cat([s_x,q_x],dim=1)
        N,C,H,W = x.size()
        PH, PW = H//QH, W//QW
        x = x.reshape(N,C,QH,PH,QW,PW)
        x = x.permute(0,2,4,1,3,5)
        x = x.reshape(N*QH*QW,C,PH,PW)

        batch_size = N*QH*QW
    
        g_x = self.g(x).reshape(batch_size,self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).reshape(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        
        f_div_C = F.softmax(f, dim=-1) 
        self.matrix = f_div_C.detach()

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, PH,PW)
        W_y = self.W(y)
        if self.shortcut:
            z = W_y + x
        else:
            z = W_y

        z = z.reshape(N,QH,QW,C,PH,PW)
        z = z.permute(0,3,1,4,2,5).reshape(N,C,H,W)

        self.attention = torch.sigmoid(self.conv(z))
        out = q_x * self.attention

        return out

class lnlc_module1_ls(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        self.in_channels = in_channels*2
        self.nlc_long = nlc_basic_module(in_channels*2)
        self.nlc_short = nlc_basic_module(in_channels*2)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        QH = QW = 4
        assert(s_x.shape ==q_x.shape)
        x = torch.cat([s_x,q_x],dim=1)
        N,C,H,W = x.size()
        PH, PW = H//QH, W//QW
        x = x.reshape(N,C,QH,PH,QW,PW)
        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(N*PH*PW,C,QH,QW)

        x = self.nlc_long(x)
        x = x.reshape(N,PH,PW,C,QH,QW)

        x = x.permute(0,4,5,3,1,2)
        x = x.reshape(N*QH*QW,C,PH,PW)
        x = self.nlc_short(x)
        x = x.reshape(N,QH,QW,C,PH,PW)
        x = x.permute(0,3,1,4,2,5).reshape(N,C,H,W)

        self.attention = torch.sigmoid(self.conv(x))
        out = q_x * self.attention

        return out

class lnlc_module1_sl(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        self.in_channels = in_channels*2
        self.nlc_long = nlc_basic_module(in_channels*2)
        self.nlc_short = nlc_basic_module(in_channels*2)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        QH = QW = 4
        assert(s_x.shape ==q_x.shape)
        x = torch.cat([s_x,q_x],dim=1)
        N,C,H,W = x.size()
        PH, PW = H//QH, W//QW
        x = x.reshape(N,C,QH,PH,QW,PW)

        x = x.permute(0,2,4,1,3,5)
        x = x.reshape(N*QH*QW,C,PH,PW)

        x = self.nlc_short(x)
        x = x.reshape(N,QH,QW,C,PH,PW)

        x = x.permute(0,4,5,3,1,2)
        x = x.reshape(N*PH*PW,C,QH,QW)
        x = self.nlc_long(x)
        x = x.reshape(N,PH,PW,C,QH,QW)

        x = x.permute(0,3,4,1,5,2).reshape(N,C,H,W)

        self.attention = torch.sigmoid(self.conv(x))
        out = q_x * self.attention

        return out

class nlc_basic_module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels //2
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

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = x.shape[0]
    
        g_x = self.g(x).reshape(batch_size,self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).reshape(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        
        f_div_C = F.softmax(f, dim=-1) 
        self.matrix = f_div_C.detach()

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        if self.shortcut:
            z = W_y + x
        else:
            z = W_y
     
        return z

class nlc_fuse_module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels //2
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

        if sub_sample:
            self.g = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
            self.phi = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0), nn.MaxPool2d(kernel_size=(down_rate, down_rate)))
        else:
            self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)
            
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        batch_size = s_x.shape[0]
    
        g_x = self.g(s_x).reshape(batch_size,self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(q_x).reshape(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(s_x).reshape(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        
        f_div_C = F.softmax(f, dim=-1) 
        self.matrix = f_div_C.detach()

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, *s_x.size()[2:])
        W_y = self.W(y)
        if self.shortcut:
            z = W_y + q_x
        else:
            z = W_y

        return z



class lnlc_module2_ls(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        self.in_channels = in_channels*2
        self.nlc_long = nlc_basic_module(in_channels*2)
        self.nlc_short = nlc_basic_module(in_channels*2)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels,
                        kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        QH = QW = 4
        assert(s_x.shape ==q_x.shape)
        x = torch.cat([s_x,q_x],dim=1)
        N,C,H,W = x.size()
        PH, PW = H//QH, W//QW
        x = x.reshape(N,C,QH,PH,QW,PW)
        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(N*PH*PW,C,QH,QW)

        x = self.nlc_long(x)
        x = x.reshape(N,PH,PW,C,QH,QW)

        x = x.permute(0,4,5,3,1,2)
        x = x.reshape(N*QH*QW,C,PH,PW)
        x = self.nlc_short(x)
        x = x.reshape(N,QH,QW,C,PH,PW)
        x = x.permute(0,3,1,4,2,5).reshape(N,C,H,W)

        self.nlc_res = self.conv(x)
        out = q_x + self.nlc_res
   

        # self.attention = torch.sigmoid(self.conv(x))
        # out = q_x * self.attention

        return out

class lnlc_module3_ls(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, down_rate =2,
        bn_layer=True,shortcut =True, in_spatial=None):
        super().__init__()
        self.in_channels = in_channels
        self.nlc_long = nlc_fuse_module(in_channels)
        self.nlc_short = nlc_fuse_module(in_channels)
        # self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=in_channels,
        #                 kernel_size=1, stride=1, padding=0)
     

    def forward(self, s_x,q_x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        QH = QW = 4
        assert(s_x.shape ==q_x.shape)
        x = torch.cat([s_x,q_x],dim=1)
        N,C,H,W = x.size()
        PH, PW = H//QH, W//QW
        x = x.reshape(N,C,QH,PH,QW,PW)
        x = x.permute(0,3,5,1,2,4)
        x = x.reshape(N*PH*PW,C,QH,QW)
        s_x, q_x = x[:,:C//2], x[:,C//2:]
        q_x = self.nlc_long(s_x,q_x)

        x = torch.cat([s_x,q_x],dim=1)
        x = x.reshape(N,PH,PW,C,QH,QW)

        x = x.permute(0,4,5,3,1,2)
        x = x.reshape(N*QH*QW,C,PH,PW)
        s_x, q_x = x[:,:C//2], x[:,C//2:]
        q_x = self.nlc_short(s_x,q_x)

        q_x = q_x.reshape(N,QH,QW,C//2,PH,PW)
        q_x = q_x.permute(0,3,1,4,2,5).reshape(N,C//2,H,W)

        #out = q_x + self.conv(x)
        
        # self.attention = torch.sigmoid(self.conv(x))
        # out = q_x * self.attention
        out = q_x

        return out

class pretext_segmentor(nn.Module):

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

    def forward(self, x):
        x = self.ec_block(x)
        x = F.relu(self.rn(x))
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.bn(self.conv(x)))

        x = self.bn_out(self.conv_out(x))

        x_out = torch.sigmoid(x)

        return x_out

    def close(self):
        for sf in self.sfs: sf.remove() 