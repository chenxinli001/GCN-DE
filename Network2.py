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

import resnet

class fss_feature_ex1(nn.Module):
    def __init__(self,pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True):
        super().__init__()
        print(nlc_layer)

        self.conditioner = conditioner_fea(pretrain)
        self.segmentor = segmentor_fea_nlc1(pretrain,nlc_layer,sub_sample,bn_layer,shortcut)

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
 
class segmentor_fea_nlc1(nn.Module):

    def __init__(self,pretrain,nlc_layer,sub_sample, bn_layer,shortcut):
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
            self.w1 = SE_AC(3)
        else:
            self.w1 = nlc_module1(3)
        if 2 not in nlc_layer:
            self.w2 = SE_AC(64)
        else:
            self.w2 = nlc_module1(64)
        if 3 not in nlc_layer:
            self.w3 = SE_AC(64)
        else:
            self.w3 = nlc_module1(64)
        if 4 not in nlc_layer:
            self.w4 = SE_AC(128)
        else:
            self.w4 = nlc_module1(128)
        if 5 not in nlc_layer:
            self.w5 = SE_AC(256)
        else:
            self.w5 = nlc_module1(256)
        if 6 not in nlc_layer:
            self.w6 = SE_AC(128)
        else:
            self.w6 = nlc_module1(128)
        if 7 not in nlc_layer:
            self.w7 = SE_AC(64)
        else:
            self.w7 = nlc_module1(64)
        if 8 not in nlc_layer:
            self.w8 = SE_AC(64)
        else:
            self.w8 = nlc_module1(64)
        if 9 not in nlc_layer:
            self.w9 = SE_AC(64)
        else:
            self.w9 = nlc_module1(64)


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

class nlc_module1(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, 
        bn_layer=True,shortcut =True):
        super().__init__()

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
        W_y = self.W(y)
        if self.shortcut:
            z = W_y + q_x
        else:
            z = W_y
        self.W_y = W_y
        self.q_x = q_x

        return z




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
    def __init__(self,in_channels,out_channels=1,kernel_size=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding)
    
    def forward(self,s_x,q_x):
        self.attention = torch.sigmoid(self.conv(s_x))
        x_out = q_x * self.attention
        return x_out



