# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn

from Networks.Sampler import *

from Networks.Segmentor import SegMenTor
from Networks.Network import *

import torch.optim as optim
import argparse
import numpy

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default=None)

    return parser.parse_args()


def l1_distance(f1,f2):
    bs = f1.shape[0]
    f1 = f1.view(bs,-1)
    f2 = f2.view(bs,-1)
    distance = torch.abs(f1-f2).mean()
    return distance

def dot_distance(f1,f2):
    bs = f1.shape[0]
    f1 = f1.view(bs,-1)
    f2 = f2.view(bs,-1)
    distance = 0.5-(f1*f2).sum()/(2*torch.norm(f1)*torch.norm(f2))
    return distance

def ln_exp_pos_distance(f1,f2):
    bs = f1.shape[0]
    f1 = f1.view(bs,-1)
    f2 = f2.view(bs,-1)
    distance = 0.5-(f1*f2).sum()/(2*torch.norm(f1)*torch.norm(f2))
    #distance = torch.abs(f1-f2).mean()
    #distance = torch.log(1+torch.exp(distance))
    distance = torch.exp(distance)
    return distance

def ln_exp_neg_distance(f1,f2):
    bs = f1.shape[0]
    f1 = f1.view(bs,-1)
    f2 = f2.view(bs,-1)
    #distance = 0.5-(f1*f2).sum()/(2*torch.norm(f1)*torch.norm(f2))
    distance = torch.abs(f1-f2).mean()
    distance = torch.log(1+torch.exp(-distance))
    #distance = torch.exp(-distance)
    return distance

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
max_free = np.argmax(memory_gpu)

print("choose gpu %d free %d MiB"%(max_free, memory_gpu[max_free]))
os.environ['CUDA_VISIBLE_DEVICES']=str(max_free)

args = get_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lab_list_fold = {"fold1": {"train": [2, 3, 4], "val": 1},#liver
                 "fold2": {"train": [1, 3, 4], "val": 2},#right kidney
                 "fold3": {"train": [1, 2, 4], "val": 3},#left kidney
                 "fold4": {"train": [1, 2, 3], "val": 4} #spleen
                 }

organ_fold_dict = {'liver':'fold1','right kidney':'fold2','left kidney':'fold3','spleen':'fold4'}
organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                    "right kidney": {"train": [1, 3, 4], "val": 2},
                    "left kidney": {"train": [1, 2, 4], "val": 3},
                    "spleen": {"train": [1, 2, 3], "val": 4} 
                    }

DataSet = 'CT' # 'MRI'or'CT'
ObjectOrgan = 'liver'

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
    train_bs = 2  
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs 
    num_epoch = 15
    Num_support = 8 
    lambda_t = 50
    model_path = './result_pretrain/ce_SE-cons5/MRI/'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 2
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 25
    Num_support = 16 #原始测量
    lambda_t = 25
    model_path = './result_pretrain/ce_SE-cons5/CT/'


data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']

max_dis = 0
min_dis = 9e9

# print(DataSet,len(whole_image))
# for i in range(len(whole_image)):
#     for j in range(len(whole_image)):
#         if j != i:
#             dis = np.abs(whole_image[i] - whole_image[j]).mean()
#             if dis<min_dis:
#                 min_dis = dis
#             if dis>max_dis:
#                 max_dis = dis
# print(DataSet,min_dis,max_dis)
encoder = resnet_encoder().cuda()
feature = []
bs = 20
with torch.no_grad():
    for i in range(0,len(whole_image),bs):
        input = torch.FloatTensor(whole_image[i:i+bs]).cuda()
        feature.append(encoder(input.unsqueeze(1)))

feature = torch.cat(feature)

#print(len(whole_image),len(feature))
assert len(feature) == len(whole_image)

distance = []
for i in range(len(feature)):
    for j in range(len(feature)):
        if j != i:
            dis = torch.abs(feature[i] - feature[j]).mean()
            #dis = torch.cosine_similarity(feature[i],feature[j],dim=0).mean()
            distance.append(dis.cpu().numpy())
            if dis<min_dis:
                min_dis = dis
            if dis>max_dis:
                max_dis = dis
print(DataSet,min_dis,max_dis) 
plt.hist(distance)
