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
from Networks.Network_correct import *

import torch.optim as optim
import argparse

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int,default=3)
    parser.add_argument("--step", type=int,default=1)
    return parser.parse_args()
args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

with torch.no_grad():
    dis_eval = False
    save_img = True
    save_gt = False
    
    step = args.step

    model_name = 'Rakelly'

    # net = fss_fea_lnlc(pretrain=True).cuda()
    # if model_name == 'MIA':
    #     import MIA.few_shot_segmentor as fs
    # elif model_name == 'Shaban':
    #     import MIA.other_experiments.shaban.few_shot_segmentor_shaban_baseline as fs
    # elif model_name == 'Rakelly':
    #     import MIA.other_experiments.rakelly.few_shot_segmentor_feature_fusion_baseline as fs
    # net_params = {'num_class':1,'num_channels': 1,'num_filters': 64,'kernel_h': 5,'kernel_w': 5,'kernel_c': 1,'stride_conv': 1,'pool': 2,'stride_pool': 2,'se_block': "NONE" #Valid options : NONE, CSE, SSE, CSSE
    # ,'drop_out': 0}
    # net = fs.FewShotSegmentorDoubleSDnet(net_params).cuda()

    best_run = [1,3,2,2,1,1,1,2][step-1]

    data_index = [1,1,1,1,2,2,2,2][step-1]
    organ_index = [1,2,3,4,1,2,3,4][step-1]
    
Dataset = ['MRI','CT'][data_index-1]
Organ = ['liver','spleen','left_kidney','right_kidney'][organ_index-1]
model_path = './result_paper/{}/{}/{}/'.format(Dataset,Organ,model_name)
# model_path = './result_paper/{}/{}/{}_run{}/'.format(Dataset,Organ,model_name,best_run)

Num_support = 8
if Dataset == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
elif Dataset == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
# root_save_path = './nii_save/'+model_path.split('/')[-2][:-5]+'/'+'{}_{}'.format(Dataset,Organ)+'/'
root_save_path = './nii_save/'+model_path.split('/')[-2]+'/'+'{}_{}'.format(Dataset,Organ)+'/'
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)

print(model_path)
data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']

organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                    "right_kidney": {"train": [1, 3, 4], "val": 2},
                    "left_kidney": {"train": [1, 2, 4], "val": 3},
                    "spleen": {"train": [1, 2, 3], "val": 4} 
                    }
case_start_index = list(case_start_index)
len_kfold = len(case_start_index)//5
case_start_index.append(whole_image.shape[0])

total_dc, total_dc1 = [], []
with torch.no_grad():
    for k_fold in range(5):
        # net = fss_fea(pretrain=True).cuda()
        try:
            checkpoint = torch.load(model_path + '{}-fold_best.pth'.format(k_fold))
        except:
            checkpoint = torch.load(model_path + '{}-fold_best.pth.tar'.format(k_fold))
        net.load_state_dict(checkpoint['state_dict'])
        print('orgin eval,{}_fold,best epoch:{},beset DC:{:.3f}'.format(k_fold,checkpoint['epoch'],checkpoint['best_DC']))
        if not dis_eval:
            save_path = root_save_path 
        else:
            save_path = root_save_path +'origin_eval/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        support_item = len_kfold*k_fold
        query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
        assert len(query_item)+1 == len_kfold
        dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,
            query_item,save_path,query_label=organ_label_dict[Organ]['val'],
            Num_support=Num_support,test_bs=30,save_img=save_img,save_gt=save_gt)
        val_dc = sum(dice_list)/len(dice_list)
        total_dc.append(val_dc)

        if dis_eval:
            try:
                checkpoint = torch.load(model_path + '{}-fold_best1.pth')
            except:
                checkpoint = torch.load(model_path + '{}-fold_best1.pth.tar')
            net.load_state_dict(checkpoint['state_dict'])
            print('dis eval,{}_fold,best epoch:{},beset DC:{:.3f}'.format(k_fold,checkpoint['epoch'],checkpoint['best_DC']))
            save_path = root_save_path + 'dis_eval/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,
                query_item,save_path,query_label=organ_label_dict[Organ]['val'],
                Num_support=Num_support,test_bs=30,save_img=save_img,save_gt=save_gt,Norm='Slice')
    
            val_dc1 = sum(dice_list1)/len(dice_list1)
            total_dc1.append(val_dc1)

if dis_eval:
    print('{}: dc={:.3f},dc1={:.3f}'.format(model_path,sum(total_dc)/len(total_dc),sum(total_dc1)/len(total_dc1)))
else:
    print('{}: dc={:.3f}'.format(model_path,sum(total_dc)/len(total_dc)))

