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
    # parser.add_argument("--step", type=int,default=1)
    return parser.parse_args()
args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
Num_support = 8
dis_eval = False
save_img = False
save_gt = False
repeat_run = False

save_path=None

Dataset_list = ['MRI','CT']
Organ_list = ['liver','spleen','left_kidney','right_kidney']
with torch.no_grad():
    # model_name = 'lnlc2_ls_layer23_pretrain'
    # net = fss_fea(pretrain=True).cuda()
    # net = fss_fea_lnlc(pretrain=True).cuda()
    model_name = 'MIA'
    # net = fss_fea_lnlc(pretrain=True).cuda()
    import MIA.few_shot_segmentor as fs
    net_params = {'num_class':1,'num_channels': 1,'num_filters': 64,'kernel_h': 5,'kernel_w': 5,'kernel_c': 1,'stride_conv': 1,'pool': 2,'stride_pool': 2,'se_block': "NONE" #Valid options : NONE, CSE, SSE, CSSE
    ,'drop_out': 0}
    net = fs.FewShotSegmentorDoubleSDnet(net_params).cuda()

for dataset in Dataset_list:
    if dataset == 'MRI':
        data_path = './datasets/MRI/MRIWholeData.h5'
        cm_dataset = 'CT'
    elif dataset == 'CT':
        data_path = './datasets/CT/CTWholeData.h5'
        cm_dataset = 'MRI'
    data = h5py.File(data_path, 'r')
    whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']
    case_start_index = list(case_start_index)
    len_kfold = len(case_start_index)//5
    case_start_index.append(whole_image.shape[0])

    total_dc_list_dataset, total_dc_list_dataset1 = [],[]
    for organ in Organ_list:
        total_dc_list, total_dc_list1 = [],[]
        for run in [1]:
            # model_path = './result_paper/{}/{}/{}_run{}/'.format(cm_dataset,organ,model_name,run)
            model_path = './result_paper/{}/{}/{}/'.format(cm_dataset,organ,model_name)
            #print(model_path)
            organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                                "right_kidney": {"train": [1, 3, 4], "val": 2},
                                "left_kidney": {"train": [1, 2, 4], "val": 3},
                                "spleen": {"train": [1, 2, 3], "val": 4} 
                                }

            total_dc, total_dc1 = [], []
            for k_fold in range(5):
                # net = fss_fea(pretrain=True).cuda()
                try:
                    checkpoint = torch.load(model_path + '{}-fold_best.pth'.format(k_fold))
                except:
                    checkpoint = torch.load(model_path + '{}-fold_best.pth.tar'.format(k_fold))
                net.load_state_dict(checkpoint['state_dict'])
                # print('orgin eval,{}_fold,best epoch:{},beset DC:{:.3f}'.format(k_fold,checkpoint['epoch'],checkpoint['best_DC']))
                support_item = len_kfold*k_fold
                query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
                assert len(query_item)+1 == len_kfold
                with torch.no_grad():
                    dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,
                        query_item,save_path,query_label=organ_label_dict[organ]['val'],
                        Num_support=Num_support,test_bs=30,save_img=save_img,save_gt=save_gt)
                val_dc = sum(dice_list)/len(dice_list)
                total_dc.append(val_dc)

                if dis_eval:
                    try:
                        checkpoint = torch.load(model_path + '{}-fold_best1.pth'.format(k_fold))
                    except:
                        checkpoint = torch.load(model_path + '{}-fold_best1.pth.tar'.format(k_fold))
                    net.load_state_dict(checkpoint['state_dict'])
                    # print('dis eval,{}_fold,best epoch:{},beset DC:{:.3f}'.format(k_fold,checkpoint['epoch'],checkpoint['best_DC']))
                    with torch.no_grad():
                        dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,
                            query_item,save_path,query_label=organ_label_dict[organ]['val'],
                            Num_support=Num_support,test_bs=8,save_img=save_img,save_gt=save_gt,Norm='Slice')
            
                    val_dc1 = sum(dice_list1)/len(dice_list1)
                    total_dc1.append(val_dc1)
                else:
                    total_dc1.append(-1)

            total_dc_list.append(sum(total_dc)/len(total_dc))
            total_dc_list1.append(sum(total_dc1)/len(total_dc1))

        # print('method is {}, dataset is {}, organ is {}. dice is {:.2f},{:.2f},{:.2f}, avg is {:.3f}. dice1 is {:.2f},{:.2f},{:.2f}, avg is {:.3f}'.format(model_name,dataset,organ,total_dc_list[0],total_dc_list[1],
        #             total_dc_list[2],sum(total_dc_list)/len(total_dc_list),total_dc_list1[0],total_dc_list1[1],total_dc_list1[2],sum(total_dc_list1)/len(total_dc_list1)))
        print('method is {}, dataset is {}, organ is {}. dice is {:.3f}. dice1 is {:.3f}'.format(model_name,dataset,organ,total_dc_list[0],total_dc_list1[0]))
                    
        total_dc_list_dataset.append(sum(total_dc_list)/len(total_dc_list))
        total_dc_list_dataset1.append(sum(total_dc_list1)/len(total_dc_list))

    print(model_name,dataset,sum(total_dc_list_dataset)/len(total_dc_list_dataset),sum(total_dc_list_dataset1)/len(total_dc_list_dataset1))
                    
