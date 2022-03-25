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

from Networks.Network import *
from Networks.Network2 import *
from Networks.Network_correct import *

import torch.optim as optim
import argparse

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *
import cv2
from tqdm import tqdm
import math


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu",'--g', type=str,default=None)
    parser.add_argument("--dataset", type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",type=str,default='liver',help='liver,right kidney,left kidney,spleen')
    return parser.parse_args()


def norm(img):
    result = (img-img.min()/(img.max()-img.min()))*255
    return result.astype(np.uint8)


args = get_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lab_list_fold = {"fold1": {"train": [2, 3, 4], "val": 1},#liver
                 "fold2": {"train": [1, 3, 4], "val": 2},#right kidney
                 "fold3": {"train": [1, 2, 4], "val": 3},#left kidney
                 "fold4": {"train": [1, 2, 3], "val": 4} #spleen
                 }

organ_fold_dict = {'liver':'fold1','right_kidney':'fold2','left_kidney':'fold3','spleen':'fold4'}
organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                    "right_kidney": {"train": [1, 3, 4], "val": 2},
                    "left_kidney": {"train": [1, 2, 4], "val": 3},
                    "spleen": {"train": [1, 2, 3], "val": 4} 
                    }

DataSet = args.dataset # 'MRI'or'CT'
ObjectOrgan = args.organ

train_iteration = 0
train_bs = 1

#model_path = './result_correct/MRI/liver/base_SE1_pretrain_run1/0-fold_best.pth'
model_path = './result_correct2/MRI/liver/base_pretrain_w_nlc8_[5]_run1/0-fold_best.pth'
k_fold = 0

# vis_layer = [5, 6, 7]

#net = fss_fea(pretrain=True,SE_type=1).cuda()
net = fss_fea_nlc8(pretrain=True,SE_type=1,nlc_type=8,nlc_layer=[5]).cuda()

for name,param in net.named_parameters():
    print(name)

checkpoint = torch.load(model_path)

# for key in checkpoint['state_dict'].keys():
#     print(key)
    

net.load_state_dict(checkpoint['state_dict'])

vis_save_path =  './test_vis_ex_nlc/nlc8'
if not os.path.exists(vis_save_path):
    os.makedirs(vis_save_path)

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'

data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']


print('DataSet: {}, ObjectOrgan: {}'.format(DataSet,ObjectOrgan))

case_start_index = list(case_start_index)
len_kfold = len(case_start_index)//5
case_start_index.append(whole_image.shape[0])


print('k_fold:{}'.format(k_fold))
support_item = len_kfold*k_fold
query_item = list(range(len_kfold*k_fold+1,len_kfold*(k_fold+1)))
print(support_item,query_item)

train_start_index = case_start_index[len_kfold*k_fold]
train_end_index = case_start_index[len_kfold*(k_fold+1)] 
train_image = np.concatenate([ whole_image[:train_start_index],whole_image[train_end_index:] ],axis=0)
train_label = np.concatenate([ whole_label[:train_start_index],whole_label[train_end_index:] ],axis=0)

print(train_image.shape,train_label.shape)

train_dataset = SimpleData(train_image,train_label)
train_sampler = OneShotBatchSampler(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan],
    batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)


criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

with torch.no_grad():
    net.eval()
    for i_batch, sampled_batch in enumerate(train_loader):
        path = vis_save_path + '/figure-{}'.format(i_batch)
        if not os.path.exists(path):
            os.makedirs(path)

        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        _query_label = train_loader.batch_sampler.query_label
    
        support_image, query_image, support_label, query_label = split_batch(image, label, _query_label)


        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg,s_feature,q_feature,qw_feature,attention = net(condition_input,query_image)
        #seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)


        _, _, h, w = query_image.shape

        Fq_path = path + '/Q_feature'
        if not os.path.exists(Fq_path):
            os.makedirs(Fq_path)

        ac_map = attention.cpu().numpy().squeeze() #16,16
        # qf = q_feature[4].cpu().numpy().squeeze() #C,16,16
        # qwf = qw_feature[4].cpu().numpy().squeeze()

        #ac_map = ( ac_map- ac_map.min() ) / ( ac_map.max() - ac_map.min() )
        # print(ac_map.min(),ac_map.max())
        # print(norm_ac.min(),norm_ac.max())

        # qf = norm(qf)
        # qwf = norm(qwf)
        #print(q_feature)
        #print(qw_feature)
        for i in range(9):
            qf = q_feature[i]
            qf = qf.cpu().numpy().squeeze() #C,16,16
            print(type(q_feature[i]))
            print(type(qw_feature[i]))
        for layer in range(9):
            for c in range(3):
                qf = q_feature[layer]
                qf = qf.cpu().numpy().squeeze() #C,16,16
                qwf = qw_feature[layer]
                qwf = qwf.cpu().numpy().squeeze()
                q_temp = qf[c]
                qw_temp = qwf[c]
                q_temp = cv2.resize(q_temp, dsize=(h, w))
                qw_temp = cv2.resize(qw_temp, dsize=(h, w))
                q_feature = np.hstack((q_temp,qw_temp))
                cv2.imwrite(Fq_path+'/q_feature_layer-{}_channel-{}.png'.format(layer,c),q_feature)
            # cv2.imwrite(Fq_path+'/qf_channel-{}.png'.format(c),q_temp)
            # cv2.imwrite(Fwq_path+'/qwf_channel-{}.png'.format(c),qw_temp)

        # ac_map = cv2.resize(ac_map, dsize=(h, w))
        # ac_map = np.uint8(ac_map*255)

        # # print(ac_map.min(),ac_map.max())

        # heat_img = cv2.applyColorMap(ac_map, cv2.COLORMAP_JET)
   
        # cv2.imwrite(path+'/ac_map.png',heat_img)

        support_con= torch.cat([support_image,support_image,support_image],dim=1)
        support_con[:,2] += support_label
        support_con = support_con.permute(0,2,3,1)
        cv2.imwrite(path+'/S.png',support_con.cpu().numpy().squeeze()*255)

        query_con = torch.cat([query_image,query_image,query_image],dim=1)
        query_con[:,2] += query_label
        query_con = query_con.permute(0,2,3,1)
        cv2.imwrite(path+'/Q.png',query_con.cpu().numpy().squeeze()*255)
            
        # img_add_s = cv2.addWeighted(img_s, 0.3, heat_img, 0.7, 0)
        # img_add_q = cv2.addWeighted(img_q, 0.3, heat_img, 0.7, 0)

        # support_con= torch.cat([support_image,support_image,support_image],dim=1)
        # support_con = support_con.permute(0,2,3,1).cpu().numpy().squeeze() *255
        # s_ac = np.uint8((support_con*0.3+heat_img))

        # query_con = torch.cat([query_image,query_image,query_image],dim=1)
        # query_con = query_con.permute(0,2,3,1).cpu().numpy().squeeze() *255
        # q_ac = np.uint8((query_con*0.3+heat_img))

        # cv2.imwrite(path+'/S_ac.png' , s_ac)
        # cv2.imwrite(path+'/Q_ac.png' , q_ac)


        


    

    
