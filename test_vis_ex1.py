# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn

from Sampler import *

from Network import *
from Network2 import *

import torch.optim as optim
import argparse

#from nn_common_modules import losses as additional_losses

import os
from evaluator import *
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

train_iteration = 2
train_bs = 1

model_path = \
    './result_ex/MRI/liver/ex_pretrain_nlc1_[5]/0-fold_best.pth.tar'
k_fold = 0

vis_layer = [5]

net = fss_feature_ex1(pretrain=True,nlc_layer=[5]).cuda()

for name,param in net.named_parameters():
    print(name)

checkpoint = torch.load(model_path)

# for key in checkpoint['state_dict'].keys():
#     print(key)
    

net.load_state_dict(checkpoint['state_dict'])

vis_save_path =  './test_vis_ex'
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
        save_path = vis_save_path + '/figure-{}'.format(i_batch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        _query_label = train_loader.batch_sampler.query_label
    
        support_image, query_image, support_label, query_label = split_batch(image, label, _query_label)


        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)

        S_IMG = np.uint8(support_image.cpu().numpy().squeeze()* 255)
        S_IMG = cv2.cvtColor(S_IMG,cv2.COLOR_GRAY2BGR)

        Q_IMG = np.uint8(query_image.cpu().numpy().squeeze()* 255)
        Q_IMG = cv2.cvtColor(Q_IMG,cv2.COLOR_GRAY2BGR)

        _, _, h, w = query_image.shape

        matrix = []
        for layer in vis_layer:
            path = save_path + '/layer-{}'.format(layer)
            if not os.path.exists(path):
                os.makedirs(path)
            matrix_path = path + '/matrix'
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
            Wy_path = path + '/W_y'
            if not os.path.exists(Wy_path):
                os.makedirs(Wy_path)
            Fq_path = path + '/Q_feature'
            if not os.path.exists(Fq_path):
                os.makedirs(Fq_path)
    
            # nl_map = net.segmentor.weight[layer-1].matrix.cpu().numpy().squeeze() #256,256//4
            nl_map = net.segmentor.w5.matrix.cpu().numpy().squeeze() #256,256//4
            W_y = net.segmentor.w5.W_y.cpu().numpy().squeeze() #C,s,s
            q_x = net.segmentor.w5.q_x.cpu().numpy().squeeze() #C,s,s
            y_feature = norm(W_y)
            q_feature = norm(q_x)
            total_region, nl_map_length = nl_map.shape
            region_per_row = round(math.sqrt(total_region))
            size_of_region = round(w / region_per_row)
            region_per_row = round(math.sqrt(total_region))
            size_of_region = round(w / region_per_row)

            nl_map_size = round(math.sqrt(nl_map_length))

            for index in range(total_region):
                img_draw = Q_IMG.copy()
                nl_map_temp = nl_map[index]
                nl_map_temp = nl_map_temp.reshape(nl_map_size, nl_map_size)
                nl_map_temp = cv2.resize(nl_map_temp, dsize=(h, w))
        
                nl_map_temp = np.uint8(nl_map_temp * 255)
        
                heat_img = cv2.applyColorMap(nl_map_temp, cv2.COLORMAP_JET)
                heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
                img_add = cv2.addWeighted(img_draw, 0.3, heat_img, 0.7, 0)
        
                x0 = index // region_per_row * size_of_region
                x1 = x0 + size_of_region
        
                y0 = index % region_per_row * size_of_region
                y1 = y0 + size_of_region
        
                cv2.rectangle(img_add, (y0, x0), (y1, x1), (255, 0, 0), 1)
                cv2.imwrite(matrix_path+'/%d.png' % index, cv2.cvtColor(img_add, cv2.COLOR_BGR2RGB))
            
            for c in range(y_feature.shape[0]):
                y_temp = y_feature[c]
                q_temp = q_feature[c]
                y_temp = cv2.resize(y_temp, dsize=(h, w))
                q_temp = cv2.resize(q_temp, dsize=(h, w))
                cv2.imwrite(Wy_path+'/channel-{}.png'.format(c),y_temp)
                cv2.imwrite(Fq_path+'/channel-{}.png'.format(c),q_temp)
                

            S_add = S_IMG
            S_add[:,:,2] += np.uint8(support_label.cpu().numpy().squeeze()*255) 
            Q_add = Q_IMG
            Q_add[:,:,2] += np.uint8(query_label.cpu().numpy().squeeze()*255)

            # print(support_label.min(),support_label.max())
            # print(query_label.min(),query_label.max())

            # print(np.unique(np.uint8(support_label.cpu().numpy().squeeze()*255) ))
            # print(np.unique(np.uint8(query_label.cpu().numpy().squeeze()*255)))

            cv2.imwrite(path+'/S.png',S_add.astype(np.uint8))
            cv2.imwrite(path+'/Q.png',Q_add.astype(np.uint8))
            


        

    