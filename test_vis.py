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

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *
import cv2
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu",'--g', type=str,default=None)
    parser.add_argument("--dataset", type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",type=str,default='liver',help='liver,right kidney,left kidney,spleen')
    return parser.parse_args()

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

train_iteration = 5
train_bs = 1

model_path = \
    './result_paper3/MRI/liver/t5_cons_pretrain_nlc_layer[5, 6]_run1/0-fold_best.pth.tar'
k_fold = 0

net = my_fss_fea_nlc(pretrain=True,nlc_layer=[5,6],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False).cuda()

checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['state_dict'])

vis_save_path =  './test_vis/nlc_matrix2/'
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
# train_sampler = OneShotBatchSampler_ms4(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan], 
#     batch_size=train_bs, iteration=train_iteration)
train_sampler = OneShotBatchSampler(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan],
    batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)


criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

with torch.no_grad():
    net.eval
    for i_batch, sampled_batch in enumerate(train_loader):
        path = vis_save_path + 'iter-{}/'.format(i_batch)
        if not os.path.exists(path):
            os.makedirs(path)
        raw_path = path + 'raw_map/'
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        norm_path = path + 'norm_map/'
        if not os.path.exists(norm_path):
            os.makedirs(norm_path)
        # path1 = path + 'layer5/'
        # if not os.path.exists(path1):
        #     os.makedirs(path1)
        # path2 = path + 'layer6/'
        # if not os.path.exists(path2):
        #     os.makedirs(path2)

        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        _query_label = train_loader.batch_sampler.query_label
    
        #support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
        support_image, query_image, support_label, query_label = split_batch(image, label, _query_label)
    
        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)

        nlc_matrix1 = net.segmentor.weight[4].matrix.cpu().numpy().squeeze() #256,256//4 ==>256*16*16
        nlc_matrix2 = net.segmentor.weight[5].matrix.cpu().numpy().squeeze() #1024,1024//4 ==>1024*32*32
        # print(np.sum(nlc_matrix1,-1).shape,np.unique(np.sum(nlc_matrix1,-1)))
        # print(np.sum(nlc_matrix2,-1).shape,np.unique(np.sum(nlc_matrix2,-1)))

        #print(nlc_matrix1.shape,np.unique(nlc_matrix1))
        #print(nlc_matrix2.shape,np.unique(nlc_matrix2))

        #nlc_matrix1 = cv2.resize(nlc_matrix1,(256,256))
        # nlc_matrix2 = cv2.resize(nlc_matrix2,(1024,1024))
        nlc_matrix1 = nlc_matrix1.reshape((256,8,8))
        for i in tqdm(range(nlc_matrix1.shape[0])):
            matrix = nlc_matrix1[i]
            matrix = cv2.resize(matrix,(256,256))
            matrix = (matrix*255).astype(np.uint8)
            heatmap = cv2.applyColorMap(matrix, cv2.COLORMAP_JET)
            cv2.imwrite(raw_path+'loc_{},{}.png'.format(i//16,i%16),heatmap)
            matrix = (nlc_matrix1[i]-nlc_matrix1[i].min())/(nlc_matrix1[i].max()-nlc_matrix1[i].min())
            matrix = cv2.resize(matrix,(256,256))
            matrix = (matrix*255).astype(np.uint8)
            heatmap = cv2.applyColorMap(matrix, cv2.COLORMAP_JET)
            cv2.imwrite(norm_path+'loc_{},{}.png'.format(i//16,i%16),heatmap)
        
    

        # nlc_matrix1 = (nlc_matrix1-nlc_matrix1.min())/(nlc_matrix1.max()-nlc_matrix1.min())
        # nlc_matrix2 = (nlc_matrix2-nlc_matrix2.min())/(nlc_matrix2.max()-nlc_matrix2.min())

        # nlc_matrix1 = (nlc_matrix1*255).astype(np.uint8)
        # nlc_matrix2 = (nlc_matrix2*255).astype(np.uint8)

        # heatmap1 = cv2.applyColorMap(nlc_matrix1, cv2.COLORMAP_JET)
        # heatmap2 = cv2.applyColorMap(nlc_matrix2, cv2.COLORMAP_JET)
    
        # print(np.unique(nlc_matrix1))
        # print(np.unique(nlc_matrix2))
        support_con = torch.cat([support_image,support_image,support_image],dim=1)
        support_con[:,2] += support_label
        support_con = support_con.permute(0,2,3,1)

        query_con = torch.cat([query_image,query_image,query_image],dim=1)
        query_con[:,2] += query_label
        query_con = query_con.permute(0,2,3,1)

        # cv2.imwrite(vis_save_path+'layer5_i{}.png'.format(i_batch),heatmap1)
        # cv2.imwrite(vis_save_path+'layer6_i{}.png'.format(i_batch),heatmap2) 
        cv2.imwrite(path+'support.png',support_con.cpu().numpy().squeeze()*255)
        cv2.imwrite(path+'query.png',query_con.cpu().numpy().squeeze()*255)

        # print(net.segmentor.weight[4].matrix.cpu().numpy().shape)
        # print(net.segmentor.weight[5].matrix.cpu().numpy().shape)

        #seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)


