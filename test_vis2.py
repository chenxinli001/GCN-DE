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

def tensor_translate(tensor,t_x,t_y):
    assert len(tensor.shape)== 4 or len(tensor.shape)== 3
    new_tensor = torch.zeros_like(tensor)
    assert t_x>=0 and t_y>=0
    h, w = tensor.size()[-2:]
    if len(tensor.shape)== 4:
        new_tensor[:,:,t_x:h,t_y:w] = tensor[:,:,0:h-t_x,0:w-t_y]
    else:
        new_tensor[:,t_x:h,t_y:w] = tensor[:,0:h-t_x,0:w-t_y]

    return new_tensor

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

train_iteration = 1
train_bs = 1

model_path = \
    './result_paper3/MRI/liver/t0.1_cons_pretrain_run1/0-fold_best.pth.tar'
k_fold = 0

net = fss_vis(pretrain=True,nlc_layer=[],sub_sample=True, bn_layer=True,shortcut=True,
        cos_dis=False).cuda()

checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['state_dict'])

vis_save_path =  './test_vis2/'
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
    net.eval()
    for i_batch, sampled_batch in enumerate(train_loader):
        # path = vis_save_path + '平移S当做Q/'.format(i_batch)
        path = vis_save_path + '平移S/'.format(i_batch)
        if not os.path.exists(path):
            os.makedirs(path)
        fea_path = path + 'feature/'.format(i_batch)
        if not os.path.exists(fea_path):
            os.makedirs(fea_path)
        w_fea_path = path + 'w_feature/'.format(i_batch)
        if not os.path.exists(w_fea_path):
            os.makedirs(w_fea_path)
        

        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        _query_label = train_loader.batch_sampler.query_label
    
        #support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
        support_image, query_image, support_label, query_label = split_batch(image, label, _query_label)

        #query_image1 = tensor_translate(support_image,5,5)
        #query_image2 = tensor_translate(support_image,50,50)

        query_image = support_image 
        support_image1 = tensor_translate(support_image,5,5)
        support_image2 = tensor_translate(support_image,50,50)
        support_label1 = tensor_translate(support_label,5,5)
        support_label2 = tensor_translate(support_label,50,50)


        # condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        # seg,attention,q_feature,qw_feature = net(condition_input,query_image1)
        condition_input = torch.cat([support_image1,support_label1.unsqueeze(dim=1)],dim=1)
        seg,attention,q_feature,qw_feature = net(condition_input,query_image)
        att_map1 = np.zeros((9,256,256))
        #fea_map1 = np.zeros((9,256,256))
        #w_fea_map1 = np.zeros((9,256,256))
        for i in range(9):
            att = attention[i][0,0]
            att_map1[i] = cv2.resize(att.cpu().numpy(),(256,256))
            fea = q_feature[i][0].cpu().numpy()
            w_fea  = qw_feature[i][0].cpu().numpy()

            for j in range(3):
                fea_map = cv2.resize(fea[j],(256,256))
                #fea_heatmap = cv2.applyColorMap(fea_map, cv2.COLORMAP_JET)
                w_fea_map = cv2.resize(w_fea[j],(256,256))
                #w_fea_heatmap = cv2.applyColorMap(w_fea_map, cv2.COLORMAP_JET)
                cv2.imwrite(fea_path+'near_layer{}_{}.png'.format(i+1,j),norm(fea_map))
                cv2.imwrite(w_fea_path+'near_layer{}_{}.png'.format(i+1,j),norm(w_fea_map))

        # seg,attention,q_feature,qw_feature  = net(condition_input,query_image2)
        condition_input = torch.cat([support_image2,support_label2.unsqueeze(dim=1)],dim=1)
        seg,attention,q_feature,qw_feature = net(condition_input,query_image)
        att_map2 = np.zeros((9,256,256))
        #fea_map2 = np.zeros((9,256,256))
        #w_fea_map2 = np.zeros((9,256,256))
        for i in range(9):
            att = attention[i][0,0]
            att_map2[i] = cv2.resize(att.cpu().numpy(),(256,256))
            fea = q_feature[i][0].cpu().numpy()
            w_fea  = qw_feature[i][0].cpu().numpy()

            for j in range(3):
                fea_map = cv2.resize(fea[j],(256,256))
                #fea_heatmap = cv2.applyColorMap(fea_map, cv2.COLORMAP_JET)
                w_fea_map = cv2.resize(w_fea[j],(256,256))
                #w_fea_heatmap = cv2.applyColorMap(w_fea_map, cv2.COLORMAP_JET)
                cv2.imwrite(fea_path+'far_layer{}_{}.png'.format(i+1,j),norm(fea_map))
                cv2.imwrite(w_fea_path+'far_layer{}_{}.png'.format(i+1,j),norm(w_fea_map))


        att_map1 = (att_map1*255).astype(np.uint8)
        #fea_map1 = norm(fea_map1)
        #w_fea_map1 = norm(w_fea_map1)
        att_map2 = (att_map2*255).astype(np.uint8)
        #fea_map2 = norm(fea_map2)
        #w_fea_map2 = norm(w_fea_map2)

        for i in range(9):
            heatmap1 = cv2.applyColorMap(att_map1[i], cv2.COLORMAP_JET)
            heatmap2 = cv2.applyColorMap(att_map2[i], cv2.COLORMAP_JET)

            cv2.imwrite(path+'att_layer{}_near.png'.format(i+1),heatmap1)
            cv2.imwrite(path+'att_layer{}_far.png'.format(i+1),heatmap2)
            

        # support_con = torch.cat([support_image,support_image,support_image],dim=1)
        # support_con[:,2] += support_label
        # support_con = support_con.permute(0,2,3,1)
        # cv2.imwrite(path+'support.png',support_con.cpu().numpy().squeeze()*255)

        # query_con = torch.cat([query_image1,query_image1,query_image1],dim=1)
        # query_label1 = tensor_translate(support_label,5,5)
        # query_con[:,2] += query_label1
        # query_con = query_con.permute(0,2,3,1)
        # cv2.imwrite(path+'query_near.png',query_con.cpu().numpy().squeeze()*255)

        # query_con = torch.cat([query_image2,query_image2,query_image2],dim=1)
        # query_label2 = tensor_translate(support_label,50,50)
        # query_con[:,2] += query_label2
        # query_con = query_con.permute(0,2,3,1)
        # cv2.imwrite(path+'query_far.png',query_con.cpu().numpy().squeeze()*255)

        support_con1 = torch.cat([support_image1,support_image1,support_image1],dim=1)
        support_con1[:,2] += support_label1
        support_con1 = support_con1.permute(0,2,3,1)
        cv2.imwrite(path+'support_near.png',support_con1.cpu().numpy().squeeze()*255)

        support_con2 = torch.cat([support_image2,support_image2,support_image2],dim=1)
        support_con2[:,2] += support_label2
        support_con2 = support_con2.permute(0,2,3,1)
        cv2.imwrite(path+'support_far.png',support_con2.cpu().numpy().squeeze()*255)

        query_con = torch.cat([query_image,query_image,query_image],dim=1)
        query_con[:,2] += support_label
        query_con = query_con.permute(0,2,3,1)
        cv2.imwrite(path+'query.png',query_con.cpu().numpy().squeeze()*255)
        

        #nlc_matrix1 = net.segmentor.weight[4].matrix.cpu().numpy().squeeze() #256,256//4 ==>256*16*16
        #nlc_matrix2 = net.segmentor.weight[5].matrix.cpu().numpy().squeeze() #1024,1024//4 ==>1024*32*32
        # print(np.sum(nlc_matrix1,-1).shape,np.unique(np.sum(nlc_matrix1,-1)))
        # print(np.sum(nlc_matrix2,-1).shape,np.unique(np.sum(nlc_matrix2,-1)))

        #print(nlc_matrix1.shape,np.unique(nlc_matrix1))
        #print(nlc_matrix2.shape,np.unique(nlc_matrix2))

        #nlc_matrix1 = cv2.resize(nlc_matrix1,(256,256))
        # nlc_matrix2 = cv2.resize(nlc_matrix2,(1024,1024))
        #nlc_matrix1 = nlc_matrix1.reshape((256,8,8))
        # for i in tqdm(range(nlc_matrix1.shape[0])):
        #     matrix = nlc_matrix1[i]
        #     matrix = cv2.resize(matrix,(256,256))
        #     matrix = (matrix*255).astype(np.uint8)
        #     heatmap = cv2.applyColorMap(matrix, cv2.COLORMAP_JET)
        #     cv2.imwrite(raw_path+'loc_{},{}.png'.format(i//16,i%16),heatmap)
        #     matrix = (nlc_matrix1[i]-nlc_matrix1[i].min())/(nlc_matrix1[i].max()-nlc_matrix1[i].min())
        #     matrix = cv2.resize(matrix,(256,256))
        #     matrix = (matrix*255).astype(np.uint8)
        #     heatmap = cv2.applyColorMap(matrix, cv2.COLORMAP_JET)
        #     cv2.imwrite(norm_path+'loc_{},{}.png'.format(i//16,i%16),heatmap)
        
    

        # nlc_matrix1 = (nlc_matrix1-nlc_matrix1.min())/(nlc_matrix1.max()-nlc_matrix1.min())
        # nlc_matrix2 = (nlc_matrix2-nlc_matrix2.min())/(nlc_matrix2.max()-nlc_matrix2.min())

        # nlc_matrix1 = (nlc_matrix1*255).astype(np.uint8)
        # nlc_matrix2 = (nlc_matrix2*255).astype(np.uint8)

        # heatmap1 = cv2.applyColorMap(nlc_matrix1, cv2.COLORMAP_JET)
        # heatmap2 = cv2.applyColorMap(nlc_matrix2, cv2.COLORMAP_JET)
    
        # print(np.unique(nlc_matrix1))
        # print(np.unique(nlc_matrix2))
        

        # print(net.segmentor.weight[4].matrix.cpu().numpy().shape)
        # print(net.segmentor.weight[5].matrix.cpu().numpy().shape)

        #seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)


