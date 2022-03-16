# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py
import torch
import torch.utils.data as data
import torch.nn as nn
from utils.Sampler import *
from utils.Segmentor import SegMenTor
from Networks.Network import *
import torch.optim as optim
import argparse
from Networks.Network_correct import *
import os
from evaluator import *

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default=None)
    parser.add_argument("--dataset", type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",type=str,default='left_kidney',help='liver,right kidney,left kidney,spleen')
    parser.add_argument("--run_order",type=str,default=None)
    parser.add_argument("--pretrain",action='store_true',default=True)
    parser.add_argument("--t",type=float,default=1.0)

    return parser.parse_args()


def distance(f1,f2):
    bs,c,h,w = f1.size()
    f1 = f1.permute(0,2,3,1).reshape(bs*h*w,c)
    f2 = f2.permute(0,2,3,1).reshape(bs*h*w,c)
    # f1 = f1.view(bs,-1)
    # f2 = f2.view(bs,-1)
    #distance = torch.abs(f1-f2).mean(dim=-1)/(bs*h*w)
    distance = F.pairwise_distance(f1,f2,p=2).reshape(bs,h,w).mean(dim=[-2,-1])
    return distance

args = get_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("choose gpu %s"%(args.gpu))

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

# pretext_path = './result_correct5'
pretext_path = './test'

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
    train_bs = 2  
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 15
    Num_support = 8
    # lambda_t = 50
    lambda_t = args.t
    model_path = pretext_path + '/MRI/'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 2
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 25
    Num_support = 8 #原始测量
    # lambda_t = 25
    lambda_t = args.t
    model_path = pretext_path + '/CT/'

model_path = model_path + ObjectOrgan + '/ms_cons1_qw2s_layer9_DIM_t1'

MEAN = 'None'  #DIM,SPA,NONE
margin = 0
cons_layer = -1
W_cons = True

if args.pretrain:
    model_path = model_path + '_pretrain'

if args.run_order:
    model_path += '_%s'%args.run_order

model_path += '/'

print_freq = 50
print(model_path)
txt_path = model_path + 'result.txt'
if not os.path.exists(model_path):
    os.makedirs(model_path)

root_save_path = model_path + 'nii_save/'
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)
 
f = open(txt_path, "a+")
f.write('train_bs:{},iter:{}|{}, num_epoch:{}, num_support:{} \n'.format(train_bs,train_iteration,val_iteration,num_epoch,Num_support))
f.close()

data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']

kfold_best_val_dc, kfold_best_val_dc1, kfold_best_val_dc2 = [], [], []
kfold_best_e, kfold_best_e1, kfold_best_e2 = [], [], []

print('DataSet: {}, ObjectOrgan: {}'.format(DataSet,ObjectOrgan))
f = open(txt_path, "a+")
f.write('DataSet: {}, ObjectOrgan: {} \n'.format(DataSet,ObjectOrgan))
f.close()

case_start_index = list(case_start_index)
len_kfold = len(case_start_index)//5
case_start_index.append(whole_image.shape[0])

bs = train_bs

for k_fold in range(1):
    net = fss_fea(args.pretrain).cuda()

    print('k_fold:{}'.format(k_fold))
    support_item = len_kfold*k_fold
    query_item = list(range(len_kfold*k_fold+1,len_kfold*(k_fold+1)))

    train_start_index = case_start_index[len_kfold*k_fold]
    train_end_index = case_start_index[len_kfold*(k_fold+1)] 
    train_image = np.concatenate([ whole_image[:train_start_index],whole_image[train_end_index:] ],axis=0)
    train_label = np.concatenate([ whole_label[:train_start_index],whole_label[train_end_index:] ],axis=0)

    train_dataset = SimpleData(train_image,train_label)
    train_sampler = OneShotBatchSampler_ms4(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan], batch_size=train_bs, iteration=train_iteration)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

    optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

    criterion1 = DiceLoss2D()
    criterion2 = nn.BCELoss()

    best_val_dc, best_val_dc1, best_val_dc2 = -1,-1,-1
    best_e, best_e1, best_e2 = 0, 0 ,0

    for e in range(num_epoch+1):
        net.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            label = sampled_batch[1].type(torch.FloatTensor).cuda()
            _query_label = train_loader.batch_sampler.query_label
            support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
            condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
            seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)

            if W_cons:
                qf_layer = qw_feature[cons_layer]
            else:
                qf_layer = q_feature[cons_layer]
            sf_layer = s_feature[cons_layer]
            
            if MEAN == 'DIM':
                sf = torch.mean(sf_layer,dim=1)
                qf = torch.mean(qf_layer,dim=1)
            elif MEAN == 'SPA':
                sf = torch.mean(sf_layer,dim=[2,3])
                qf = torch.mean(qf_layer,dim=[2,3])
            elif MEAN == 'None':
                sf, qf = sf_layer, qf_layer
            
            assert seg.max()<=1 and seg.min()>=0
            seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)

            if len(_query_label)==1:
                loss = seg_loss
            if len(_query_label)==2:
                for i in range(2):
                    anchor,pos,neg = torch.split(qf,bs)[i], torch.split(sf,bs)[i],torch.split(sf,bs)[1-i]
                    dist_p, dist_n = distance(anchor,pos), distance(anchor,neg)
                    dist_diff = dist_p - dist_n + margin
                    if i==0:
                        tri_loss = torch.log(1+torch.exp(dist_diff)).mean()
                    else:
                        tri_loss += torch.log(1+torch.exp(dist_diff)).mean()
                loss = seg_loss + lambda_t*tri_loss
            if len(_query_label)==3:
                for i in range(3):
                    out_list = [0,1,2]
                    out_list.remove(i)
                    anchor,pos,neg1,neg2 = torch.split(qf,bs)[i],torch.split(sf,bs)[i],torch.split(sf,bs)[out_list[0]],\
                        torch.split(sf,bs)[out_list[1]]
                    dist_p, dist_n1, dist_n2 = distance(anchor,pos), distance(anchor,neg1), distance(anchor, neg2)
                    dist_diff = dist_p - dist_n1*0.5 - dist_n2*0.5 + margin
                    if i==0:
                        tri_loss = torch.log(1+torch.exp(dist_diff)).mean()
                    else:
                        tri_loss += torch.log(1+torch.exp(dist_diff)).mean()
                loss = seg_loss + lambda_t*tri_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i_batch % print_freq==0:
            #     if len(_query_label)==1:
            #         if SC_DIS == 0:
            #             print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}'.format(e, i_batch, len(train_loader), \
            #             loss,seg_loss))
            #         else:
            #             print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}+{:.3f}*{:.3f}'.format(e, i_batch, len(train_loader),\
            #             loss,seg_loss,lambda_t,tri_loss1))
            #     if len(_query_label)>=2:
            #         print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}+{:.3f}x{:.3f}'.format(e, i_batch, \
            #         len(train_loader), loss,seg_loss,lambda_t,tri_loss))
            if i_batch%5 == 0:
                # if len(_query_label)==1:
                #     print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}'.format(e, i_batch, len(train_loader), 
                #         loss,seg_loss))
                if len(_query_label)==2:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}+{:.3f}x{:.3f}'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss))
                    print('N=2 dist_p {:.3f}, dist_n {:.3f}, dist_diff {:.3f}, tri_loss {:.3f}'.format(dist_p[0],
                        dist_n[0], dist_diff[0], torch.log(1+torch.exp(dist_diff[0]))))
                elif len(_query_label)==3:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}+{:.3f}x{:.3f}'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss))
                    print('N=3 dist_p {:.3f}, dist_n1 {:.3f}, dist_n2 {:.3f}, dist_diff {:.3f}, tri_loss {:.3f}'.format( \
                        dist_p[0], dist_n1[0], dist_n2[0], dist_diff[0], torch.log(1+torch.exp(dist_diff[0]))))
    
        with torch.no_grad():
            # save_path = root_save_path +'epoch-{}/'.format(e) 
            save_path = root_save_path 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            support_item = len_kfold*k_fold
            query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
            assert len(query_item)+1 == len_kfold

            dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=10)
            val_dc = sum(dice_list)/len(dice_list)

            val_dc1 = val_dc

            # dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
            #     query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=10)
            # val_dc1 = sum(dice_list1)/len(dice_list1)

            # val_dc2 = val_dc

            # dice_list2 = evaluate_fss_kfold_encoder1b(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
            #     query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16)
            # val_dc2 = sum(dice_list2)/len(dice_list2)
            
            # print('Epoch {:d}, avg dice: {:.1f}, avg dice1: {:.1f}, avg dice2: {:.1f}'.format(e,val_dc,val_dc1,val_dc2))
            print('#############Epoch {:d}, avg dice: {:.1f}, avg dice1: {:.1f}'.format(e,val_dc,val_dc1))


            if val_dc>best_val_dc:
                best_val_dc = val_dc
                best_e = e
                PATH = model_path + '{}-fold_best.pth.tar'.format(k_fold)
                torch.save({'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': best_e,
                    'best_DC': best_val_dc}, PATH)
            # if val_dc1>best_val_dc1:
            #     best_val_dc1 = val_dc1
            #     best_e1 = e 
            #     PATH = model_path + '{}-fold_best1.pth.tar'.format(k_fold)
            #     torch.save({'state_dict': net.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'epoch': best_e1,
            #         'best_DC': best_val_dc1}, PATH)
            # if val_dc2>best_val_dc2:
            #     best_val_dc2 = val_dc2
            #     best_e2 = e
                # PATH = model_path + '{}-fold_best2.pth'.format(k_fold)
                # torch.save({'state_dict': net.state_dict(),
                #             'optimizer': optimizer.state_dict(),
                #             'epoch': best_e2,
                #             'best_DC': best_val_dc2}, PATH)

    kfold_best_val_dc.append(best_val_dc)
    kfold_best_e.append(best_e)

    kfold_best_val_dc1.append(best_val_dc1)
    kfold_best_e1.append(best_e1)

    # kfold_best_val_dc2.append(best_val_dc2)
    # kfold_best_e2.append(best_e2)

                
    print('{}-fold, Best Epoch {:d} Avg Val dice: {:.1f}'.format(k_fold,best_e,best_val_dc))
    f = open(txt_path, "a+")
    f.write('{}-fold, Best Epoch {:d} Avg Val dice: {:.1f} \n'.format(k_fold,best_e,best_val_dc))
    f.close()
    print('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f}'.format(k_fold,best_e1,best_val_dc1))
    f = open(txt_path, "a+")
    f.write('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f} \n'.format(k_fold,best_e1,best_val_dc1))
    f.close()
    # print('{}-fold, Best Epoch {:d} Avg Val dice2: {:.1f}'.format(k_fold,best_e2,best_val_dc2))
    # f = open(txt_path, "a+")
    # f.write('{}-fold, Best Epoch {:d} Avg Val dice2: {:.1f} \n'.format(k_fold,best_e2,best_val_dc2))
    # f.close()

print(model_path)
# print('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f}, Avg Best Val dice2: {:.1f}'.format(DataSet,
#     ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),sum(kfold_best_val_dc1)/len(kfold_best_val_dc1),
#     (sum(kfold_best_val_dc2)/len(kfold_best_val_dc2))))
print('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f}'.format(DataSet,
    ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),
    sum(kfold_best_val_dc1)/len(kfold_best_val_dc1)))
print(kfold_best_e,kfold_best_val_dc)
print(kfold_best_e1,kfold_best_val_dc1)
# print(kfold_best_e2,kfold_best_val_dc2)
f = open(txt_path, "a+")
# f.write('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f}, Avg Best Val dice2: {:.1f}  \n'.format(DataSet,
#     ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),sum(kfold_best_val_dc1)/len(kfold_best_val_dc1),
#     sum(kfold_best_val_dc2)/len(kfold_best_val_dc2)))
f.write('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f}  \n'.format(DataSet,
    ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),
    sum(kfold_best_val_dc1)/len(kfold_best_val_dc1)))
for f_in in range(len(kfold_best_e)):
    f.write(str(kfold_best_e[f_in]) + ' ')
f.write(', ')
for f_in in range(len(kfold_best_val_dc)):
    f.write(str(kfold_best_val_dc[f_in]) + ' ')
f.write('\n')
for f_in in range(len(kfold_best_e1)):
    f.write(str(kfold_best_e1[f_in]) + ' ')
f.write(', ')
for f_in in range(len(kfold_best_val_dc1)):
    f.write(str(kfold_best_val_dc1[f_in]) + ' ')
f.write('\n')
# for f_in in range(len(kfold_best_e2)):
#     f.write(str(kfold_best_e2[f_in]) + ' ')
# f.write(', ')
# for f_in in range(len(kfold_best_val_dc2)):
#     f.write(str(kfold_best_val_dc2[f_in]) + ' ')
# f.write('\n')
f.close()


