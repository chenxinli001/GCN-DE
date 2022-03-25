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

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
max_free = np.argmax(memory_gpu)

print("choose gpu %d free %d MiB"%(max_free, memory_gpu[max_free]))
os.environ['CUDA_VISIBLE_DEVICES']=str(max_free)

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


##### ms1 初步引入类别互斥信息, 同时对特征图取维度上的平均，去除冗余####
##### 对unet的中间特征计算各类之间的距离==>如果一张图只含一类,正常的分割损失；含两类及以上,增加一个S和Q的类内特征距离要小于类外特征距离的约束####
##### SC_DIS = 0, MEAN = 'DIM', distance = l1_distance ==>best####
##### 加finetune ####==>基本起到负作用

##### 加辅助分类，两种方案
####### 方案1：判断有多少类（1，2，3），方案2：判断有哪些器官（1，2，3）
####### 思路的出发点可以是引入辅助信息正则化编码器；还可以是为了在finetune时也可以使用距离正则化

##### 先做验证时精细化support slice选择的方案

train_image_path = './datasets/MRI/Train-Image.h5'
train_label_path = './datasets/MRI/Train-Label.h5'
support_file = './datasets/MRI/FSS-Eval-36.h5'
query_path = ['./datasets/MRI/FSS-Eval-37.h5','./datasets/MRI/FSS-Eval-38.h5','./datasets/MRI/FSS-Eval-39.h5']

train_bs = 2
val_bs = 8
train_iteration = 25#//train_bs
val_iteration = 25#//val_bs
num_epoch = 15

reload_mdoel = 0
print_freq = 25

support_num = 8
 
lambda_t = 50
model_path = './result_pretrain/SE_ms/'


# train_image_path = './datasets/CT/Train-Image.h5'
# train_label_path = './datasets/CT/Train-Label.h5'
# support_file = './datasets/CT/FSS-Eval-25.h5'
# query_path = ['./datasets/CT/FSS-Eval-26.h5','./datasets/CT/FSS-Eval-27.h5','./datasets/CT/FSS-Eval-28.h5',
#                 './datasets/CT/FSS-Eval-29.h5','./datasets/CT/FSS-Eval-30.h5']

# train_bs = 2
# val_bs = 8
# train_iteration = 25#//train_bs
# val_iteration = 25#//val_bs
# num_epoch = 30

# reload_mdoel = 0
# print_freq = 25
 
# lambda_t = 50
# model_path = './result_pretrain/SE_ms_CT/'


SC_DIS = 0
MEAN = 'DIM'  #DIM,SPA,NONE

distance = l1_distance

#model_path = './result/SE_ms/' 
#net = FewShotSegmentorDoubleSDnet().cuda()


net = my_fss_fea().cuda()

print(model_path)
txt_path = model_path + 'result.txt'
if not os.path.exists(model_path): 
    os.makedirs(model_path)

root_save_path = model_path + 'nii_save/'
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)

f = open(txt_path, "a+")
f.write('train_bs:{}|iter:{}, val_bs:{}|iter:{}, num_epoch:{} \n'.format(train_bs,train_iteration, val_bs,val_iteration,num_epoch))
f.close()

train_dataset = get_simple_dataset(train_image_path,train_label_path)
#val_dataset = get_simple_dataset(val_image_path,val_label_path)

train_sampler = OneShotBatchSampler_ms4(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

# val_sampler = OneShotBatchSampler(val_dataset.label, 'val', "fold1", batch_size=val_bs, iteration=val_iteration)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)

optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

best_val_dc = 0
best_e = 0

best_val_dc1n = 0
best_e1n = 0
best_val_dc1c = 0
best_e1c = 0
best_val_dc1s = 0
best_e1s = 0

best_val_dc2n = 0
best_e2n = 0 
best_val_dc2c = 0
best_e2c = 0
best_val_dc2s = 0
best_e2s = 0

if reload_mdoel:
    checkpoint = torch.load(model_path + 'latest.pth')

    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_DC = checkpoint['best_DC']
    print('Reload epoch {} model OK!, best DC is {}'.format(start_epoch, best_DC))
else:
    start_epoch = 0

for e in range(start_epoch,num_epoch+1):
    net.train()
    optimizer.zero_grad()
    for i_batch, sampled_batch in enumerate(train_loader):
        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()

        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        _query_label = train_loader.batch_sampler.query_label
        
        support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
        
        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)

        if MEAN == 'DIM':
            for list_index in range(len(s_feature)):
                s_feature[list_index] = torch.mean(s_feature[list_index],dim=1)
                q_feature[list_index] = torch.mean(q_feature[list_index],dim=1)
                qw_feature[list_index] = torch.mean(qw_feature[list_index],dim=1)
        elif MEAN == 'SPA':
            for list_index in range(len(s_feature)):
                s_feature[list_index] = torch.mean(s_feature[list_index],dim=[2,3])
                q_feature[list_index] = torch.mean(q_feature[list_index],dim=[2,3])
                qw_feature[list_index] = torch.mean(qw_feature[list_index],dim=[2,3])
        
        seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)

        if len(_query_label)==1:
            ### 单类不加距离损失 ###
            if SC_DIS==0:
                loss = seg_loss
            ### 单类加距离损失 ### 
            else:
                tri_loss1 = distance(s_feature[4][0:train_bs]-q_feature[4][0:train_bs])
                tri_loss1 =  torch.log(1+torch.exp(tri_loss1))
                loss = seg_loss + lambda_t * tri_loss1  
        if len(_query_label)==2:
            tri_loss1 = ( distance(s_feature[4][0:train_bs],q_feature[4][0:train_bs]) - 
                distance(s_feature[4][0:train_bs],q_feature[4][train_bs:train_bs*2]) )
            tri_loss2 = ( distance(s_feature[4][train_bs:train_bs*2],q_feature[4][train_bs:train_bs*2]) -
                distance(s_feature[4][train_bs:train_bs*2],q_feature[4][0:train_bs]) )
            tri_loss1 =  torch.log(1+torch.exp(tri_loss1))/2
            tri_loss2  = torch.log(1+torch.exp(tri_loss2))/2
            loss = seg_loss + lambda_t*(tri_loss1+tri_loss2)
        if len(_query_label)==3:
            tri_loss12 = ( distance(s_feature[4][0:train_bs],q_feature[4][0:train_bs]) - 
                distance(s_feature[4][0:train_bs],q_feature[4][train_bs:train_bs*2]) )
            tri_loss13 = ( distance(s_feature[4][0:train_bs],q_feature[4][0:train_bs]) - 
                distance(s_feature[4][0:train_bs],q_feature[4][train_bs*2:train_bs*3]) )

            tri_loss21 = ( distance(s_feature[4][train_bs:train_bs*2],q_feature[4][train_bs:train_bs*2]) - 
                distance(s_feature[4][train_bs:train_bs*2],q_feature[4][0:train_bs]) )
            tri_loss23 = ( distance(s_feature[4][train_bs:train_bs*2],q_feature[4][train_bs:train_bs*2])-
                distance(s_feature[4][train_bs:train_bs*2],q_feature[4][train_bs*2:train_bs*3]) )
            
            tri_loss31 = ( distance(s_feature[4][train_bs*2:train_bs*3],q_feature[4][train_bs*2:train_bs*3]) - 
                distance(s_feature[4][train_bs*2:train_bs*3],q_feature[4][0:train_bs]) )
            tri_loss32 = ( distance(s_feature[4][train_bs*2:train_bs*3],q_feature[4][train_bs*2:train_bs*3]) -
                distance(s_feature[4][train_bs*2:train_bs*3],q_feature[4][train_bs:train_bs*2]) )
            
            tri_loss1 =  (torch.log(1+torch.exp(tri_loss12)) + torch.log(1+torch.exp(tri_loss13)))/2/3 
            tri_loss2  = (torch.log(1+torch.exp(tri_loss21)) + torch.log(1+torch.exp(tri_loss23)))/2/3 
            tri_loss3  = (torch.log(1+torch.exp(tri_loss31)) + torch.log(1+torch.exp(tri_loss32)))/2/3 
            loss = seg_loss + lambda_t*(tri_loss1+tri_loss2+tri_loss3)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i_batch % print_freq==0:
            if len(_query_label)==1:
                if SC_DIS == 0:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}'.format(e, i_batch, len(train_loader), loss,seg_loss))
                else:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}+{:f}*{:f}'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss1))
            if len(_query_label)==2:
                print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}+{:f}x({:f}+{:f})'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss1,tri_loss2))
            if len(_query_label)==3:
                print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}+{:f}x({:f}+{:f}+{:f})'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss1,tri_loss2,tri_loss3))
        
    

    with torch.no_grad():
        save_path = root_save_path +'epoch-{}/'.format(e) 
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        ### 常规验证            
        dice_list = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=6,test_bs=5)
        val_dc = sum(dice_list)/len(dice_list)
        print('##conven## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f = open(txt_path, "a+")
        f.write('##conven## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f.close()
        ### ours1a 验证
        #dice_list1n = evaluate_fss_ours1a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        #dice_list1n = evaluate_fss_ours3a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        dice_list1n = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=3,test_bs=5)
        #val_dc1n = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,test_bs=5,Norm='Slice')
        val_dc1n = sum(dice_list1n)/len(dice_list1n)
        print('##ours1a_n## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list1n[0],dice_list1n[1],dice_list1n[2],val_dc1n))
        f = open(txt_path, "a+")
        f.write('##ours1a_n## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list1n[0],dice_list1n[1],dice_list1n[2],val_dc1n))
        f.close()
        ### ours1a 验证
        #dice_list1c = evaluate_fss_ours1a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='Case')
        #dice_list1c = evaluate_fss_ours3b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        dice_list1c = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=8,test_bs=5)
        val_dc1c = sum(dice_list1c)/len(dice_list1c)
        print('##ours1a_c## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list1c[0],dice_list1c[1],dice_list1c[2],val_dc1c))
        f = open(txt_path, "a+")
        f.write('##ours1a_c## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list1c[0],dice_list1c[1],dice_list1c[2],val_dc1c))
        f.close()
        ### ours1a 验证
        # dice_list1s = evaluate_fss_ours1a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='Slice')
        # dice_list1s = evaluate_fss_ours3c(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        # dice_list1s = evaluate_fss_ours1b_w(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        dice_list1s = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=1,test_bs=5)
        val_dc1s = sum(dice_list1s)/len(dice_list1s)
        print('##ours1a_s## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list1s[0],dice_list1s[1],dice_list1s[2],val_dc1s))
        f = open(txt_path, "a+")
        f.write('##ours1a_s## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list1s[0],dice_list1s[1],dice_list1s[2],val_dc1s))
        f.close()

        ### ours1b 验证
        #dice_list2n = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None')
        #dice_list2n = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=7,test_bs=5)
        dice_list2n = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='Slice')
        val_dc2n = sum(dice_list2n)/len(dice_list2n)
        print('##ours1b_n## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list2n[0],dice_list2n[1],dice_list2n[2],val_dc2n))
        f = open(txt_path, "a+")
        f.write('##ours1b_n## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list2n[0],dice_list2n[1],dice_list2n[2],val_dc2n))
        f.close()
        ### ours1b 验证
        #dice_list2c = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='Case')
        #dice_list2c = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=8,test_bs=5)
        dice_list2c = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=3,Norm='Slice')
        val_dc2c = sum(dice_list2c)/len(dice_list2c)
        print('##ours1b_c## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list2c[0],dice_list2c[1],dice_list2c[2],val_dc2c))
        f = open(txt_path, "a+")
        f.write('##ours1b_c## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list2c[0],dice_list2c[1],dice_list2c[2],val_dc2c))
        f.close()
        ### ours1b 验证
        #dice_list2s = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='Slice')
        #dice_list2s = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=9,test_bs=5)
        dice_list2s = evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=8,Norm='Slice')
        val_dc2s = sum(dice_list2s)/len(dice_list2s)
        print('##ours1b_s## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list2s[0],dice_list2s[1],dice_list2s[2],val_dc2s))
        f = open(txt_path, "a+")
        f.write('##ours1b_s## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list2s[0],dice_list2s[1],dice_list2s[2],val_dc2s))
        f.close()

    if val_dc>best_val_dc:
        best_val_dc = val_dc
        best_e = e
    if val_dc1n>best_val_dc1n:
        best_val_dc1n = val_dc1n
        best_e1n = e
    if val_dc1c>best_val_dc1c:
        best_val_dc1c = val_dc1c
        best_e1c = e
    if val_dc1s>best_val_dc1s:
        best_val_dc1s = val_dc1s
        best_e1s = e
    if val_dc2n>best_val_dc2n:
        best_val_dc2n = val_dc2n
        best_e2n = e
    if val_dc2c>best_val_dc2c:
        best_val_dc2c = val_dc2c
        best_e2c = e
    if val_dc2s>best_val_dc2s:
        best_val_dc2s = val_dc2s
        best_e2s = e
    
    PATH = model_path + 'epoch-{}.pth'.format(e)
    torch.save({'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'best_DC': best_val_dc}, PATH)
print('Best Epoch {:d} Avg Val dice: {:.1f}'.format(best_e,best_val_dc))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val dice: {:.1f} \n'.format(best_e,best_val_dc))
f.close()

print('Best Epoch {:d} Avg Val ours1a_n_dice: {:.1f}'.format(best_e1n,best_val_dc1n))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1a_n_dice: {:.1f} \n'.format(best_e1n,best_val_dc1n))
f.close()
print('Best Epoch {:d} Avg Val ours1a_c_dice: {:.1f}'.format(best_e1c,best_val_dc1c))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1a_c_dice: {:.1f} \n'.format(best_e1c,best_val_dc1c))
f.close()
print('Best Epoch {:d} Avg Val ours1a_s_dice: {:.1f}'.format(best_e1s,best_val_dc1s))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1a_s_dice: {:.1f} \n'.format(best_e1s,best_val_dc1s))
f.close()

print('Best Epoch {:d} Avg Val ours1b_n_dice: {:.1f}'.format(best_e2n,best_val_dc2n))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1b_n_dice: {:.1f} \n'.format(best_e2n,best_val_dc2n))
f.close()
print('Best Epoch {:d} Avg Val ours1b_c_dice: {:.1f}'.format(best_e2c,best_val_dc2c))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1b_c_dice: {:.1f} \n'.format(best_e2c,best_val_dc2c))
f.close()
print('Best Epoch {:d} Avg Val ours1b_s_dice: {:.1f}'.format(best_e2s,best_val_dc2s))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val ours1b_s_dice: {:.1f} \n'.format(best_e2s,best_val_dc2s))
f.close()

























































