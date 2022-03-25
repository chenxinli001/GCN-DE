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

from nn_common_modules import losses as additional_losses

import os

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
max_free = np.argmax(memory_gpu)

print("choose gpu %d free %d MiB"%(max_free, memory_gpu[max_free]))
os.environ['CUDA_VISIBLE_DEVICES']=str(max_free)

##### 将孙博代码改为2D版本，网络结构同源码 ####

train_image_path = './datasets/Train-Image.h5'
train_label_path = './datasets/Train-Label.h5'

val_image_path = './datasets/Val-Image.h5'
val_label_path = './datasets/Val-Label.h5'


train_bs = 1
val_bs = 8
train_iteration = 500//train_bs
val_iteration = 500//val_bs
num_epoch = 10

reload_mdoel = 0
print_freq = 50

lambda_d = 0

model_path = './result/ours3_ms_segbn_mse_ld0/'
net = FewShotSegmentorDoubleSDnet_ours3ms().cuda()

print(model_path)
txt_path = model_path + 'result.txt'
if not os.path.exists(model_path):
    os.makedirs(model_path)
f = open(txt_path, "a+")
f.write('train_bs:{}|iter:{}, val_bs:{}|iter:{}, num_epoch:{} \n'.format(train_bs,train_iteration, val_bs,val_iteration,num_epoch))
f.close()

train_dataset = get_simple_dataset(train_image_path,train_label_path)
val_dataset = get_simple_dataset(val_image_path,val_label_path)

train_sampler = OneShotBatchSampler(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

val_sampler = OneShotBatchSampler(val_dataset.label, 'val', "fold1", batch_size=val_bs, iteration=val_iteration)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)


optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

best_val_dc = 0
best_e = 0
last_query_label = 0

if reload_mdoel:
    checkpoint = torch.load(model_path + 'latest.pth')
    #checkpoint = torch.load(load_model_path + 'epoch-15.pth')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    best_DC = checkpoint['best_DC']
    print('Reload epoch {} model OK!, best DC is {}'.format(start_epoch, best_DC))
else:
    start_epoch = 0

for e in range(start_epoch,num_epoch+1):
    net.train()
    for i_batch, sampled_batch in enumerate(train_loader):
        #image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        #label = sampled_batch[1].type(torch.LongTensor).cuda()/4
        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        label = sampled_batch[1].type(torch.FloatTensor).cuda()
    
        this_query_label = train_loader.batch_sampler.query_label
        
        support_image, query_image, support_label, query_label = split_batch(image, label, int(this_query_label))
        #print(image.min(),image.max())
        #print(label.min(),label.max())
        #print(support_image.shape,query_image.shape,support_label.shape,query_label.shape)
        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg,seg_fea = net(condition_input,query_image)
        
        #query_label.squeeze_(dim=1)
        
        #loss = criterion(seg, label)
        seg_loss = criterion1(seg, query_label) + criterion2(seg, query_label)
        if last_query_label != 0:
            if this_query_label == last_query_label:
                dis_loss = ((seg_fea-last_seg_fea)**2).mean()
            else:
                dis_loss = (-(seg_fea-last_seg_fea)**2).mean()
            loss = seg_loss + lambda_d*dis_loss
        else:
            loss = seg_loss
            dis_loss = 0

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        last_query_label = this_query_label
        last_seg_fea = seg_fea.detach()
            
        if i_batch % print_freq==0:
            print('Epoch {:d} | Episode {:d}/{:d} | Seg Loss {:f}, Dis Loss {:f}, Total Loss {:f}'.format(e, i_batch, len(train_loader), \
                seg_loss,dis_loss,loss))
    
    with torch.no_grad():
        net.eval()
        dice_list = []
        for i_batch, sampled_batch in enumerate(val_loader):
            image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            label = sampled_batch[1].type(torch.FloatTensor).cuda()
            query_label = val_loader.batch_sampler.query_label
            support_image, query_image, support_label, query_label = split_batch(image, label, int(query_label))
            condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
            seg,_ = net(condition_input,query_image)
            dice = dice_score_binary(seg,query_label)*100
            dice_list.append(dice)
        val_dc = sum(dice_list)/len(dice_list)
        print('Epoch {:d} Val Sample dice: {:.1f}'.format(e,val_dc))
        f = open(txt_path, "a+")
        f.write('Epoch {:d} Val Sample dice: {:.1f} \n'.format(e,val_dc))
        f.close()
        if val_dc>best_val_dc:
            best_val_dc = val_dc
            best_e = e
            PATH = model_path + 'best.pth'
            torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_e,
                        'best_DC': best_val_dc}, PATH)
        if e%5==0:
            PATH = model_path + 'epoch-{}.pth'.format(e)
            torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': e,
                        'best_DC': best_val_dc}, PATH)
        PATH = model_path + 'latest.pth'
        torch.save({'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                    'best_DC': best_val_dc}, PATH)
print('Best Epoch {:d} Val Sample dice: {:.1f}'.format(best_e,best_val_dc))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Val Sample dice: {:.1f} \n'.format(best_e,best_val_dc))
f.close()
























































