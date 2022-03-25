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
from loss.evaluator import *



os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
max_free = np.argmax(memory_gpu)

print("choose gpu %d free %d MiB"%(max_free, memory_gpu[max_free]))
os.environ['CUDA_VISIBLE_DEVICES']=str(max_free)


# train_image_path = './datasets/Train-Image.h5'
# train_label_path = './datasets/Train-Label.h5'
# support_file = './datasets/FSS-Eval-36.h5'
# query_path = ['./datasets/FSS-Eval-37.h5','./datasets/FSS-Eval-38.h5','./datasets/FSS-Eval-39.h5']

# train_bs = 2  
# val_bs = 8
# train_iteration = 25#//train_bs
# val_iteration = 25#//val_bs
# num_epoch = 15
# num_support = 8
# model_path = './result_pretrain/SE/'

train_image_path = './datasets/CT/Train-Image.h5'
train_label_path = './datasets/CT/Train-Label.h5'
support_file = './datasets/CT/FSS-Eval-25.h5'
query_path = ['./datasets/CT/FSS-Eval-26.h5','./datasets/CT/FSS-Eval-27.h5','./datasets/CT/FSS-Eval-28.h5',
                './datasets/CT/FSS-Eval-29.h5','./datasets/CT/FSS-Eval-30.h5']

train_bs = 2
val_bs = 8
train_iteration = 25#//train_bs
val_iteration = 25#//val_bs
num_epoch = 30
num_support = 8
model_path = './result_pretrain/SE_CT/'

reload_mdoel = 0
print_freq = 25
 


# val_image_path = './datasets/MRI/Val-Image.h5'
# val_label_path = './datasets/MRI/Val-Label.h5'

#net = FewShotSegmentorDoubleSDnet().cuda()
net = my_fss().cuda()
#net = my_fss_ex().cuda()

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

train_sampler = OneShotBatchSampler(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

# val_sampler = OneShotBatchSampler(val_dataset.label, 'val', "fold1", batch_size=val_bs, iteration=val_iteration)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=train_bs,shuffle=True)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=val_bs,shuffle=False)

optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

# class combine_criterion():
#     def __init__(self):
#         super(combine_criterion, self).__init__()
#         self.criterion1 = DiceLoss2D()
#         self.criterion2 = nn.BCELoss()
#     def forward(self,input,target):
#         loss = self.criterion1(input,target) + self.criterion2(input,target)
#         return loss

# criterion = combine_criterion()
# criterion_ds1 = combine_criterion() 
# criterion_ds2 = combine_criterion() 
# criterion_ds3 = combine_criterion() 

criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

best_val_dc = 0
best_e = 0

best_ft_val_dc = 0
best_ft_e = 0

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
    
        query_label = train_loader.batch_sampler.query_label
        
        support_image, query_image, support_label, query_label = split_batch(image, label, int(query_label))
        #print(image.min(),image.max())
        #print(label.min(),label.max())
        #print(support_image.shape,query_image.shape,support_label.shape,query_label.shape)
        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        seg = net(condition_input,query_image)
        #seg,ds1,ds2,ds3 = net(condition_input,query_image)

        #query_label.squeeze_(dim=1)
        
        #print(seg.shape,query_label.shape)
        #loss = criterion1(seg, query_label)
        #print(ds1.shape,ds2.shape,ds3.shape)

        #loss = criterion1(seg, query_label) + 1.0/3*criterion1(ds1, query_label) + 1.0/3*criterion1(ds2, query_label) +1.0/3*criterion1(ds3, query_label)
        #loss += criterion2(seg, query_label) + 1.0/3*criterion2(ds1, query_label) + 1.0/3*criterion2(ds2, query_label) +1.0/3*criterion2(ds3, query_label)
        loss = criterion1(seg, query_label) + criterion2(seg, query_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i_batch % print_freq==0:
            print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}'.format(e, i_batch, len(train_loader), loss))
    
    with torch.no_grad():
        save_path = root_save_path +'epoch-{}/'.format(e) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)        
        dice_list = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=num_support)
        val_dc = sum(dice_list)/len(dice_list)
        print('Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f = open(txt_path, "a+")
        f.write('Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f.close()
    # ft_dice_list = evaluate_fss_finetune(net,support_file,query_path,save_path,query_label=1,Num_support=6)
    # ft_val_dc = sum(ft_dice_list)/len(ft_dice_list)
    # print('##finetune## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,ft_dice_list[0],ft_dice_list[1],ft_dice_list[2],ft_val_dc))
    # f = open(txt_path, "a+")
    # f.write('##finetune## Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,ft_dice_list[0],ft_dice_list[1],ft_dice_list[2],ft_val_dc))
    # f.close()
    if val_dc>best_val_dc:
        best_val_dc = val_dc
        best_e = e
    # if ft_val_dc>best_ft_val_dc:
    #     best_ft_val_dc = ft_val_dc
    #     best_ft_e = e
    PATH = model_path + 'epoch-{}.pth'.format(e)
    torch.save({'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'best_DC': best_val_dc}, PATH)
print('Best Epoch {:d} Avg Val dice: {:.1f}'.format(best_e,best_val_dc))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val dice: {:.1f} \n'.format(best_e,best_val_dc))
f.close()
# print('Best Epoch {:d} Avg Val ft_dice: {:.1f}'.format(best_ft_e,best_ft_val_dc))
# f = open(txt_path, "a+")
# f.write('Best Epoch {:d} Avg Val ft_dice: {:.1f} \n'.format(best_ft_e,best_ft_val_dc))
# f.close()
























































