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

from Segmentor import SegMenTor
from Network import *

import torch.optim as optim

import os
from evaluator import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default=None)

    return parser.parse_args()


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
max_free = np.argmax(memory_gpu)

print("choose gpu %d free %d MiB"%(max_free, memory_gpu[max_free]))
os.environ['CUDA_VISIBLE_DEVICES']=str(max_free)

args = get_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lab_list_fold = {"fold1": {"train": [2, 3, 4], "val": 1},#liver
                 "fold2": {"train": [1, 3, 4], "val": 2},#right kidney
                 "fold3": {"train": [1, 2, 4], "val": 3},#left kidney
                 "fold4": {"train": [1, 2, 3], "val": 4} #spleen
                 }

organ_fold_dict = {'liver':'fold1','right kidney':'fold2','left kidney':'fold3','spleen':'fold4'}
organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                    "right kidney": {"train": [1, 3, 4], "val": 2},
                    "left kidney": {"train": [1, 2, 4], "val": 3},
                    "spleen": {"train": [1, 2, 3], "val": 4} 
                    }

DataSet = 'MRI' # 'MRI'or'CT' 
ObjectOrgan = 'liver'

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
    train_bs = 32 
    num_epoch = 100
    model_path = './result_pretrain/pretrain_ae/MRI/'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 32
    num_epoch = 100
    model_path = './result_pretrain/pretrain_ae/CT/'

model_path = model_path 
if not os.path.exists(model_path):
    os.makedirs(model_path)

reload_mdoel = 0 

print(model_path)

data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']


net = autoencoder().cuda()

train_dataset = SimpleData(whole_image,whole_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

lr=1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

criterion = nn.MSELoss()

if reload_mdoel:
    checkpoint = torch.load(model_path + 'latest.pth')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

for e in range(start_epoch,num_epoch+1):
    net.train()
    for i_batch, sampled_batch in enumerate(train_loader):
        image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
        noise_image = image + torch.randn_like(image)*4e-1
        denoise_image = net(noise_image)
        loss = criterion(denoise_image, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_batch % 100==0:
            print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}'.format(e, i_batch, len(train_loader), loss))

    if e%10==0:
        PATH = model_path + 'ae_epoch-{}.pth'.format(e)
        torch.save({'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()}, PATH)       
    













































































































