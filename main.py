# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn

from Sampler import OneShotBatchSampler

from Segmentor import SegMenTor

import torch.optim as optim

from nn_common_modules import losses as additional_losses

# with h5py.File("Training-Image-Patches.h5","r") as r:
#     training_image_patches = r["Image"][:]
    

class SimpleData(data.Dataset):
    def __init__(self, Image, Label):

        self.image = Image
        self.label = Label

    def __getitem__(self, index):
        img = torch.from_numpy(self.image[index])
        label = torch.from_numpy(self.label[index])
        return img, label

    def __len__(self):
        return len(self.label)
    
def get_simple_dataset(image_path,label_path):
    
    image_train = h5py.File(image_path, 'r')
    label_train = h5py.File(label_path, 'r')
    
    simple_dataset = SimpleData(image_train['Image'][:], label_train['Label'][:])

    return simple_dataset

def split_batch(image, label, query_label):
    batch_size = len(image) // 2
    input1 = image[0:batch_size, :, :, :].cuda()
    input2 = image[batch_size:, :, :, :].cuda()
    label1 = (label[0:batch_size, :, :] == query_label).type(torch.FloatTensor).cuda()
    label2 = (label[batch_size:, :, :] == query_label).type(torch.FloatTensor).cuda()

    return input1, input2, label1, label2
    

train_image_path = './datasets/Train-Image.h5'
train_label_path = './datasets/Train-Label.h5'

val_image_path = './datasets/Val-Image.h5'
val_label_path = './datasets/Val-Label.h5'

training_dataset = get_simple_dataset(training_image_path,training_label_path)

train_sampler = OneShotBatchSampler(training_dataset.label, 'train', "fold1", batch_size=1, iteration=500)
train_loader = torch.utils.data.DataLoader(training_dataset, batch_sampler=train_sampler)

print_freq = 1

SegNet = SegMenTor().cuda()

optimizer = optim.Adam(SegNet.parameters(), lr=0.001)

# criterion = additional_losses.DiceLoss()
criterion = nn.BCELoss()

num_epoch = 100

print_freq = 10

for e in range(num_epoch):

    for i_batch, sampled_batch in enumerate(train_loader):
        image = sampled_batch[0].unsqueeze_(dim=1).cuda()
        label = sampled_batch[1].unsqueeze_(dim=1).cuda()
    
        query_label = train_loader.batch_sampler.query_label
        
        support_image, query_image, support_label, query_label = split_batch(image, label, int(query_label))
        
        seg = SegNet(support_image, query_image, support_label)
        
        query_label.squeeze_(dim=1)
        
        # loss = criterion(seg, query_label, binary=True)
        loss = criterion(seg, query_label)
        
        loss.backward(retain_graph=True)
        
        optimizer.step()
        
        if i_batch % print_freq==0:
            print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}'.format(e, i_batch, len(train_loader), loss  ))
























































