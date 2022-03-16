# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 20:35:42 2020

@author: Sunly
"""

# Mapping
# 0 is background
# 1 is liver
# 2 is right kidney
# 3 is left kidney
# 4 is spleen

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn

# from Sampler import *

# from Segmentor import SegMenTor
# from Network import *

import torch.optim as optim
import numpy as np
#from nn_common_modules import losses as additional_losses
import os


# lab_list_fold = {"fold1": {"train": [2, 6, 7, 8, 9], "val": [1]},
#                  "fold2": {"train": [1, 6, 7, 8, 9], "val": [2]},
#                  "fold3": {"train": [1, 2, 8, 9], "val": [6, 7]},
#                  "fold4": {"train": [1, 2, 6, 7], "val": [8, 9]}}

lab_list_fold = {"fold1": {"train": [2, 3, 4], "val": [1]},
                 "fold2": {"train": [1, 3, 4], "val": [2]},
                 "fold3": {"train": [1, 2, 4], "val": [3]},
                 "fold4": {"train": [1, 2, 3], "val": [4]}
                 }


def get_lab_list(phase, fold):
    return lab_list_fold[fold][phase]


###3D
# def get_class_slices(labels, i):
#     num_slices, H, W, D = labels.shape
#     # thresh = 0.005
#     thresh = 0
#     total_slices = labels == i
#     pixel_sum = np.sum(total_slices, axis=(1, 2, 3)).squeeze()
#     pixel_sum = pixel_sum / (H * W * D)
#     threshold_list = [idx for idx, slice in enumerate(
#         pixel_sum) if slice > thresh]
#     return threshold_list

###2D
def get_class_slices(labels, i):
    num_slices, H, W = labels.shape
    thresh = 0.005
    #thresh = 0
    total_slices = labels == i
    pixel_sum = np.sum(total_slices, axis=(1, 2)).squeeze()
    pixel_sum = pixel_sum / (H * W )
    threshold_list = [idx for idx, slice in enumerate(
        pixel_sum) if slice > thresh]
    return threshold_list

def get_index_dict(labels, lab_list):
    index_list = {i: get_class_slices(labels, i) for i in lab_list}
    p = [1 - (len(val) / len(labels)) for val in index_list.values()]
    p = p / np.sum(p)
    return index_list, p

class OneShotBatchSampler:
    def _gen_query_label(self):
        """
        Returns a query label uniformly from the label list of current phase. Also returns indexes of the slices which contain that label

        :return: random query label, index list of slices with generated class available
        """
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):
        '''

        '''
        super(OneShotBatchSampler, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        #self.index_dict, self.p = get_index_dict(labels, [1])



    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        self.n = 0
        return self

    def __next__(self):
        """
        Called on each iteration to return slices a random class label. On each iteration gets a random class label from label list and selects 2 x batch_size slices uniformly from index list
        :return: randomly select 2 x batch_size slices of a class label for the given iteration
        """
        if self.n > self.iteration:
            raise StopIteration

        self.query_label = self._gen_query_label()
        #self.query_label = 1
        self.index_list = self.index_dict[self.query_label]
        batch = np.random.choice(self.index_list, size=2 * self.batch_size)

        self.n += 1
        return batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        :return: number os iterations
        """

        return self.iteration

class OneShotBatchSampler_ms:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):

        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        self.index_list = self.index_dict[2]
        batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        self.index_list = self.index_dict[3]
        batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        self.index_list = self.index_dict[4]
        batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        batch = np.concatenate([batch1, batch2, batch3], axis=0)

        self.n += 1
        return batch

    def __len__(self):
        return self.iteration

class OneShotBatchSampler_ms2:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms2, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        # for k in self.index_dict.keys():
        #     print(k)
        index1 = self.index_dict[2]
        index2 = self.index_dict[3]
        index3 = self.index_dict[4]

        index = []
        for i in range(len(index1)):
            if index1[i] in index2 and index1[i] in index3:
                index.append(index1[i])
        self.index = index
        print(len(index1),len(index2),len(index3),len(index))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        # self.index_list = self.index_dict[2]
        # batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[3]
        # batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[4]
        # batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # batch = np.concatenate([batch1, batch2, batch3], axis=0)
        batch = np.random.choice(self.index,size=2 * self.batch_size)

        self.n += 1
        return batch

    def __len__(self):
        return self.iteration

class OneShotBatchSampler_ms3:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms3, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        # for k in self.index_dict.keys():
        #     print(k)
        # index1 = self.index_dict[2]
        # index2 = self.index_dict[3]
        # index3 = self.index_dict[4]

        # index = []
        # for i in range(len(index1)):
        #     if index1[i] in index2 and index1[i] in index3:
        #         index.append(index1[i])
        # self.index = index
        # print(len(index1),len(index2),len(index3),len(index))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        # self.index_list = self.index_dict[2]
        # batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[3]
        # batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[4]
        # batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # batch = np.concatenate([batch1, batch2, batch3], axis=0)
        label = [2,3,4]
        self.query_label = label[self.n%3]
        batch = np.random.choice(self.index_dict[self.query_label],size=2 * self.batch_size)

        self.n += 1
        return batch

    def __len__(self):
        return self.iteration

class OneShotBatchSampler_ms4:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms4, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        # for k in self.index_dict.keys():
        #     print(k)
        index1 = list(self.index_dict.values())[0]
        index2 = list(self.index_dict.values())[1]
        index3 = list(self.index_dict.values())[2]

        index12 = list(set(index1).intersection(set(index2)))
        index13 = list(set(index1).intersection(set(index3)))
        index23 = list(set(index2).intersection(set(index3)))
        index_ = list(set(index12).union(index13,index23))
        index = list(set(index1).union(index2,index3))
        index_id_dict={}
        for i in range(len(index)):
            index_id_dict[index[i]] = [index[i] in index1,index[i] in index2,index[i] in index3]
        print(len(index1),len(index2),len(index3),len(index12),len(index13),len(index23),len(index_),len(index))
        self.index = index
        self.index_id_dict = index_id_dict

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        # self.index_list = self.index_dict[2]
        # batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[3]
        # batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[4]
        # batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # batch = np.concatenate([batch1, batch2, batch3], axis=0)
        batch_s = np.random.choice(self.index,size=1)
        same_index = []
        for k,v in self.index_id_dict.items():
            if v == self.index_id_dict[batch_s[0]]:
                same_index.append(k)
        if self.batch_size>1:
            batch_s = np.concatenate([batch_s,np.random.choice(same_index,size=self.batch_size-1)],axis=0)
        batch_q = np.random.choice(same_index,size=self.batch_size)
        self.query_label = []
        assert self.index_id_dict[batch_s[0]] == self.index_id_dict[batch_q[0]]

        if self.index_id_dict[batch_s[0]][0] == 1:
            self.query_label.append(2)
        if self.index_id_dict[batch_s[0]][1] == 1:
            self.query_label.append(3)    
        if self.index_id_dict[batch_s[0]][2] == 1:
            self.query_label.append(4)
    
        batch = np.concatenate([batch_s,batch_q],axis=0)
        self.n += 1
        return batch

    def __len__(self):
        return self.iteration

class OneShotBatchSampler_ms5:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms5, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        # for k in self.index_dict.keys():
        #     print(k)
        index1 = self.index_dict[2]
        index2 = self.index_dict[3]
        index3 = self.index_dict[4]

        index12 = list(set(index1).intersection(set(index2)))
        index13 = list(set(index1).intersection(set(index3)))
        index23 = list(set(index2).intersection(set(index3)))
        index_ = list(set(index12).union(index13,index23))
        index = list(set(index1).union(index2,index3))
        index_sc = list(set(index).difference(set(index_))) 
        index_id_dict={}
        for i in range(len(index_sc)):
            index_id_dict[index_sc[i]] = [index_sc[i] in index1,index_sc[i] in index2,index_sc[i] in index3]
        assert len(index_) + len(index_sc) == len(index)
        print(len(index1),len(index2),len(index3),len(index12),len(index13),len(index23),len(index_),len(index_sc),len(index))
        self.index = index
        self.index_sc = index_sc
        self.index_id_dict = index_id_dict

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        # self.index_list = self.index_dict[2]
        # batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[3]
        # batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[4]
        # batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # batch = np.concatenate([batch1, batch2, batch3], axis=0)
        batch_s = np.random.choice(self.index_sc,size=1)
        same_index = []
        for k,v in self.index_id_dict.items():
            if v == self.index_id_dict[batch_s[0]]:
                same_index.append(k)
        if self.batch_size>1:
            batch_s = np.concatenate([batch_s,np.random.choice(same_index,size=self.batch_size-1)],axis=0)
        batch_q = np.random.choice(same_index,size=self.batch_size)
        self.query_label = []
        assert self.index_id_dict[batch_s[0]] == self.index_id_dict[batch_q[0]]

        if self.index_id_dict[batch_s[0]][0] == 1:
            self.query_label.append(2)
        if self.index_id_dict[batch_s[0]][1] == 1:
            self.query_label.append(3)    
        if self.index_id_dict[batch_s[0]][2] == 1:
            self.query_label.append(4)
        assert len(self.query_label) == 1
    
        batch = np.concatenate([batch_s,batch_q],axis=0)
        self.n += 1
        return batch

    def __len__(self):
        return self.iteration

class OneShotBatchSampler_ms6:
    def _gen_query_label(self):
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

    def __init__(self, labels, phase, fold, batch_size, iteration=500):

        super(OneShotBatchSampler_ms6, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)
        # for k in self.index_dict.keys():
        #     print(k)
        index1 = self.index_dict[2]
        index2 = self.index_dict[3]
        index3 = self.index_dict[4]

        index12 = list(set(index1).intersection(set(index2)))
        index13 = list(set(index1).intersection(set(index3)))
        index23 = list(set(index2).intersection(set(index3)))
        index_ = list(set(index12).union(index13,index23))
        index = list(set(index1).union(index2,index3))
        index_sc = list(set(index).difference(set(index_))) 
        index_id_dict={}
        for i in range(len(index_)):
            index_id_dict[index_[i]] = [index_[i] in index1,index_[i] in index2,index_[i] in index3]
        assert len(index_) + len(index_sc) == len(index)
        print(len(index1),len(index2),len(index3),len(index12),len(index13),len(index23),len(index_),len(index_sc),len(index))
        self.index = index
        self.index_ = index_
        self.index_id_dict = index_id_dict

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.iteration:
            raise StopIteration
        
        #self.query_label = [2,3,4]
        # self.index_list = self.index_dict[2]
        # batch1 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[3]
        # batch2 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # self.index_list = self.index_dict[4]
        # batch3 = np.random.choice(self.index_list, size=2 * self.batch_size)
        # batch = np.concatenate([batch1, batch2, batch3], axis=0)
        batch_s = np.random.choice(self.index_,size=1)
        same_index = []
        for k,v in self.index_id_dict.items():
            if v == self.index_id_dict[batch_s[0]]:
                same_index.append(k)
        if self.batch_size>1:
            batch_s = np.concatenate([batch_s,np.random.choice(same_index,size=self.batch_size-1)],axis=0)
        batch_q = np.random.choice(same_index,size=self.batch_size)
        self.query_label = []
        assert self.index_id_dict[batch_s[0]] == self.index_id_dict[batch_q[0]]

        if self.index_id_dict[batch_s[0]][0] == 1:
            self.query_label.append(2)
        if self.index_id_dict[batch_s[0]][1] == 1:
            self.query_label.append(3)    
        if self.index_id_dict[batch_s[0]][2] == 1:
            self.query_label.append(4)
        assert len(self.query_label) >= 2
    
        batch = np.concatenate([batch_s,batch_q],axis=0)
        self.n += 1
        return batch

    def __len__(self):
        return self.iteration


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

# def get_whole_dataset(file_path):
    
#     data = h5py.File(file_path, 'r')
    
#     simple_dataset = SimpleData(data['Image'][:], data['Label'][:])

#     return simple_dataset,data['case_start_index']

class ClassData(data.Dataset):
    def __init__(self, Image, Label,class_index,binary):

        self.image = Image
        self.label = Label
        self.class_index = class_index
        self.binary = binary
        self.index_dict, self.p = get_index_dict(self.label,[class_index])
        #print(self.index_dict)
        #print(len(self.label),len(self.index_dict[self.class_index]))

    def __getitem__(self, index):
        _index = self.index_dict[self.class_index][index]
        img = torch.from_numpy(self.image[_index])
        label = torch.from_numpy(self.label[_index])
        if self.binary:
            label = (label== self.class_index).type(torch.FloatTensor)
        #print(label.min(),label.max())
        return img, label

    def __len__(self):
        # return len(self.label)
        return len(self.index_dict[self.class_index])
    
def get_class_dataset(image_path,label_path,class_index=4,binary=True):
    
    image_train = h5py.File(image_path, 'r')
    label_train = h5py.File(label_path, 'r')
    
    class_dataset = ClassData(image_train['Image'][:], label_train['Label'][:],class_index,binary)

    return class_dataset

class DiceLoss2D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1
 
        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1)+ smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 1)    

def split_batch(image, label, query_label):
    batch_size = len(image) // 2
    input1 = image[0:batch_size, :, :, :].cuda()
    input2 = image[batch_size:, :, :, :].cuda()
    label1 = (label[0:batch_size, :, :] == query_label).type(torch.FloatTensor).cuda()
    label2 = (label[batch_size:, :, :] == query_label).type(torch.FloatTensor).cuda()

    return input1, input2, label1, label2

def split_batch_ms(image, label, query_label):
    num_query_label = len(query_label)
    batch_size = len(image) // num_query_label// 2
    input1 = torch.cat([image[0:batch_size],image[batch_size*2:batch_size*3],image[batch_size*4:batch_size*5]],dim=0).cuda()
    input2 = torch.cat([image[batch_size:batch_size*2],image[batch_size*3:batch_size*4],image[batch_size*5:batch_size*6]],dim=0).cuda()
    label1 = torch.cat([label[0:batch_size] == query_label[0],label[batch_size*2:batch_size*3] == query_label[1],\
        label[batch_size*4:batch_size*5] == query_label[2]],dim=0).type(torch.FloatTensor).cuda()
    label2 = torch.cat([label[batch_size:batch_size*2] == query_label[0],label[batch_size*3:batch_size*4] == query_label[1],\
        label[batch_size*5:batch_size*6] == query_label[2]],dim=0).type(torch.FloatTensor).cuda()

    #print(np.unique(label1.cpu().numpy()),np.unique(label2.cpu().numpy()))

    return input1, input2, label1, label2

def split_batch_ms2(image, label, query_label):
    batch_size = len(image) // 2
    input1 = image[0:batch_size, :, :, :].cuda()
    input2 = image[batch_size:, :, :, :].cuda()
    label1 = ( (label[0:batch_size, :, :] == query_label[0]) + (label[0:batch_size, :, :] == query_label[1]) +(label[0:batch_size, :, :] == query_label[2]) >0 ).type(torch.FloatTensor).cuda()
    label2 = ( (label[batch_size:, :, :] == query_label[0]) + (label[batch_size:, :, :] == query_label[1])+(label[batch_size:, :, :] == query_label[2]) >0 ).type(torch.FloatTensor).cuda()

    return input1, input2, label1, label2

def split_batch_ms4(image, label, query_label):
    batch_size = len(image) // 2
    input1 = image[0:batch_size, :, :, :].cuda()
    input2 = image[batch_size:, :, :, :].cuda()
    label1 = label[0:batch_size, :, :] == query_label[0]
    label2 = label[batch_size:, :, :] == query_label[0]
    for i in range(1,len(query_label)):
        label1 = torch.cat([label1,label[0:batch_size, :, :] == query_label[i]],dim=0)
        label2 = torch.cat([label2,label[batch_size:, :, :] == query_label[i]],dim=0)
    label1 = label1.type(torch.FloatTensor).cuda()
    label2 = label2.type(torch.FloatTensor).cuda()
    input1 = input1.repeat(len(query_label),1,1,1)
    input2 = input2.repeat(len(query_label),1,1,1)

    return input1, input2, label1, label2

def dice_score_binary(vol_output, ground_truth):
    #vol_output = vol_output>0.5
    #vol_output = vol_output>0.5

    bs,_,_,_ = vol_output.size() 
    vol_output = vol_output.view(bs,-1)
    ground_truth = ground_truth.view(bs,-1) 
    ground_truth = ground_truth.type(torch.FloatTensor)
    vol_output = vol_output.type(torch.FloatTensor)
    inter = 2 * torch.sum(torch.mul(ground_truth, vol_output),dim=-1)
    union = torch.sum(ground_truth,dim=-1) + torch.sum(vol_output,dim=-1) + 0.0001
    return (inter/union).mean()

def dice_func(vol_output, ground_truth):
    #vol_output = vol_output>0.5
    vol_output = vol_output>0.5

    bs = vol_output.shape[0] 
    vol_output = vol_output.view(bs,-1)
    ground_truth = ground_truth.view(bs,-1) 
    ground_truth = ground_truth.type(torch.FloatTensor)
    vol_output = vol_output.type(torch.FloatTensor)
    inter = 2 * torch.sum(torch.mul(ground_truth, vol_output),dim=-1)
    union = torch.sum(ground_truth,dim=-1) + torch.sum(vol_output,dim=-1) + 0.0001
    return (inter/union).mean()
