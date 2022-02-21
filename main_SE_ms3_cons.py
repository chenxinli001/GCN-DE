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

from nn_common_modules import losses as additional_losses

import os
from evaluator import *

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

 
train_bs = 2
val_bs = 8
train_iteration = 25#//train_bs
val_iteration = 25#//val_bs
num_epoch = 20

reload_mdoel = 0
print_freq = 25

epoch_sc = 10
 
lambda_t = 10

#model_path = './result/SE_ms/' 
#net = FewShotSegmentorDoubleSDnet().cuda()

model_path = './result_pretrain/SE_ms/'
net = my_fss_fea().cuda()
 
print(model_path)
txt_path = model_path + 'result.txt'
if not os.path.exists(model_path): 
    os.makedirs(model_path)

support_file = './datasets/FSS-Eval-36.h5'
query_path = ['./datasets/FSS-Eval-37.h5','./datasets/FSS-Eval-38.h5','./datasets/FSS-Eval-39.h5']
root_save_path = model_path + 'nii_save/'
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)

f = open(txt_path, "a+")
f.write('train_bs:{}|iter:{}, val_bs:{}|iter:{}, num_epoch:{} \n'.format(train_bs,train_iteration, val_bs,val_iteration,num_epoch))
f.close()

train_dataset = get_simple_dataset(train_image_path,train_label_path)
val_dataset = get_simple_dataset(val_image_path,val_label_path)

train_sampler_sc = OneShotBatchSampler_ms5(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader_sc = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_sc)

train_sampler_mc = OneShotBatchSampler_ms6(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader_mc = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_mc)


train_sampler = OneShotBatchSampler_ms5(train_dataset.label, 'train', "fold1", batch_size=train_bs, iteration=train_iteration)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

val_sampler = OneShotBatchSampler(val_dataset.label, 'val', "fold1", batch_size=val_bs, iteration=val_iteration)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)


optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

criterion1 = DiceLoss2D()
criterion2 = nn.BCELoss()

best_val_dc = 0
best_e = 0



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
    optimizer.zero_grad()
    if e<=epoch_sc:
        for i_batch, sampled_batch in enumerate(train_loader_sc):
            #image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            #label = sampled_batch[1].type(torch.LongTensor).cuda()/4
            image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            label = sampled_batch[1].type(torch.FloatTensor).cuda()
        
            _query_label = train_loader_sc.batch_sampler.query_label
            assert len(_query_label)==1
            
            # support_image, query_image, support_label, query_label = split_batch_ms(image,label,query_label=[2,3,4])
            # support_image, query_image, support_label, query_label = split_batch_ms2(image,label,query_label=[2,3,4])
            support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
            
            #support_image, query_image, support_label, query_label = split_batch(image,label,query_label)

            #print(image.min(),image.max())
            #print(label.min(),label.max())
            #print(support_image.shape,query_image.shape,support_label.shape,query_label.shape)
            condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
            seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)
            
            #query_label.unsqueeze_(dim=1)
            
            #loss = criterion(seg, label)
            loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if i_batch % print_freq==0:
                print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}'.format(e, i_batch, len(train_loader), loss))

    else: 
        for i_batch, sampled_batch in enumerate(train_loader_mc):
            #image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            #label = sampled_batch[1].type(torch.LongTensor).cuda()/4
            image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
            label = sampled_batch[1].type(torch.FloatTensor).cuda()
        
            _query_label = train_loader_mc.batch_sampler.query_label
            assert len(_query_label) >= 2
            
            # support_image, query_image, support_label, query_label = split_batch_ms(image,label,query_label=[2,3,4])
            # support_image, query_image, support_label, query_label = split_batch_ms2(image,label,query_label=[2,3,4])
            support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
             
            #support_image, query_image, support_label, query_label = split_batch(image,label,query_label)

            #print(image.min(),image.max())
            #print(label.min(),label.max())
            #print(support_image.shape,query_image.shape,support_label.shape,query_label.shape)
            condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
            seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)
            
            #query_label.unsqueeze_(dim=1)
            
            #loss = criterion(seg, label)
            seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)

            if len(_query_label)==2:
                tri_loss1 = ( torch.abs((s_feature[4][0:train_bs]-q_feature[4][0:train_bs])).mean() - 
                    torch.abs((s_feature[4][0:train_bs]-q_feature[4][train_bs:train_bs*2])).mean() )
                tri_loss2 = ( torch.abs((s_feature[4][train_bs:train_bs*2]-q_feature[4][train_bs:train_bs*2])).mean() -
                    (torch.abs(s_feature[4][train_bs:train_bs*2]-q_feature[4][0:train_bs])).mean() )
                tri_loss1 =  torch.log(1+torch.exp(tri_loss1))/2
                tri_loss2  = torch.log(1+torch.exp(tri_loss2))/2
                loss = seg_loss + lambda_t*(tri_loss1+tri_loss2)
            if len(_query_label)==3:
                tri_loss1 = ( torch.abs((s_feature[4][0:train_bs]-q_feature[4][0:train_bs])).mean() - 
                    (torch.abs(s_feature[4][0:train_bs]-q_feature[4][train_bs:train_bs*2])).mean() -
                    (torch.abs(s_feature[4][0:train_bs]-q_feature[4][train_bs*2:train_bs*3])).mean() )

                tri_loss2= ( torch.abs((s_feature[4][train_bs:train_bs*2]-q_feature[4][train_bs:train_bs*2])).mean() - 
                    (torch.abs(s_feature[4][train_bs:train_bs*2]-q_feature[4][0:train_bs])).mean() -
                    (torch.abs(s_feature[4][train_bs:train_bs*2]-q_feature[4][train_bs*2:train_bs*3])).mean() )
                
                tri_loss3 = ( torch.abs((s_feature[4][train_bs*2:train_bs*3]-q_feature[4][train_bs*2:train_bs*3])).mean() - 
                    (torch.abs(s_feature[4][train_bs*2:train_bs*3]-q_feature[4][0:train_bs])).mean() -
                    (torch.abs(s_feature[4][train_bs*2:train_bs*3]-q_feature[4][train_bs:train_bs*2])).mean() )
                
                tri_loss1 =  torch.log(1+torch.exp(tri_loss1))/3
                tri_loss2  = torch.log(1+torch.exp(tri_loss2))/3
                tri_loss3  = torch.log(1+torch.exp(tri_loss3))/3
                loss = seg_loss + lambda_t*(tri_loss1+tri_loss2+tri_loss3)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if i_batch % print_freq==0:
                if len(_query_label)==2:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}+{:f}x({:f}+{:f})'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss1,tri_loss2))
                if len(_query_label)==3:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}={:f}+{:f}x({:f}+{:f}+{:f})'.format(e, i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss1,tri_loss2,tri_loss3))    

    
    with torch.no_grad():
        # net.eval()
        # dice_list = []
        # for i_batch, sampled_batch in enumerate(val_loader):
        #     image = sampled_batch[0].unsqueeze_(dim=1).cuda()
        #     label = sampled_batch[1].unsqueeze_(dim=1).cuda()
        #     seg = net(image)
        #     dice = dice_score_binary(seg,label)*100
        #     dice_list.append(dice)
        # val_dc = sum(dice_list)/len(dice_list)
        # print('Epoch {:d} Val dice: {:.1f}'.format(e,val_dc))
        # f = open(txt_path, "a+")
        # f.write('Epoch {:d} Val dice: {:.1f} \n'.format(e,val_dc))
        # f.close()
        save_path = root_save_path +'epoch-{}/'.format(e) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)        
        dice_list = evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=6)
        val_dc = sum(dice_list)/len(dice_list)
        print('Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f}'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f = open(txt_path, "a+")
        f.write('Epoch {:d}, Val dice: {:.1f}|{:.1f}|{:.1f}, avg is {:.1f} \n'.format(e,dice_list[0],dice_list[1],dice_list[2],val_dc))
        f.close()
        if val_dc>best_val_dc:
            best_val_dc = val_dc
            best_e = e
        PATH = model_path + 'epoch-{}.pth'.format(e)
        torch.save({'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': e,
                    'best_DC': best_val_dc}, PATH)
        # if val_dc>best_val_dc:
        #     best_val_dc = val_dc
        #     best_e = e
        #     PATH = model_path + 'best.pth'
        #     torch.save({'state_dict': net.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'epoch': best_e,
        #                 'best_DC': best_val_dc}, PATH)
        # if e%5==0:
        #     PATH = model_path + 'epoch-{}.pth'.format(e)
        #     torch.save({'state_dict': net.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'epoch': e,
        #                 'best_DC': best_val_dc}, PATH)
        # PATH = model_path + 'latest.pth'
        # torch.save({'state_dict': net.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'epoch': e,
        #             'best_DC': best_val_dc}, PATH)
print('Best Epoch {:d} Avg Val dice: {:.1f}'.format(best_e,best_val_dc))
f = open(txt_path, "a+")
f.write('Best Epoch {:d} Avg Val dice: {:.1f} \n'.format(best_e,best_val_dc))
f.close()























































