# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn

import Sampler
from Sampler import *

from Segmentor import SegMenTor
from Network import *

import torch.optim as optim
import argparse
import cv2

#from nn_common_modules import losses as additional_losses

import os
from evaluator import *

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     #random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(1234)

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default=0)
    parser.add_argument("--dataset", '--d',type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",'--o',type=str,default='liver',help='liver,right kidney,left kidney,spleen')
    parser.add_argument("--run_order",type=str,default=None)
    parser.add_argument("--nlc_layer",type=list,default=[])
    parser.add_argument("--pretrain",action='store_true',default=False)

    #parser.add_argument("--class_cons",action='store_true',default=False)
    #parser.add_argument("--save_best_model",type=bool,default=True)
    # parser.add_argument("--eval_ours",action='store_true',default=False)
    parser.add_argument("--test_vis",action='store_true',default=False)
    parser.add_argument("--save_img",action='store_true',default=False)
    
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

organ_fold_dict = {'liver':'fold1','right_kidney':'fold2','left_kidney':'fold3','spleen':'fold4'}
organ_label_dict =  {"liver": {"train": [2, 3, 4], "val": 1},
                    "right_kidney": {"train": [1, 3, 4], "val": 2},
                    "left_kidney": {"train": [1, 2, 4], "val": 3},
                    "spleen": {"train": [1, 2, 3], "val": 4} 
                    }

DataSet = args.dataset # 'MRI'or'CT'
ObjectOrgan = args.organ

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
    train_bs = 1  
    train_iteration = 1#//train_bs
    val_iteration = 1#//val_bs
    num_epoch = 300
    Num_support = 8 
    model_path = './test/MRI/'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 2
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 25
    Num_support = 8
    model_path = './test/CT/'

model_path = model_path + ObjectOrgan + '/base_'

if args.pretrain:
    model_path = model_path + 'pretrain_'

if args.run_order:
    model_path += '%s'%args.run_order

model_path += '/'

reload_mdoel = 0 
print_freq = 50


print(model_path)
txt_path = model_path + 'result.txt'
if not os.path.exists(model_path):
    os.makedirs(model_path)

root_save_path = model_path + 'nii_save/'
if not os.path.exists(root_save_path):
    os.makedirs(root_save_path)


data = h5py.File(data_path, 'r')
whole_image,whole_label,case_start_index = data['Image'],data['Label'],data['case_start_index']
# assert len(case_start_index)%5==0 


kfold_best_val_dc = []
kfold_best_e = []

kfold_best_val_dc1 = []
kfold_best_e1 = []

print('DataSet: {}, ObjectOrgan: {}'.format(DataSet,ObjectOrgan))

if not args.test_vis:
    f = open(txt_path, "a+")
    f.write('train_bs:{},iter:{}|{}, num_epoch:{}, num_support:{} \n'.format(train_bs,train_iteration,val_iteration,num_epoch,Num_support))
    f.close()

    f = open(txt_path, "a+")
    f.write('DataSet: {}, ObjectOrgan: {} \n'.format(DataSet,ObjectOrgan))
    f.close()

case_start_index = list(case_start_index)
len_kfold = len(case_start_index)//5
case_start_index.append(whole_image.shape[0])

if not args.test_vis:
    for k_fold in range(1):
        net = my_fss(args.pretrain).cuda()

        print('k_fold:{}'.format(k_fold))
        support_item = len_kfold*k_fold
        query_item = list(range(len_kfold*k_fold+1,len_kfold*(k_fold+1)))
        print(support_item,query_item)

        train_start_index = case_start_index[len_kfold*k_fold]
        train_end_index = case_start_index[len_kfold*(k_fold+1)] 
        train_image = np.concatenate([ whole_image[:train_start_index],whole_image[train_end_index:] ],axis=0)
        train_label = np.concatenate([ whole_label[:train_start_index],whole_label[train_end_index:] ],axis=0)

        print(train_image.shape)

        train_dataset = SimpleData(train_image,train_label)
        train_sampler = OneShotBatchSampler(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan], batch_size=train_bs, iteration=train_iteration)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

        optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

        criterion1 = DiceLoss2D()
        criterion2 = nn.BCELoss()

        best_val_dc = 0
        best_e = 0

        best_val_dc1 = 0
        best_e1 = 0

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
            for i_batch, sampled_batch in enumerate(train_loader):
                image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
                label = sampled_batch[1].type(torch.FloatTensor).cuda()
            
                query_label = train_loader.batch_sampler.query_label
                
                support_image, query_image, support_label, query_label = split_batch(image, label, int(query_label))
                condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
                seg = net(condition_input,query_image)
            
                #loss = criterion1(seg, query_label) + criterion2(seg, query_label)
                loss = criterion2(seg, query_label)

                if i_batch % print_freq==0:
                    #real_loss = criterion(seg,query_mask.squeeze(1).long())
                    # sitk.WriteImage(sitk.GetImageFromArray(np.squeeze((seg>0.5).cpu().numpy())), \
                    #     root_save_path + 'savepred_{}_loss{:.3f}.nii'.format(e,loss.item()))    
                    # sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_label.cpu().numpy())), \
                    #     root_save_path + 'savemask_{}_loss{:.3f}.nii'.format(e,loss.item()))
                    dice = Sampler.dice_func(seg,query_label)
                    cv2.imwrite(root_save_path + 'savepred_{}_dice{:.3f}_loss{:.3f}.png'.format(e,dice.item(),loss.item()),
                        np.squeeze(((seg>0.5)*255).cpu().numpy()))    
                    # cv2.imwrite( root_save_path + 'savemask_{}_dice{:.3f}_loss{:.3f}.png'.format(e,dice.item(),loss.item()),
                    #     np.squeeze((query_label*255).cpu().numpy()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i_batch % print_freq==0:
                    print('Epoch {:d} | Episode {:d}/{:d} | Loss {:f}'.format(e, i_batch, len(train_loader), loss))
            
            with torch.no_grad():
                # save_path = root_save_path +'epoch-{}/'.format(e) 
                save_path = root_save_path 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                support_item = len_kfold*k_fold
                query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
                assert len(query_item)+1 == len_kfold

                dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                    query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16,
                    save_img=0,epoch=e)
                val_dc = sum(dice_list)/len(dice_list)

                # dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                #     query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16,
                #     Norm='Slice')
                # val_dc1 = sum(dice_list1)/len(dice_list1)
                val_dc1 = val_dc
                
                print('Epoch {:d}, avg dice: {:.1f}, avg dice1: {:.1f}'.format(e,val_dc,val_dc1))

                if val_dc>best_val_dc:
                    best_val_dc = val_dc
                    best_e = e
                    PATH = model_path + '{}-fold_best.pth'.format(k_fold)
                    torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_e,
                        'best_DC': best_val_dc}, PATH)
                if val_dc1>best_val_dc1:
                    best_val_dc1 = val_dc1
                    best_e1 = e
                    PATH = model_path + '{}-fold_best1.pth'.format(k_fold)
                    torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_e1,
                        'best_DC': best_val_dc1}, PATH)

        
        kfold_best_val_dc.append(best_val_dc)
        kfold_best_e.append(best_e)

        kfold_best_val_dc1.append(best_val_dc1)
        kfold_best_e1.append(best_e1)

        
        print('{}-fold, Best Epoch {:d} Avg Val dice: {:.1f}'.format(k_fold,best_e,best_val_dc))
        f = open(txt_path, "a+")
        f.write('{}-fold, Best Epoch {:d} Avg Val dice: {:.1f} \n'.format(k_fold,best_e,best_val_dc))
        f.close()
        print('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f}'.format(k_fold,best_e1,best_val_dc1))
        f = open(txt_path, "a+")
        f.write('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f} \n'.format(k_fold,best_e1,best_val_dc1))
        f.close()

    print(model_path)
    print('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f}'.format(DataSet,ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),sum(kfold_best_val_dc1)/len(kfold_best_val_dc1)))
    print(kfold_best_e,kfold_best_val_dc)
    print(kfold_best_e1,kfold_best_val_dc1)
    f = open(txt_path, "a+")
    f.write('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.1f}, Avg Best Val dice1: {:.1f} \n'.format(DataSet,ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc),sum(kfold_best_val_dc1)/len(kfold_best_val_dc1)))
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
    f.close()
else:
    total_dc = []
    total_dc1 = []
    total_dc2 = []
    with torch.no_grad():
        for i in range(2):
            for k_fold in range(5):
                net = my_fss(args.pretrain).cuda()
                if i == 0:
                    PATH = model_path + '{}-fold_best.pth'.format(k_fold)
                else:
                    PATH = model_path + '{}-fold_best1.pth'.format(k_fold)
                checkpoint = torch.load(PATH)
                net.load_state_dict(checkpoint['state_dict'])
                #print(k_fold,checkpoint['epoch'],checkpoint['best_DC'])
                save_path = root_save_path + 'origin_eval/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path1 = root_save_path + 'dis_eval/'
                if not os.path.exists(save_path1):
                    os.makedirs(save_path1)
                save_path2 = root_save_path + 'encdis_eval/'
                if not os.path.exists(save_path2):
                    os.makedirs(save_path2)
                support_item = len_kfold*k_fold
                query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
                assert len(query_item)+1 == len_kfold

                dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,
                    query_item,save_path,query_label=organ_label_dict[ObjectOrgan]['val'],
                    Num_support=Num_support,test_bs=16,save_img=args.save_img)
                dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,
                    query_item,save_path1,query_label=organ_label_dict[ObjectOrgan]['val'],
                    Num_support=Num_support,test_bs=16,save_img=args.save_img,Norm='Slice')
                dice_list2 = evaluate_fss_kfold_encoder1a(net,whole_image,whole_label,case_start_index,
                    support_item,query_item,save_path2,query_label=organ_label_dict[ObjectOrgan]['val'],
                    Num_support=Num_support,test_bs=16,save_img=args.save_img)    
                val_dc = sum(dice_list)/len(dice_list)
                val_dc1 = sum(dice_list1)/len(dice_list1)
                val_dc2 = sum(dice_list2)/len(dice_list2)
                total_dc.append(val_dc)
                total_dc1.append(val_dc1)
                total_dc2.append(val_dc2)
                #print(k_fold,val_dc,val_dc1,val_dc2)
            print('{}: dc={:.1f},dc1={:.1f},dc2={:.1f}'.format(model_path,sum(total_dc)/len(total_dc),
                sum(total_dc1)/len(total_dc1),sum(total_dc2)/len(total_dc2)))