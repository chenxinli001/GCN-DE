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
from Networks.Network_correct import *

import torch.optim as optim
import argparse

#from nn_common_modules import losses as additional_losses

import os
from loss.evaluator import *

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
    parser.add_argument("--dataset", '--d',type=str,default='CT',help='MRI,CT')
    parser.add_argument("--organ",'--o',type=str,default='left_kidney',help='liver,right_kidney,left_kidney,spleen')
    parser.add_argument("--run_order",type=str,default=None)
    parser.add_argument("--pretrain",action='store_true',default=True)
    parser.add_argument("--test_vis",action='store_true',default=True)
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
    train_bs = 2  
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 15
    Num_support = 8 
    model_path = './result_paper/MRI/'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 2
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 25
    Num_support = 8
    model_path = './result_paper/CT/'

model_path = model_path + ObjectOrgan + '/pretext'

if args.pretrain:
    model_path = model_path + '_pretrain'

if args.run_order:
    model_path += '_%s'%args.run_order

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
    for k_fold in range(5):
        # net = my_fss(args.pretrain).cuda()
        net = pretext_segmentor(args.pretrain).cuda()

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
                #condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
                #seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)
                seg = net(query_image)

                loss = criterion1(seg, query_label) + criterion2(seg, query_label)

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

                dice_list = evaluate_pretext_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                    query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16)
                val_dc = sum(dice_list)/len(dice_list)

                # dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16,
                #     Norm='Slice')
                # val_dc1 = sum(dice_list1)/len(dice_list1)
                
                print('Epoch {:d}, avg dice: {:.2f}'.format(e,val_dc))

                if val_dc>best_val_dc:
                    best_val_dc = val_dc
                    best_e = e
                    PATH = model_path + '{}-fold_best.pth'.format(k_fold)
                    torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': best_e,
                        'best_DC': best_val_dc}, PATH)
                # if val_dc1>best_val_dc1:
                #     best_val_dc1 = val_dc1
                #     best_e1 = e
                #     PATH = model_path + '{}-fold_best1.pth'.format(k_fold)
                #     torch.save({'state_dict': net.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'epoch': best_e1,
                #         'best_DC': best_val_dc1}, PATH)

        
        kfold_best_val_dc.append(best_val_dc)
        kfold_best_e.append(best_e)

        # kfold_best_val_dc1.append(best_val_dc1)
        # kfold_best_e1.append(best_e1)

        
        print('{}-fold, Best Epoch {:d} Avg Val dice: {:.2f}'.format(k_fold,best_e,best_val_dc))
        f = open(txt_path, "a+")
        f.write('{}-fold, Best Epoch {:d} Avg Val dice: {:.2f} \n'.format(k_fold,best_e,best_val_dc))
        f.close()
        # print('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f}'.format(k_fold,best_e1,best_val_dc1))
        # f = open(txt_path, "a+")
        # f.write('{}-fold, Best Epoch {:d} Avg Val dice1: {:.1f} \n'.format(k_fold,best_e1,best_val_dc1))
        # f.close()

    print(model_path)
    print('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.2f}'.format(DataSet,ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc)))
    print(kfold_best_e,kfold_best_val_dc)
    # print(kfold_best_e1,kfold_best_val_dc1)
    f = open(txt_path, "a+")
    f.write('DataSet:{}, ObjectOrgan:{}, Avg Best Val dice: {:.2f} \n'.format(DataSet,ObjectOrgan,sum(kfold_best_val_dc)/len(kfold_best_val_dc)))
    for f_in in range(len(kfold_best_e)):
        f.write(str(kfold_best_e[f_in]) + ' ')
    f.write(', ')
    for f_in in range(len(kfold_best_val_dc)):
        f.write(str(kfold_best_val_dc[f_in]) + ' ')
    f.write('\n')
    # for f_in in range(len(kfold_best_e1)):
    #     f.write(str(kfold_best_e1[f_in]) + ' ')
    # f.write(', ')
    # for f_in in range(len(kfold_best_val_dc1)):
    #     f.write(str(kfold_best_val_dc1[f_in]) + ' ')
    # f.write('\n')
    f.close()
else:
    total_dc = []
    for k_fold in range(5):
        net = pretext_segmentor(args.pretrain).cuda()
        PATH = model_path + '{}-fold_best.pth'.format(k_fold)
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['state_dict'])
        support_item = len_kfold*k_fold
        query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
        assert len(query_item)+1 == len_kfold
        save_path = root_save_path 
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dice_list = finetune_pretext_kfold(net,whole_image,whole_label,case_start_index,support_item,
            query_item,save_path,query_label=organ_label_dict[ObjectOrgan]['val'],
            Num_support=Num_support,test_bs=16,save_img=args.save_img) 
        val_dc = sum(dice_list)/len(dice_list)
        total_dc.append(val_dc)
        print(k_fold, val_dc)
    print('{}: mean_dc={:.2f}'.format(model_path,sum(total_dc)/len(total_dc)))

