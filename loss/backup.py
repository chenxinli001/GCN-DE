# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import h5py

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from Sampler import *

from Segmentor import SegMenTor
from Network import *

import torch.optim as optim
import argparse

from Network_correct import *

#from nn_common_modules import losses as additional_losses

import os
from evaluator import *

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default='0')
    parser.add_argument("--dataset", type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",type=str,default='liver',help='liver,right kidney,left kidney,spleen')
    parser.add_argument("--run_order",type=str,default=None)
    parser.add_argument("--pretrain",action='store_true',default=True)
    parser.add_argument("--t",type=float,default=1)
    parser.add_argument("--fuse_type",type=int,default=None)
    parser.add_argument("--test_vis",action='store_true',default=False)

    return parser.parse_args()

# def distance(f1,f2):
#     bs= f1.size()[0]
#     f1 = f1.view(bs,-1)
#     f2 = f2.view(bs,-1)
#     distance = torch.sqrt( ((f1-f2)**2).sum(dim=-1) ) #bs
#     return distance.sum()/bs#*distance

# def L2_norm(x):
#     bs = x.size()[0]
#     x = x.view(bs,-1)
#     norm = torch.sqrt( (x**2).sum(dim=-1) )  #bs
#     return norm.sum()/bs

def distance(f1,f2):
    bs= f1.size()[0]
    f1 = f1.view(bs,-1)
    f2 = f2.view(bs,-1)
    # distance = torch.sqrt( ((f1-f2)**2).sum(dim=-1) ).mean() #bs
    distance = F.pairwise_distance(f1,f2,p=2)
    return distance#*distance

def L2_norm(x):
    bs = x.size()[0]
    x = x.view(bs,-1)
    norm = torch.sqrt( (x**2).sum(dim=-1) ).mean()  #bs
    return norm


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

if DataSet == 'MRI':
    data_path = './datasets/MRI/MRIWholeData.h5'
    train_bs = 2  
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 15
    Num_support = 8
    # lambda_t = 50
    lambda_t = args.t
    model_path = './result_correct4/MRI/'
    #encoder_path = './result_correct4/pretrain_ae/CT/ae_epoch-100.pth'

elif DataSet == 'CT':
    data_path = './datasets/CT/CTWholeData.h5'
    train_bs = 2
    train_iteration = 25#//train_bs
    val_iteration = 25#//val_bs
    num_epoch = 25
    Num_support = 8 #原始测量
    # lambda_t = 25
    lambda_t = args.t
    model_path = './result_correct4/CT/'
    #encoder_path = './result_pretrain/pretrain_ae/CT/ae_epoch-90.pth'

model_path = model_path + ObjectOrgan + '/cons3_t{}'.format(lambda_t) 

if args.pretrain:
    model_path = model_path + '_pretrain'

if args.run_order:
    model_path += '_%s'%args.run_order

model_path += '/'

reload_mdoel = 0 
print_freq = 50

MEAN = False  
# distance = l1_distance

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
# assert len(case_start_index)%5==0 

kfold_best_val_dc = []
kfold_best_e = []

kfold_best_val_dc1 = []
kfold_best_e1 = []

kfold_best_val_dc2 = []
kfold_best_e2 = []

print('DataSet: {}, ObjectOrgan: {}'.format(DataSet,ObjectOrgan))
f = open(txt_path, "a+")
f.write('DataSet: {}, ObjectOrgan: {} \n'.format(DataSet,ObjectOrgan))
f.close()

case_start_index = list(case_start_index)
len_kfold = len(case_start_index)//5
case_start_index.append(whole_image.shape[0])

if not args.test_vis:
    for k_fold in range(5):
        net = fss_lastfea(args.pretrain).cuda()
        with torch.no_grad():
            initial_net = fss_lastfea(args.pretrain).cuda()
            for initial_param, param in zip(initial_net.parameters(), net.parameters()):
                initial_param.data.mul_(0).add_(1, param.data)

        print('k_fold:{}'.format(k_fold))
        support_item = len_kfold*k_fold
        query_item = list(range(len_kfold*k_fold+1,len_kfold*(k_fold+1)))
        print(support_item,query_item)

        train_start_index = case_start_index[len_kfold*k_fold]
        train_end_index = case_start_index[len_kfold*(k_fold+1)] 
        train_image = np.concatenate([ whole_image[:train_start_index],whole_image[train_end_index:] ],axis=0)
        train_label = np.concatenate([ whole_label[:train_start_index],whole_label[train_end_index:] ],axis=0)

        print(train_image.shape,train_label.shape)

        train_dataset = SimpleData(train_image,train_label)
        train_sampler = OneShotBatchSampler_ms4(train_dataset.label, 'train', organ_fold_dict[ObjectOrgan], batch_size=train_bs, iteration=train_iteration)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

        bs = train_bs
        optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.99, weight_decay=1e-4)

        criterion1 = DiceLoss2D()
        criterion2 = nn.BCELoss()

        best_val_dc = 0
        best_e = 0
        best_val_dc1 = 0
        best_e1 = 0
        best_val_dc2 = 0
        best_e2 = 0
        start_epoch = 0

        for e in range(start_epoch,num_epoch+1):
            net.train()
            for i_batch, sampled_batch in enumerate(train_loader):
                image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
                label = sampled_batch[1].type(torch.FloatTensor).cuda()
            
                _query_label = train_loader.batch_sampler.query_label
            
                support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)
                condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)

                if len(_query_label) == 1:
                    seg,_,_,_ = net(condition_input,query_image)
                elif len(_query_label) == 2:
                    class_for_seg = np.random.choice([0,1])
                    condition_input_for_seg = torch.split(condition_input,bs)[class_for_seg]
                    query_image_for_seg = torch.split(query_image,bs)[class_for_seg]
                    seg,sf_for_seg,qf_for_seg,qwf_for_seg = net(condition_input_for_seg,query_image_for_seg)
                    query_label_for_seg = torch.split(query_label,bs)[class_for_seg]
                    condition_input_for_ref = torch.split(condition_input,bs)[1-class_for_seg]
                    sf_for_ref = net.conditioner(condition_input_for_ref)[-1]
                    with torch.no_grad():
                        _,sf_initial,qf_initial,qwf_initial = initial_net(condition_input,query_image)
                        qf_initial_for_seg = torch.split(qf_initial,bs)[class_for_seg]
                        qwf_initial_for_seg = torch.split(qwf_initial,bs)[class_for_seg]
                elif len(_query_label) == 3:
                    class_for_seg = np.random.choice([0,1,2])
                    class_list = [0,1,2]
                    class_list.remove(class_for_seg)
                    condition_input_for_seg = torch.split(condition_input,bs)[class_for_seg]
                    query_image_for_seg = torch.split(query_image,bs)[class_for_seg]
                    seg,sf_for_seg,qf_for_seg,qwf_for_seg = net(condition_input_for_seg,query_image_for_seg)
                    query_label_for_seg = torch.split(query_label,bs)[class_for_seg]
                    condition_input_for_ref = []
                    condition_input_for_ref.append(torch.split(condition_input,bs)[class_list[0]])
                    condition_input_for_ref.append(torch.split(condition_input,bs)[class_list[1]])
                    condition_input_for_ref = torch.cat(condition_input_for_ref)
                    sf_for_ref = net.conditioner(condition_input_for_ref)[-1]
                    with torch.no_grad():
                        _,sf_initial,qf_initial,qwf_initial = initial_net(condition_input,query_image)
                        qf_initial_for_seg = torch.split(qf_initial,bs)[class_for_seg]
                        qwf_initial_for_seg = torch.split(qwf_initial,bs)[class_for_seg]

                assert seg.max()<=1 and seg.min()>=0

                if len(_query_label)==1:
                    seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)
                    loss = seg_loss
                else:
                    if MEAN:
                        sf_for_seg = torch.mean(sf_for_seg,dim=1)
                        qf_for_seg = torch.mean(qf_for_seg,dim=1)
                        qwf_for_seg = torch.mean(qwf_for_seg,dim=1)
                        sf_for_ref = torch.mean(sf_for_ref,dim=1)
                        sf_initial = torch.mean(sf_initial,dim=1)
                        qf_initial_for_seg = torch.mean(qf_initial_for_seg,dim=1)
                        qwf_initial_for_seg = torch.mean(qwf_initial_for_seg,dim=1)
                    
                    seg_loss = criterion1(seg, query_label_for_seg) + criterion2(seg.squeeze(dim=1), query_label_for_seg)

                    if len(_query_label)==2:
                        anchor = qwf_for_seg
                        pos = sf_for_seg
                        neg = sf_for_ref
                        tri_loss = distance(anchor,pos) - distance(anchor,neg) + \
                            0.5*L2_norm(qwf_initial_for_seg) + 0.5*L2_norm(sf_initial)
                        tri_loss = max(0,tri_loss)
                        loss = seg_loss + lambda_t*tri_loss
                    elif len(_query_label)==3:
                        anchor = qwf_for_seg
                        pos = sf_for_seg
                        neg1,neg2 = torch.split(sf_for_ref,bs)
                        initial_concat = torch.cat([qwf_initial_for_seg,sf_initial],dim=0)
                        tri_loss = distance(anchor,pos) - distance(anchor,neg1)/2 - distance(anchor,neg2)/2 + \
                            0.5*L2_norm(qwf_initial_for_seg) + 0.5*L2_norm(sf_initial)
                        tri_loss = max(0,tri_loss)
                        loss = seg_loss + lambda_t*tri_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i_batch % print_freq == 0:
                    if len(_query_label)==1:
                        print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}'.format(e, i_batch, len(train_loader), loss,seg_loss))
                    elif len(_query_label)==2:
                        print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}+{:.3f}x{:.3f}'.format(e, \
                            i_batch, len(train_loader), loss,seg_loss,lambda_t,tri_loss))
                # if len(_query_label)==1:
                #     print('Epoch {:d} | Episode {:d}/{:d} | Loss {:.3f}={:.3f}'.format(e, i_batch, len(train_loader), 
                #         loss,seg_loss))
                # elif len(_query_label)==2:
                #     print('Epoch {:d} | Episode {:d}/{:d} | SEG {:.3f}, TRI {:.3f} '.format(e, \
                #         i_batch, len(train_loader), seg_loss,tri_loss))
                # elif len(_query_label)==3:
                #     print('Epoch {:d} | Episode {:d}/{:d} | SEG {:.3f}, TRI {:.3f} '.format(e, \
                #         i_batch, len(train_loader), seg_loss,tri_loss))
              
        
            with torch.no_grad():
                # save_path = root_save_path +'epoch-{}/'.format(e) 
                save_path = root_save_path 
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                support_item = len_kfold*k_fold
                query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
                assert len(query_item)+1 == len_kfold

                # if args.cnlc:
                #     state_dict = net.state_dict()
                #     eval_net = my_fss_fea_nlc(args.pretrain,nlc_layer,sub_sample,bn_layer,shortcut).cuda()
                #     eval_net.load_state_dict(state_dict)

                #     dice_list = evaluate_fss_kfold(eval_net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                #         query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=10)
                #     val_dc = sum(dice_list)/len(dice_list)
                #     dice_list1 = evaluate_fss_kfold_encoder1a(eval_net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                #         query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=10)
                #     val_dc1 = sum(dice_list1)/len(dice_list1)
                # else:
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
                print('Epoch {:d}, avg dice: {:.1f}, avg dice1: {:.1f}'.format(e,val_dc,val_dc1))


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

else:
    total_dc = []
    total_dc1 = []
    with torch.no_grad():
        for k_fold in range(5):
            net = my_fss_fea_nlc(args.pretrain,nlc_layer,sub_sample,bn_layer,shortcut).cuda()
            PATH = model_path + '{}-fold_best.pth'.format(k_fold)
            checkpoint = torch.load(PATH)
            net.load_state_dict(checkpoint['state_dict'])
            print(k_fold,checkpoint['epoch'],checkpoint['best_DC'])
            save_path1 = root_save_path + 'origin_eval/'
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            save_path2 = root_save_path + 'encoder_eval/'
            if not os.path.exists(save_path2):
                os.makedirs(save_path2)
            support_item = len_kfold*k_fold
            query_item = list(range(len(case_start_index)//5*k_fold+1,len(case_start_index)//5*(k_fold+1)))
            assert len(query_item)+1 == len_kfold

            dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path1,
                query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=8,save_img=True)
            dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path2,
                query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=8,save_img=True)    
            val_dc = sum(dice_list)/len(dice_list)
            val_dc1 = sum(dice_list1)/len(dice_list1)
            total_dc.append(val_dc)
            total_dc1.append(val_dc1)
            print(k_fold,val_dc,val_dc1)
        print(model_path,sum(total_dc)/len(total_dc),sum(total_dc1)/len(total_dc1))

