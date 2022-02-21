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

# from Segmentor import SegMenTor
# from Network import *
# from Network_correct import *

import torch.optim as optim
import argparse

#from nn_common_modules import losses as additional_losses

import os
# from evaluator import *


net_params = {'num_class':1,'num_channels': 1,'num_filters': 64,'kernel_h': 5,'kernel_w': 5,'kernel_c': 1,'stride_conv': 1,'pool': 2,'stride_pool': 2,'se_block': "NONE" #Valid options : NONE, CSE, SSE, CSSE
,'drop_out': 0}

def evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=8,
    test_bs=10,save_img=False):
    net.eval()
    #选定support,query
    # support = h5py.File(support_file, 'r')
    # support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]
    
    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    support_slice_indexes = support_slice_indexes[:-1]
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    volume_dice_score_list = []
    
    for item in query_item:
        # query = h5py.File(query_file, 'r')
        # query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]

        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]
        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)
        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
           
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
                support_batch_x_10 = torch.FloatTensor(support_batch_x_10).cuda()
                
                try:
                    out_10 = net(support_batch_x_10,query_batch_x_10)
                    batch_output_10 = out_10>0.5
                except:
                    try:
                        out_10,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                        batch_output_10 = out_10>0.5
                    except:
                        out_10,_,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                        batch_output_10 = out_10>0.5


                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')

        # if save_img:
        #     sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        #     sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))
        #     sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       
    # if save_img:
    #     sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    #     sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list
    


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str,default=0)
    parser.add_argument("--dataset", '--d',type=str,default='MRI',help='MRI,CT')
    parser.add_argument("--organ",'--o',type=str,default='liver',help='liver,right kidney,left kidney,spleen')
    parser.add_argument("--run_order",type=str,default=None)
    parser.add_argument("--nlc_layer",type=list,default=[])
    parser.add_argument("--pretrain",action='store_true',default=False)
    parser.add_argument("--SE_type",type=int,default=None)
    parser.add_argument("--method", '--m',type=str,default='Shaban',choices=['MIA','Shaban','Rakelly'])
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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

model_path = model_path + ObjectOrgan + '/{}'.format(args.method)

# if args.pretrain:
#     model_path = model_path + '_pretrain'

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
        if args.method == 'MIA':
            import MIA.few_shot_segmentor as fs
        elif args.method == 'Shaban':
            import MIA.other_experiments.shaban.few_shot_segmentor_shaban_baseline as fs
        elif args.method == 'Rakelly':
            import MIA.other_experiments.rakelly.few_shot_segmentor_feature_fusion_baseline as fs
        net = fs.FewShotSegmentorDoubleSDnet(net_params).cuda()



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
                seg= net(condition_input,query_image)
            
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

                dice_list = evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,
                    query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16)
                val_dc = sum(dice_list)/len(dice_list)

                # dice_list1 = evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label=organ_label_dict[ObjectOrgan]['val'],Num_support=Num_support,test_bs=16,
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


