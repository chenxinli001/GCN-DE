import os

import nibabel as nib
import numpy as np
import torch

#import utils.common_utils as common_utils
#import utils.data_utils as du
import torch.nn.functional as F
import h5py

import SimpleITK as sitk
from Sampler import *

from sklearn import decomposition

def evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,
    test_bs=10,save_img=False,epoch=None):
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

                support_images =  torch.cat([support_batch_x_10[:,0:1],support_batch_x_10[:,0:1],
                    support_batch_x_10[:,0:1]],dim=1 )
                support_fg_mask = support_batch_x_10[:,1:2]
                query_images = torch.cat([query_batch_x_10,query_batch_x_10,query_batch_x_10],dim=1 )
                
                logits = net(query_images, support_images, support_fg_mask, support_fg_mask)

                outB, outA_pos, vec_pos, outB_side = logits
                w, h = query_images.size()[-2:]
                outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
                #out_softmax = F.softmax(outB_side, dim=1).squeeze(0)
                #values, pred = torch.max(out_softmax, dim=0)
    
                #out_softmax, pred = net.get_pred(logits, query_images)
                #batch_output_10 = pred.unsqueeze(0).unsqueeze(0)

                assert outB_side.shape[1] == 1
                batch_output_10 = outB_side>0
    

                # support_images =  [[ torch.cat([support_batch_x_10[:,0:1],support_batch_x_10[:,0:1],
                #     support_batch_x_10[:,0:1]],dim=1 )]]
                # support_fg_mask, support_bg_mask = [[support_batch_x_10[:,1]]], [[1 - support_batch_x_10[:,1]]]
                # query_images = [ torch.cat([query_batch_x_10,query_batch_x_10,query_batch_x_10],dim=1 )]

                # out_10,_ = net(support_images, support_fg_mask, support_bg_mask, query_images)
                # assert out_10.shape[1] ==2
                # batch_output_10 = out_10[:,0:1]<out_10[:,1:2]

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')

        if save_img:
            if epoch:
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_e{}_dice{:.1f}.nii'.format(item,epoch,volume_dice_score))
            else:
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       
    if save_img:
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list
    