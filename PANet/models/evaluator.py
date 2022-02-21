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

def evaluate_fss_finetune(net,support_file,query_path,save_path,query_label=1,Num_support=6):
    #net.eval()
    #选定support,query
    _query_label = query_label

    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==_query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    support_slice_indexes = support_slice_indexes[:-1]
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)
    
    lr_finetune_decay = 0.1
    net.train()  
    optimizer = optim.SGD(net.parameters(), lr=1e-2*lr_finetune_decay,momentum=0.99, weight_decay=1e-4)
    criterion1 = DiceLoss2D()
    criterion2 = nn.BCELoss()

    # image = sampled_batch[0].unsqueeze_(dim=1).type(torch.FloatTensor).cuda()
    # label = sampled_batch[1].type(torch.FloatTensor).cuda()
    #support_slice_indexes = np.asarray(support_slice_indexes)
    #image = support_volume[support_slice_indexes]
    #label = support_labelmap[support_slice_indexes]
    
    # _query_label = train_loader.batch_sampler.query_label
    #support_image, query_image, support_label, query_label = split_batch_ms4(image,label,_query_label)

    support_index_lst = []
    query_index_lst = []
    for slice_index in support_slice_indexes:
        support_index_lst.extend([slice_index]*len(support_slice_indexes))
        query_index_lst.extend(support_slice_indexes)
    assert len(support_index_lst) == len(query_index_lst)

    support_volume_tensor = torch.from_numpy(support_volume)
    support_labelmap_tensor = torch.from_numpy(support_labelmap)
    for train_iter in range(len(support_index_lst)):
        support_image = support_volume_tensor[support_index_lst[train_iter]].type(torch.FloatTensor).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
        support_label = support_labelmap_tensor[support_index_lst[train_iter]].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
        query_image = support_volume_tensor[query_index_lst[train_iter]].type(torch.FloatTensor).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
        query_label = support_labelmap_tensor[query_index_lst[train_iter]].type(torch.FloatTensor).cuda().unsqueeze(dim=0)
    
        condition_input = torch.cat([support_image,support_label.unsqueeze(dim=1)],dim=1)
        try:
            seg,s_feature,q_feature,qw_feature = net(condition_input,query_image)
            seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)
        except:
            seg = net(condition_input,query_image)
            seg_loss = criterion1(seg, query_label) + criterion2(seg.squeeze(dim=1), query_label)

        optimizer.zero_grad()
        seg_loss.backward(retain_graph=True)
        optimizer.step() 

    with torch.no_grad():
        net.eval()

        volume_dice_score_list = []

        for query_file in query_path:
            query = h5py.File(query_file, 'r')
            query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
            query_labelmap = np.asarray(np.where(query_labelmap==_query_label,1,0))
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
                for b in range(0, len(query_batch_x), 10):
                    query_batch_x_10 = query_batch_x[b:b + 10]
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

                    #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                    #batch_output_10 = out_10

                    volume_prediction_10.append(batch_output_10)
                volume_prediction.extend(volume_prediction_10)
            volume_prediction = torch.cat(volume_prediction)
            volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

            volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
            #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
            
            volume_dice_score = volume_dice_score.item()
            volume_dice_score_list.append(volume_dice_score)
            #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
        #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
        return volume_dice_score_list

def evaluate_fss_ours1(net,support_file,query_path,save_path,query_label=1,Num_support=6,n_components=6):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    ### S和Q分别进行PCA
    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    # norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    # slice_norm_support_volume = np.zeros_like(support_volume)
    # for i in range(len(support_volume)):
    #     slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    support_slice_indexes = support_slice_indexes[:-1]
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]
        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        cen_query_slice_indexes = query_slice_indexes + (len(query_volume) // Num_support) // 2 #center
        cen_query_slice_indexes = cen_query_slice_indexes[:-1]

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        if len(cen_query_slice_indexes) < Num_support:
            # cen_query_slice_indexes.append(len(query_volume) - 1)
            cen_query_slice_indexes = np.concatenate([cen_query_slice_indexes,np.asarray(len(query_volume)-1)[np.newaxis]])
        assert len(support_slice_indexes) == len(cen_query_slice_indexes) ==len(query_slice_indexes)

        # norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        # slice_norm_query_volume = np.zeros_like(query_volume)
        # for i in range(len(query_volume)):
        #     slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

       
        # refine_support_slice_indexes = []
        # for cen_query_slice_index in cen_query_slice_indexes:
        #     shortest_dis = 9e9
        #     match_slice_index = -1
        #     for support_slice_index in support_slice_indexes:  
        #         dis = np.abs(query_volume[cen_query_slice_index] - support_volume[support_slice_index]).mean()
        #         # pca_q = pca.fit_transform(query_volume[cen_query_slice_index].reshape(1,-1))
        #         # pca_s = pca.fit_transform(support_volume[support_slice_index].reshape(1,-1))
        #         #print(query_volume[cen_query_slice_index].shape,pca_q.shape,pca_s.shape)
        #         #dis = np.abs(pca_q-pca_s).mean()
        #         if dis<shortest_dis:
        #             shortest_dis = dis
        #             match_slice_index = support_slice_index
        #     refine_support_slice_indexes.append(match_slice_index)
        pca = decomposition.PCA(n_components=n_components)

        query_slices = query_volume[cen_query_slice_indexes].reshape(len(cen_query_slice_indexes),-1)
        support_slices = support_volume[support_slice_indexes].reshape(len(support_slice_indexes),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        query_volume_pca = pca.fit_transform(query_slices)
        support_volume_pca = pca.fit_transform(support_slices)
    
        # print(query_volume_pca.shape,support_volume_pca.shape)

        # concat_slices = np.concatenate([query_slices,support_slices],axis=0)
        # concat_slices_pca = pca.fit_transform(concat_slices)
        # query_volume_pca = concat_slices_pca[:len(query_slices)]
        # support_volume_pca = concat_slices_pca[len(query_slices):]
        #print(query_volume_pca.shape,support_volume_pca.shape)

        refine_support_slice_indexes = []
        for i in range(len(query_volume_pca)):
            shortest_dis = 9e9
            match_slice_index = -1
            for j in range(len(support_volume_pca)):  
                dis = np.abs(query_volume_pca[i] - support_volume_pca[j]).mean()
                # pca_q = pca.fit_transform(query_volume[cen_query_slice_index].reshape(1,-1))
                # pca_s = pca.fit_transform(support_volume[support_slice_index].reshape(1,-1))
                #print(query_volume[cen_query_slice_index].shape,pca_q.shape,pca_s.shape)
                #dis = np.abs(pca_q-pca_s).mean()
                if dis<shortest_dis:
                    shortest_dis = dis
                    match_slice_index = support_slice_indexes[j]
            refine_support_slice_indexes.append(match_slice_index)
        
        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in support_slice_indexes],[i + 1 for i in refine_support_slice_indexes])
        #print(len(support_volume),len(query_volume))
                

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[refine_support_slice_indexes[i]:1+refine_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours1a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None'):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    ### S和Q分别进行PCA 与位置信息结合，就是说S和Q按区域对应，但具体该Q区域内的每张切片对应S区域内的哪张切片，用PCA计算出来

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
        
        if Norm=='None':
            query_slices = query_volume.reshape(len(query_volume),-1)
            support_slices = support_volume.reshape(len(support_volume),-1)
        elif Norm == 'Case':
            query_slices = norm_query_volume.reshape(len(query_volume),-1)
            support_slices = norm_support_volume.reshape(len(support_volume),-1)
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume.reshape(len(query_volume),-1)
            support_slices = slice_norm_support_volume.reshape(len(support_volume),-1)

        #print(query_slices.shape,support_slices.shape)

        query_volume_pca = pca.fit_transform(query_slices)
        support_volume_pca = pca.fit_transform(support_slices)
    
        #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support
        
        #print(query_file)
        #print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        #print(len(support_volume),len(query_volume))
                

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours1b(net,support_file,query_path,save_path,query_label=1,Num_support=6,test_bs=5,Norm='None'):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    ### S和Q分别进行PCA 与位置信息结合，就是说S和Q按区域对应，但具体该Q区域内的每张切片对应S区域内的哪张切片，用PCA计算出来
    ### 欧式距离与位置信息结合

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        # pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
    
        # query_slices = query_volume.reshape(len(query_volume),-1)
        # support_slices = support_volume.reshape(len(support_volume),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        # query_volume_pca = pca.fit_transform(query_slices)
        # support_volume_pca = pca.fit_transform(support_slices)
    
        # #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)
        # query_slices = query_volume
        # support_slices = support_volume

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours1b_w(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None'):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    ### S和Q分别进行PCA 与位置信息结合，就是说S和Q按区域对应，但具体该Q区域内的每张切片对应S区域内的哪张切片，用PCA计算出来
    ### 欧式距离与位置信息结合
    ### 多切片以加权形式共同指导

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        # pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
    
        # query_slices = query_volume.reshape(len(query_volume),-1)
        # support_slices = support_volume.reshape(len(support_volume),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        # query_volume_pca = pca.fit_transform(query_slices)
        # support_volume_pca = pca.fit_transform(support_slices)
    
        # #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        # query_slices = query_volume
        # support_slices = support_volume

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        # total_match_support_slice_indexes = []
        # for i in range(len(query_slice_indexes)):
        #     match_support_slice_indexes = []
        #     if i==len(query_slice_indexes)-1:
        #         for j in range(query_slice_indexes[i],len(query_slices)):
        #             s_dis = 9e9
        #             match_slice_index = -1
        #             for k in range(support_slice_indexes[i],len(support_slices)):
        #                 #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
        #                 dis = np.abs(query_slices[j] - support_slices[k]).mean()
        #                 if dis<s_dis:
        #                     s_dis = dis
        #                     match_slice_index = k
        #             match_support_slice_indexes.append(match_slice_index)
        #     else:        
        #         for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
        #             s_dis = 9e9
        #             match_slice_index = -1
        #             for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
        #                 # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
        #                 dis = np.abs(query_slices[j] - support_slices[k]).mean()
        #                 if dis<s_dis:
        #                     s_dis = dis
        #                     match_slice_index = k
        #             match_support_slice_indexes.append(match_slice_index)
        #     total_match_support_slice_indexes.append(match_support_slice_indexes)
        # assert len(total_match_support_slice_indexes) == Num_support
        
        match_support_slice_indexes = []
        match_support_slice_dis = []
        for i in range(len(query_slice_indexes)):
            dis_lst = []
            index_lst = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    dis = []
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        dis.append(np.abs(query_slices[j] - support_slices[k]).mean())
                    dis_sorted = np.sort(np.asarray(dis))
                    index_sorted = np.arange(support_slice_indexes[i],len(support_slices))
                    index_sorted = index_sorted[np.argsort(np.asarray(dis))]
                    dis_lst.append(dis_sorted)
                    index_lst.append(index_sorted)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    dis = []
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        dis.append(np.abs(query_slices[j] - support_slices[k]).mean())
                    dis_sorted = np.sort(np.asarray(dis))
                    index_sorted = np.arange(support_slice_indexes[i],support_slice_indexes[i+1])
                    index_sorted = index_sorted[np.argsort(np.asarray(dis))]
                    dis_lst.append(dis_sorted)
                    index_lst.append(index_sorted)
            match_support_slice_indexes.append(index_lst)
            match_support_slice_dis.append(dis_lst)
        assert len(match_support_slice_indexes) == len(match_support_slice_dis) == Num_support
        #print(match_support_slice_indexes)
        #print(match_support_slice_dis)

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))
        #print(query_file)
        #print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i for j in match_support_slice_indexes for i in j for k in i])
        #print(len(support_volume),len(query_volume))


        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x_lst = [ np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[match_support_slice_indexes[i][0]] ]
            for l in range(1,len(match_support_slice_indexes[i])):
                support_batch_x_lst.append(np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[match_support_slice_indexes[i][l]])
            dis_batch_lst = match_support_slice_dis[i]

            #print(len(support_batch_x_lst))
            #print(support_batch_x_lst[0].shape)

            volume_prediction_10 = []
            for b in range(0, len(query_batch_x)):
                query_batch_x_10 = query_batch_x[b:b + 1]
                query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
                pred_ens = []
                for s in range(0,len(support_batch_x_lst[b])):
                    support_batch_x_10 = support_batch_x_lst[b][s:s+1]
                    support_batch_x_10 = torch.FloatTensor(support_batch_x_10).cuda()
                    #print(support_batch_x_10.shape,query_batch_x_10.shape)
                    out_10,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                    pred_ens.append(out_10)
                #print(dis_batch_lst[b].shape)
                weight = F.softmax(1/torch.FloatTensor(dis_batch_lst[b]))
                #print(dis_batch_lst[b],weight)
                pred_ens_w = [pred_ens[g]*weight[g]>0.5 for g in range(len(support_batch_x_lst[b]))]
                #print(len(pred_ens_w))
                volume_prediction_10.append(sum(pred_ens_w))
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list


def evaluate_fss_ours1_ensemble(net,support_file,query_path,save_path,query_label=1,Num_support=6):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

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

    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]
        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        cen_query_slice_indexes = query_slice_indexes + (len(query_volume) // Num_support) // 2 #center
        cen_query_slice_indexes = cen_query_slice_indexes[:-1]

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        if len(cen_query_slice_indexes) < Num_support:
            # cen_query_slice_indexes.append(len(query_volume) - 1)
            cen_query_slice_indexes = np.concatenate([cen_query_slice_indexes,np.asarray(len(query_volume)-1)[np.newaxis]])
        assert len(support_slice_indexes) == len(cen_query_slice_indexes) ==len(query_slice_indexes)

        match1_support_slice_indexes = []
        match2_support_slice_indexes = []
        for cen_query_slice_index in cen_query_slice_indexes:
            s1_dis = 9e9
            match1_slice_index = -1
            s2_dis = 9e9
            match2_slice_index = -1
            for support_slice_index in support_slice_indexes:  
                dis = np.abs(query_volume[cen_query_slice_index] - support_volume[support_slice_index]).mean()
                if match1_slice_index == -1 and match2_slice_index == -1:
                    s1_dis = dis
                    s2_dis = dis
                    match1_slice_index = support_slice_index
                    match2_slice_index = support_slice_index
                else:
                    if dis<s2_dis and dis>=s1_dis:
                        s2_dis = dis
                        match2_slice_index = support_slice_index
                    elif dis<s1_dis:
                        s1_dis = dis
                        match1_slice_index = support_slice_index
            match1_support_slice_indexes.append(match1_slice_index)
            match2_support_slice_indexes.append(match2_slice_index)
        
        # print(query_file)
        # print(query_slice_indexes,support_slice_indexes,match1_support_slice_indexes,match2_support_slice_indexes)
                
        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
                support_batch_x_10 = torch.FloatTensor(support_batch_x_10).cuda()
                
                out_10,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)

        volume_prediction2 = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[match1_support_slice_indexes[i]:1+match1_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
                support_batch_x_10 = torch.FloatTensor(support_batch_x_10).cuda()
                
                out_10,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction2.extend(volume_prediction_10)

        volume_prediction3 = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[match2_support_slice_indexes[i]:1+match2_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
                support_batch_x_10 = torch.FloatTensor(support_batch_x_10).cuda()
                
                out_10,_,_,_ = net(support_batch_x_10,query_batch_x_10)
                batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction3.extend(volume_prediction_10)


        volume_prediction = torch.cat(volume_prediction)
        volume_prediction2 = torch.cat(volume_prediction2)
        volume_prediction3 = torch.cat(volume_prediction3)

        volume_prediction = (volume_prediction + volume_prediction2 + volume_prediction3)/3
        volume_prediction = volume_prediction > 0.5

        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list
        
def evaluate_fss_ours2(net,support_file,query_path,save_path,query_label=1,Num_support=6):
    ### 精细化support slices的选择
    ### ours2 逐片地寻找距离最近的切片对
    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    # support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    # support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    # support_slice_indexes = support_slice_indexes[:-1]
    # if len(support_slice_indexes) < Num_support:
    #     support_slice_indexes.append(len(support_volume) - 1)
    support_slice_indexes = list(range(0,len(support_volume)))

    volume_dice_score_list = []

    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]
        #query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        #cen_query_slice_indexes = query_slice_indexes + (len(query_volume) // Num_support) // 2 #center
        #cen_query_slice_indexes = cen_query_slice_indexes[:-1]

        # if len(query_slice_indexes) < Num_support:
        #     query_slice_indexes.append(len(query_volume) - 1)

        # if len(cen_query_slice_indexes) < Num_support:
        #     # cen_query_slice_indexes.append(len(query_volume) - 1)
        #     cen_query_slice_indexes = np.concatenate([cen_query_slice_indexes,np.asarray(len(query_volume)-1)[np.newaxis]])
        # assert len(support_slice_indexes) == len(cen_query_slice_indexes) ==len(query_slice_indexes)
    
        # refine_support_slice_indexes = []
        # for cen_query_slice_index in cen_query_slice_indexes:
        #     shortest_dis = 9e9
        #     match_slice_index = -1
        #     for support_slice_index in support_slice_indexes:  
        #         dis = np.abs(query_volume[cen_query_slice_index] - support_volume[support_slice_index]).mean()
        #         if dis<shortest_dis:
        #             shortest_dis = dis
        #             match_slice_index = support_slice_index
        #     refine_support_slice_indexes.append(match_slice_index)
        
        # print(query_file)
        # print(query_slice_indexes,support_slice_indexes,refine_support_slice_indexes)

        # query_slice_indexes = list(range(0,len(query_volume)-1))
        # match_support_slice_indexes = []
        # for query_slice_index in query_slice_indexes:
        #     s_dis = 9e9
        #     match_slice_index = -1
        #     for support_slice_index in support_slice_indexes:  
        #         dis = np.abs(query_volume[query_slice_index] - support_volume[support_slice_index]).mean()
        #         if dis<s_dis:
        #             s_dis = dis
        #             match_slice_index = support_slice_index
        #     match_support_slice_indexes.append(match_slice_index)

        # print(match_support_slice_indexes)
        pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))

        query_slices = query_volume.reshape(len(query_volume),-1)
        support_slices = support_volume.reshape(len(support_volume),-1)
    
        query_volume_pca = pca.fit_transform(query_slices)
        support_volume_pca = pca.fit_transform(support_slices)
    
        query_slice_indexes = list(range(0,len(query_volume)-1))
        match_support_slice_indexes = []
        for i in range(len(query_volume_pca)):
            shortest_dis = 9e9
            match_slice_index = -1
            for j in range(len(support_volume_pca)):  
                dis = np.abs(query_volume_pca[i] - support_volume_pca[j]).mean()
                if dis<shortest_dis:
                    shortest_dis = dis
                    match_slice_index = support_slice_indexes[j]
            match_support_slice_indexes.append(match_slice_index)

        # print([i+1 for i in match_support_slice_indexes])     
 
        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[match_support_slice_indexes[i]:1+match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours3a(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None'):
    ### 精细化support slices的选择
    ### 特征空间的欧式距离与位置信息结合
    ### Support和Query送入segmentor

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        # pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
    
        # query_slices = query_volume.reshape(len(query_volume),-1)
        # support_slices = support_volume.reshape(len(support_volume),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        # query_volume_pca = pca.fit_transform(query_slices)
        # support_volume_pca = pca.fit_transform(support_slices)
    
        # #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)
        # query_slices = query_volume
        # support_slices = support_volume

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        query_slices_fea = []
        for b in range(0, len(query_slices), 10):
            query_slices_batch_x_10 = query_slices[b:b + 10]
            query_slices_batch_x_10 = torch.FloatTensor(query_slices_batch_x_10).cuda()
            _,fea,_ = net.segmentor(query_slices_batch_x_10.unsqueeze(1),[1]*9)
            query_slices_fea.append(fea[4])
        query_slices_fea = torch.cat(query_slices_fea)

        support_slices_fea = []
        for b in range(0, len(support_slices), 10):
            support_slices_batch_x_10 = support_slices[b:b + 10]
            support_slices_batch_x_10 = torch.FloatTensor(support_slices_batch_x_10).cuda()
            _,fea,_ = net.segmentor(support_slices_batch_x_10.unsqueeze(1),[1]*9)
            support_slices_fea.append(fea[4])
        support_slices_fea = torch.cat(support_slices_fea)    
    
        query_slices = query_slices_fea[:,0].cpu().numpy()
        support_slices = support_slices_fea[:,0].cpu().numpy()
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours3b(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None'):
    ### 精细化support slices的选择
    ### 特征空间的欧式距离与位置信息结合
    ### Support送入Conditioner,Query送入Segmentor,没有weight输入Seg

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        # pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
    
        # query_slices = query_volume.reshape(len(query_volume),-1)
        # support_slices = support_volume.reshape(len(support_volume),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        # query_volume_pca = pca.fit_transform(query_slices)
        # support_volume_pca = pca.fit_transform(support_slices)
    
        # #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)
        # query_slices = query_volume
        # support_slices = support_volume

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        query_slices_fea = []
        for b in range(0, len(query_slices), 10):
            query_slices_batch_x_10 = query_slices[b:b + 10]
            query_slices_batch_x_10 = torch.FloatTensor(query_slices_batch_x_10).cuda()
            _,fea,_ = net.segmentor(query_slices_batch_x_10.unsqueeze(1),[1]*9)
            query_slices_fea.append(fea[4])
        query_slices_fea = torch.cat(query_slices_fea)

        support_slices_fea = []
        for b in range(0, len(support_slices), 10):
            support_slices_batch_x_10 = support_slices[b:b + 10]
            support_slices_batch_mask_10 = support_labelmap[b:b+10]
            support_slices_batch_x_10 = torch.FloatTensor(support_slices_batch_x_10).cuda()
            support_slices_batch_mask_10 = torch.FloatTensor(support_slices_batch_mask_10).cuda()
            fea,_ = net.conditioner(torch.cat([support_slices_batch_x_10.unsqueeze(1),support_slices_batch_mask_10.unsqueeze(1)],dim=1))
            support_slices_fea.append(fea[4])
        support_slices_fea = torch.cat(support_slices_fea)    
    
        query_slices = query_slices_fea[:,0].cpu().numpy()
        support_slices = support_slices_fea[:,0].cpu().numpy()
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        #dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours3c(net,support_file,query_path,save_path,query_label=1,Num_support=6,Norm='None'):
    ### 精细化support slices的选择
    ### 特征空间的欧式距离与位置信息结合
    ### Support送入Conditioner,Query送入Segmentor,weight输入Seg

    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        # pca = decomposition.PCA(n_components=min(len(query_volume),len(support_volume)))
    
        # query_slices = query_volume.reshape(len(query_volume),-1)
        # support_slices = support_volume.reshape(len(support_volume),-1)
    
        # #print(query_slices.shape,support_slices.shape)

        # query_volume_pca = pca.fit_transform(query_slices)
        # support_volume_pca = pca.fit_transform(support_slices)
    
        # #print(query_volume_pca.shape,support_volume_pca.shape)

        assert len(query_slice_indexes)==len(support_slice_indexes)
        # query_slices = query_volume
        # support_slices = support_volume

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        # query_slices_fea = []
        # for b in range(0, len(query_slices), 10):
        #     query_slices_batch_x_10 = query_slices[b:b + 10]
        #     query_slices_batch_x_10 = torch.FloatTensor(query_slices_batch_x_10).cuda()
        #     _,fea,_ = net.segmentor(query_slices_batch_x_10.unsqueeze(1),[1]*9)
        #     query_slices_fea.append(fea[4])
        # query_slices_fea = torch.cat(query_slices_fea)

        # support_slices_fea = []
        # for b in range(0, len(support_slices), 10):
        #     support_slices_batch_x_10 = support_slices[b:b + 10]
        #     support_slices_batch_mask_10 = support_labelmap[b:b+10]
        #     support_slices_batch_x_10 = torch.FloatTensor(support_slices_batch_x_10).cuda()
        #     support_slices_batch_mask_10 = torch.FloatTensor(support_slices_batch_mask_10).cuda()
        #     fea,_ = net.conditioner(torch.cat([support_slices_batch_x_10.unsqueeze(1),support_slices_batch_mask_10.unsqueeze(1)],dim=1))
        #     support_slices_fea.append(fea[4])
        # support_slices_fea = torch.cat(support_slices_fea)    
    
        # query_slices = query_slices_fea[:,0].cpu().numpy()
        # support_slices = support_slices_fea[:,0].cpu().numpy()
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        #dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        input1 = torch.cat([torch.FloatTensor(support_slices[k]).unsqueeze(dim=0),torch.FloatTensor(support_labelmap[k]).unsqueeze(dim=0)],dim=0)
                        input2 = torch.FloatTensor(query_slices[j]).unsqueeze(dim=0)
                        input1,input2 = input1.unsqueeze(dim=0).cuda(),input2.unsqueeze(dim=0).cuda()
                        print()
                        _,s_fea,q_fea,_ = net(input1,input2)
                        dis =np.abs(s_fea[4].cpu().numpy() - q_fea[4].cpu().numpy()).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        # dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        input1 = torch.cat([torch.FloatTensor(support_slices[k]).unsqueeze(dim=0),torch.FloatTensor(support_labelmap[k]).unsqueeze(dim=0)],dim=0)
                        input2 = torch.FloatTensor(query_slices[j]).unsqueeze(dim=0)
                        input1,input2 = input1.unsqueeze(dim=0).cuda(),input2.unsqueeze(dim=0).cuda()
                        _,s_fea,q_fea,_ = net(input1,input2)
                        dis =np.abs(s_fea[4].cpu().numpy() - q_fea[4].cpu().numpy()).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), 10):
                query_batch_x_10 = query_batch_x[b:b + 10]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + 10]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_query_volume)), save_path + 'Q_NormVol{}.nii'.format(query_file[-5:-3]))
        #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'Q_SliceNormVol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(norm_support_volume)), save_path + 'S_NormVol{}.nii'.format(support_file[-5:-3]))
    #sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(slice_norm_support_volume)), save_path + 'S_SliceNormVol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list


def evaluate_fss(net,support_file,query_path,save_path,query_label=1,Num_support=6,test_bs=10):
    net.eval()
    #选定support,query
    support = h5py.File(support_file, 'r')
    support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

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

    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])

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

                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list
    
def evaluate_seg(net,support_file,query_path,save_path,query_label=1,Num_support=6):
    net.eval()
    #选定support,query
    # support = h5py.File(support_file, 'r')
    # support_volume,support_labelmap = np.asarray(support['Image']),np.asarray(support['Label'])

    # support_labelmap = np.where(support_labelmap==query_label,1,0)
    # batch, _, _ = support_labelmap.size()
    # slice_with_class = np.sum(support_labelmap.view(batch, -1), dim=1) > 10
    # index = np.where(slice_with_class)
    # support_volume,support_labelmap= support_volume[index],support_labelmap[index]

    # support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    # support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    # support_slice_indexes = support_slice_indexes[:-1]
    # if len(support_slice_indexes) < Num_support:
    #     support_slice_indexes.append(len(support_volume) - 1)

    volume_dice_score_list = []

    for query_file in query_path:
        query = h5py.File(query_file, 'r')
        query_volume,query_labelmap = np.asarray(query['Image']),np.asarray(query['Label'])
        query_labelmap = np.where(query_labelmap==query_label,1,0)
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]
        # query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)
        # if len(query_slice_indexes) < Num_support:
        #     query_slice_indexes.append(len(query_volume) - 1)

        # for i, query_start_slice in enumerate(query_slice_indexes):
        #     if query_start_slice == query_slice_indexes[-1]:
        #         query_batch_x = query_volume[query_slice_indexes[i]:]
        #     else:
        #         query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
        #     query_batch_x = query_batch_x[:,np.newaxis,:,:]
        #     support_batch_x = np.concatenate(support_volume[:,np.newaxis,:,:],\
        #                         support_labelmap[:,np.newaxis,:,:],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
        volume_prediction = []
        volume_prediction_10 = []
        query_batch_x = query_volume[:,np.newaxis,:,:]
        for b in range(0, len(query_batch_x), 10):
            query_batch_x_10 = query_batch_x[b:b + 10]
            #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
            query_batch_x_10 = torch.FloatTensor(query_batch_x_10).cuda()
            #support_batch_x_10 = support_batch_x_10.cuda()

            #out_10 = net(support_batch_x_10,query_batch_x_10)
            out_10 = net(query_batch_x_10)
            batch_output_10 = out_10>0.5
            #batch_output_10 = out_10

            volume_prediction_10.append(batch_output_10)
        volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(query_file[-5:-3],volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(query_file[-5:-3]))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(query_file[-5:-3]))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
        #print('Q_{}, Volume Dice:{:.1f}'.format(query_file[-5:-3],volume_dice_score))

    # sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume.cpu().numpy())), save_path + 'S_Vol{}.nii'.format(support_file[-5:-3]))
    # sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.cpu().numpy())), save_path + 'S_GT{}.nii'.format(support_file[-5:-3]))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list


def evaluate_fss_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,
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

        if save_img:
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
    
def evaluate_fss_ours1b_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,test_bs=10,Norm='None'):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是逐像素的距离
    ### S和Q分别进行PCA 与位置信息结合，就是说S和Q按区域对应，但具体该Q区域内的每张切片对应S区域内的哪张切片，用PCA计算出来
    ### 欧式距离与位置信息结合

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
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

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_ours1c_kfold(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,test_bs=10,Norm='None'):
    ### 精细化support slices的选择
    ### ours1 先尝试最简单的方案，把中心的support_slice和中心的query_slice做匹配，度量标准是support_mask部分的逐像素距离
    ### 欧式距离与位置信息结合

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
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

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = ( np.abs(query_slices[j] - support_slices[k])*support_labelmap[k] ).sum() / support_labelmap[k].sum()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        dis = ( np.abs(query_slices[j] - support_slices[k])*support_labelmap[k] ).sum() / support_labelmap[k].sum()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

import resnet
class resnet_encoder(nn.Module):

    def __init__(self):
        super().__init__()

        cut = 8
        base_model = resnet.resnet18
        # base_model = resnet.resnet34
        # base_model = resnet.resnet50 

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

    def forward(self, x):
        #print(x.shape) 
        x = x.repeat(1,3,1,1)
        #print(x.shape)
        x = self.rn(x)
        #print(x.shape)
        return x

def evaluate_fss_kfold_encoder1a(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,
    Num_support=6,test_bs=10,Norm='None',save_img=False):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 绝对值距离 + 位置信息

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
    
        encoder = resnet_encoder().cuda()

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.append(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.append(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)
        assert len(support_feature) == len(support_slices)
        assert len(query_feature) == len(query_slices)
        # print(len(query_slices),len(query_feature))
        # print(len(support_slices),len(support_feature))

        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        #dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        dis = torch.abs(query_feature[j] - support_feature[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        # dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        dis = torch.abs(query_feature[j] - support_feature[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
 
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))
        # print(total_match_support_slice_indexes)

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        if save_img:
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

def evaluate_fss_kfold_encoder1b(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,
    Num_support=6,test_bs=10,Norm='None'):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 内积距离 + 位置信息

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
 
        encoder = resnet_encoder().cuda()

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.append(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.append(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)
        assert len(support_feature) == len(support_slices)
        assert len(query_feature) == len(query_slices)

        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)): 
                        dis = torch.cosine_similarity(query_feature[j],support_feature[k],dim=0).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        dis = torch.cosine_similarity(query_feature[j],support_feature[k],dim=0).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))
        # print(total_match_support_slice_indexes)

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_kfold_encoder1a_(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,test_bs=10,Norm='None'):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 绝对值距离 

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        encoder = resnet_encoder().cuda()

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.extend(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.extend(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)
        
        total_match_support_slice_indexes = []
        for i in range(len(query_feature)):
            s_dis = 9e9
            match_slice_index = -1
            for j in range(len(support_feature)): 
                dis = torch.abs(query_feature[i] - support_feature[j]).mean()
                if dis<s_dis:
                    s_dis = dis
                    match_slice_index = j
            total_match_support_slice_indexes.append(match_slice_index)
        # assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            

            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]:1+total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_kfold_encoder1b_(net,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,Num_support=6,test_bs=10,Norm='None'):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 内积距离 

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        encoder = resnet_encoder().cuda()

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.extend(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.extend(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)

        total_match_support_slice_indexes = []
        for i in range(len(query_feature)):
            s_dis = 9e9
            match_slice_index = -1
            for j in range(len(support_feature)): 
                dis = torch.cosine_similarity(query_feature[i],support_feature[j],dim=0).mean()
                if dis<s_dis:
                    s_dis = dis
                    match_slice_index = j
            total_match_support_slice_indexes.append(match_slice_index)
        # assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            

            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]:1+total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_kfold_AE1a(net,encoder_path,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,
    Num_support=6,test_bs=10,Norm='None'):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 绝对值距离 + 位置信息

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        encoder = autoencoder(eval=True).cuda()
        checkpoint = torch.load(encoder_path)
        encoder.load_state_dict(checkpoint['state_dict'])

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.append(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.append(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)
        assert len(support_feature) == len(support_slices)
        assert len(query_feature) == len(query_slices)
        # print(len(query_slices),len(query_feature))
        # print(len(support_slices),len(support_feature))

        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)):
                        #dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        #dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        dis = torch.abs(query_feature[j] - support_feature[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        # dis = np.abs(query_volume_pca[j] - support_volume_pca[k]).mean()
                        # dis = np.abs(query_slices[j] - support_slices[k]).mean()
                        dis = torch.abs(query_feature[j] - support_feature[k]).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
 
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))
        # print(total_match_support_slice_indexes)

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list

def evaluate_fss_kfold_AE1b(net,encoder_path,whole_image,whole_label,case_start_index,support_item,query_item,save_path,query_label,
    Num_support=6,test_bs=10,Norm='None'):
    ### encoder输出的特征空间里度量
    ### encoder是预训练好的resnet18,直接使用
    ### 内积距离 + 位置信息

    net.eval()
    #选定support,query
    support_volume = whole_image[case_start_index[support_item]:case_start_index[support_item+1]]
    support_labelmap = whole_label[case_start_index[support_item]:case_start_index[support_item+1]]

    support_labelmap = np.where(support_labelmap==query_label,1,0)
    batch, _, _ = support_labelmap.shape
    slice_with_class = np.sum(support_labelmap.reshape(batch, -1), axis=1) > 10
    index = np.where(slice_with_class)
    support_volume,support_labelmap= support_volume[index],support_labelmap[index]
    
    norm_support_volume = (support_volume - np.mean(support_volume))/np.std(support_volume)
    slice_norm_support_volume = np.zeros_like(support_volume)
    for i in range(len(support_volume)):
        slice_norm_support_volume[i] = (support_volume[i] - np.mean(support_volume[i]))/np.std(support_volume[i])

    cen_support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support+1)).astype(int)
    cen_support_slice_indexes += (len(support_volume) // Num_support) // 2 #center
    cen_support_slice_indexes = cen_support_slice_indexes[:-1]
    if len(cen_support_slice_indexes) < Num_support:
        cen_support_slice_indexes.append(len(support_volume) - 1)

    support_slice_indexes = np.round(np.linspace(0, len(support_volume) - 1, Num_support)).astype(int)
    if len(support_slice_indexes) < Num_support:
        support_slice_indexes.append(len(support_volume) - 1)

    
    volume_dice_score_list = []
    for item in query_item:
        query_volume = whole_image[case_start_index[item]:case_start_index[item+1]]
        query_labelmap = whole_label[case_start_index[item]:case_start_index[item+1]]
        
        query_labelmap = np.asarray(np.where(query_labelmap==query_label,1,0))
        batch, _, _ = query_labelmap.shape
        slice_with_class = np.sum(query_labelmap.reshape(batch, -1), axis=1) > 10
        index = np.where(slice_with_class)
        query_volume,query_labelmap = query_volume[index],query_labelmap[index]

        norm_query_volume = (query_volume - np.mean(query_volume))/np.std(query_volume)
        slice_norm_query_volume = np.zeros_like(query_volume)
        for i in range(len(query_volume)):
            slice_norm_query_volume[i] = (query_volume[i] - np.mean(query_volume[i]))/np.std(query_volume[i])

        query_slice_indexes = np.round(np.linspace(0, len(query_volume) - 1, Num_support)).astype(int)

        if len(query_slice_indexes) < Num_support:
            query_slice_indexes.append(len(query_volume) - 1)

        assert len(query_slice_indexes)==len(support_slice_indexes)

        if Norm=='None':
            query_slices = query_volume
            support_slices = support_volume
        elif Norm == 'Case':
            query_slices = norm_query_volume
            support_slices = norm_support_volume
        elif Norm == 'Slice':
            query_slices = slice_norm_query_volume
            support_slices = slice_norm_support_volume
        
        encoder = autoencoder(eval=True).cuda()
        checkpoint = torch.load(encoder_path)
        encoder.load_state_dict(checkpoint['state_dict'])

        query_feature = []
        for b in range(0, len(query_slices), test_bs*2):
            query_batch = query_slices[b:b + test_bs*2]
            query_batch = torch.FloatTensor(query_batch).cuda()
            query_feature.append(encoder(query_batch.unsqueeze(1)))
        query_feature = torch.cat(query_feature)
        support_feature = []
        for b in range(0, len(support_slices), test_bs*2):
            support_batch = support_slices[b:b + test_bs*2]
            support_batch = torch.FloatTensor(support_batch).cuda()
            support_feature.append(encoder(support_batch.unsqueeze(1)))
        support_feature = torch.cat(support_feature)
        assert len(support_feature) == len(support_slices)
        assert len(query_feature) == len(query_slices)

        
        total_match_support_slice_indexes = []
        for i in range(len(query_slice_indexes)):
            match_support_slice_indexes = []
            if i==len(query_slice_indexes)-1:
                for j in range(query_slice_indexes[i],len(query_slices)):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],len(support_slices)): 
                        dis = torch.cosine_similarity(query_feature[j],support_feature[k],dim=0).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            else:        
                for j in range(query_slice_indexes[i],query_slice_indexes[i+1]):
                    s_dis = 9e9
                    match_slice_index = -1
                    for k in range(support_slice_indexes[i],support_slice_indexes[i+1]):
                        dis = torch.cosine_similarity(query_feature[j],support_feature[k],dim=0).mean()
                        if dis<s_dis:
                            s_dis = dis
                            match_slice_index = k
                    match_support_slice_indexes.append(match_slice_index)
            total_match_support_slice_indexes.append(match_support_slice_indexes)
        
        assert len(total_match_support_slice_indexes) == Num_support

        # print(query_file)
        # print([i + 1 for i in query_slice_indexes],[i + 1 for i in cen_support_slice_indexes],[i + 1 for j in total_match_support_slice_indexes for i in j])
        # print(len(support_volume),len(query_volume))
        # print(total_match_support_slice_indexes)

        volume_prediction = []
        for i, query_start_slice in enumerate(query_slice_indexes):
            if query_start_slice == query_slice_indexes[-1]:
                query_batch_x = query_volume[query_slice_indexes[i]:]
            else:
                query_batch_x = query_volume[query_slice_indexes[i]:query_slice_indexes[i + 1]]
            
            query_batch_x = query_batch_x[:,np.newaxis,:,:]
            # support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
            #                     support_labelmap[:,np.newaxis,:,:]],axis=1)[support_slice_indexes[i]:1+support_slice_indexes[i]]
            assert len(total_match_support_slice_indexes[i]) == len(query_batch_x)
            support_batch_x = np.concatenate([support_volume[:,np.newaxis,:,:],\
                                support_labelmap[:,np.newaxis,:,:]],axis=1)[total_match_support_slice_indexes[i]]
            
            volume_prediction_10 = []
            for b in range(0, len(query_batch_x), test_bs):
                query_batch_x_10 = query_batch_x[b:b + test_bs]
                # support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10), 1, 1, 1)
                #support_batch_x_10 = support_batch_x.repeat(len(query_batch_x_10),axis=0)
                support_batch_x_10 = support_batch_x[b:b + test_bs]
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

                
                #_, batch_output_10 = torch.max(F.softmax(out_10, dim=1), dim=1)
                #batch_output_10 = out_10

                volume_prediction_10.append(batch_output_10)
            volume_prediction.extend(volume_prediction_10)
        volume_prediction = torch.cat(volume_prediction)
        volume_dice_score = dice_score_binary(volume_prediction[:len(query_labelmap)],torch.FloatTensor(query_labelmap).cuda())*100

        volume_prediction = (volume_prediction.cpu().numpy()).astype('uint8')
        #volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(volume_prediction)), save_path + 'Q_Pred{}_dice{:.1f}.nii'.format(item,volume_dice_score))
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_labelmap.astype('uint8'))), save_path + 'Q_GT{}.nii'.format(item))

        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(query_volume)), save_path + 'Q_Vol{}.nii'.format(item))
        
        volume_dice_score = volume_dice_score.item()
        volume_dice_score_list.append(volume_dice_score)
       

    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_volume)), save_path + 'S_Vol{}.nii'.format(support_item))
    sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(support_labelmap.astype('uint8'))), save_path + 'S_GT{}.nii'.format(support_item))
    #print('Q_Percase, Volume Dice:{:.1f}'.format(sum(volume_dice_score_list)/len(volume_dice_score_list)))
    return volume_dice_score_list


