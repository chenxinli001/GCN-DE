# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:20:44 2020

@author: Sunly
"""

from utils.shot_batch_sampler import OneShotBatchSampler, get_lab_list
from utils.data_utils import get_imdb_dataset, split_batch
import argparse
from settings import Settings
import torch

# Mapping
# 0 is background
# 1 is liver
# 2 is right kidney
# 3 is left kidney
# 4 is spleen

import matplotlib.pyplot as plt

def mapping_ind_to_organ(i):
    ind = int(i)
    if ind == 1:
        return "Liver"
    elif ind == 2:
        return "Right Kidney"
    elif ind == 3:
        return "Left Kidney"
    elif ind == 4:
        return "Spleen"
    else:
        return "Background"

def load_data(data_params):
    print("Loading dataset")
    train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default="train",
                    help='run mode, valid values are train and eval')
parser.add_argument('--device', '-d', required=False,
                    help='device to run on')
args = parser.parse_args()

settings = Settings()
common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
    'NETWORK'], settings['TRAINING'], settings['EVAL']

train_data, test_data = load_data(data_params)

fold = "fold1"
train_sampler = OneShotBatchSampler(train_data.y, 'train', fold, train_params['train_batch_size'], iteration=train_params['iterations'])
test_sampler = OneShotBatchSampler(test_data.y, 'val', fold, train_params['val_batch_size'], iteration=train_params['test_iterations'])

train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(test_data, batch_sampler=test_sampler)

for i,train_data in enumerate(train_loader):
    ind = i
    image = train_data[0]
    label = train_data[1]
    query_label_organ = train_loader.batch_sampler.query_label
    
    supp_image, query_image, supp_label, query_label = split_batch(image, label, int(query_label_organ))
    supp_image = supp_image.cpu().numpy().squeeze()
    query_image = query_image.cpu().numpy().squeeze()
    supp_label = supp_label.cpu().numpy().squeeze()
    query_label = query_label.cpu().numpy().squeeze()
    
    # figure,ax = plt.subplots(4,2)
    # ax[0][0].imshow(image,cmap="gray")
    # ax[0][0].axis("off")
    # ax[0][1].imshow(label)
    # ax[0][1].axis("off")
    
    plt.figure()
    
    organ_name = mapping_ind_to_organ(query_label_organ)
    
    plt.suptitle(organ_name)
    
    ax1 = plt.subplot(241)
    plt.imshow(supp_image[0],cmap="gray")
    plt.axis("off")
    ax1.set_title("Supp-Image-B1")
    
    ax2 = plt.subplot(242)
    plt.imshow(supp_label[0])
    plt.axis("off")
    ax2.set_title("Supp-Label-B1")
    
    ax3 = plt.subplot(243)
    plt.imshow(query_image[0],cmap="gray")
    plt.axis("off")
    ax3.set_title("Query-Image-B1")
    
    ax4 = plt.subplot(244)
    plt.imshow(query_label[0])
    plt.axis("off")
    ax4.set_title("Query-Label-B1")
    
    ax5 = plt.subplot(245)
    plt.imshow(supp_image[1],cmap="gray")
    plt.axis("off")
    ax5.set_title("Supp-Image-B2")
    
    ax6 = plt.subplot(246)
    plt.imshow(supp_label[1])
    plt.axis("off")
    ax6.set_title("Supp-Label-B2")
    
    ax7 = plt.subplot(247)
    plt.imshow(query_image[1],cmap="gray")
    plt.axis("off")
    ax7.set_title("Query-Image-B2")
    
    ax8 = plt.subplot(248)
    plt.imshow(query_label[1])assistance
    plt.axis("off")
    ax8.set_title("Query-Label-B2")
    
    if i == 20:
        break
    














