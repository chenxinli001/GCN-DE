import torch.optim as optim
import numpy as np

def get_finetune_optimizer( model):
    #lr = args.lr
    lr=1e-4
    weight_list = []
    bias_list = []
    last_weight_list = []
    last_bias_list =[]
    for name,value in model.named_parameters():
        #print(name)
        if 'cls' in name or 'lstm' in name:
            if 'weight' in name:
                last_weight_list.append(value)
            elif 'bias' in name:
                last_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    opt = optim.SGD([{'params': weight_list, 'lr':lr},
                     {'params':bias_list, 'lr':lr*2},
                     {'params':last_weight_list, 'lr':lr*10},
                     {'params': last_bias_list, 'lr':lr*20}], momentum=0.99, weight_decay=0.0005)

    return opt
