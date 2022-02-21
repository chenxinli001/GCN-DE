# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import numpy as np
import argparse
import re



def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretext", '--p',type=str,default='./result_paper/')
    parser.add_argument("--repeat", '--r',type=int, default=0)
    parser.add_argument("--method", '--m', type=str, default='Rakelly')#'cons1_ms_w_layer9_t0.0_margin0_pretrain')
    return parser.parse_args()

args = get_args()

dataset_list = ['MRI','CT']
organ_list = ['liver','spleen','left_kidney','right_kidney']
# dataset_list = ['MRI']
# organ_list = ['spleen']  
# args.pretext = './result_ex/'

method = args.method
#method = 'base_SE1_pretrain'

for dataset in dataset_list:
    dice_list = []
    dice_list1 = []
    for organ in organ_list:
        folder = args.pretext + dataset + '/' + organ+ '/'+ method
        if args.repeat:
            a = []
            a1 = []
            for i in range(1,4):
                file = open(folder+'_run'+str(i)+'/result.txt')
                while(1):
                    line = file.readline()
                    if 'Avg Best Val dice' in line and 'Avg Best Val dice1' in line:
                        temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                        a.append(float(temp[0]))
                        a1.append(float(temp[2]))

                        # a.append(float(line[-33:-29]))
                        # a1.append(float(line[-7:]))
                        break
                    elif 'Avg Best Val dice' in line:
                        # a.append(float(line[-7:]))
                        a.append(float(temp[0]))
                        break
            if len(a1):
                print('method is {}, dataset is {}, organ is {}. dice is {:.2f},{:.2f},{:.2f}, avg is {:.3f}. dice1 is {:.2f},{:.2f},{:.2f}, avg is {:.3f}'.format(method,dataset,organ,a[0],a[1],
                    a[2],sum(a)/len(a),a1[0],a1[1],a1[2],sum(a1)/len(a1)))
                dice_list.append(sum(a)/len(a))
                dice_list1.append(sum(a1)/len(a1))
            else:
                print('method is {}, dataset is {}, organ is {}. dice is {:.2f},{:.2f},{:.2f}, avg is {:.3f}.'.format(method,dataset,organ,a[0],a[1],a[2],sum(a)/len(a)))
        else:
            a, a1 = [], []
            file = open(folder+'/result.txt')
            while(1):
                line = file.readline()
                if 'Avg Best Val dice' in line and 'Avg Best Val dice1' in line:
                    temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    a.append(float(temp[0]))
                    a1.append(float(temp[2]))
                    break
                elif 'Avg Best Val dice' in line:
                    temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    a.append(float(temp[0]))
                    break
            if len(a1):
                print('method is {}, dataset is {}, organ is {}. dice is {:.3f}, dice1 is {:.3f}'.format(method,dataset,organ,a[0],a1[0]))
                dice_list.append(a[0])
                dice_list1.append(a1[0])
            else:
                print('method is {}, dataset is {}, organ is {}. dice is {:.3f}'.format(method,dataset,organ,a[0]))
                dice_list.append(a[0])

    if len(dice_list1):
        print(method,dataset,sum(dice_list)/len(dice_list),sum(dice_list1)/len(dice_list1))
    else:
        print(method,dataset,sum(dice_list)/len(dice_list))

















































