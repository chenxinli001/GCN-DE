# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:55:41 2020

@author: Sunly
"""

import numpy as np
import argparse



def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretext", type=str,default='./result_paper4/')
    parser.add_argument("--repeat", type=bool, default=True)
    parser.add_argument("--method", '--m', type=str, default='t0.1_cons_pretrain_nlc_[5]')
    return parser.parse_args()

args = get_args()

# dataset_list = ['MRI','CT']
#organ_list = ['liver','spleen']
# organ_list = ['liver','spleen','left_kidney','right_kidney']
organ_list = ['liver','spleen']
dataset_list = ['MRI']
#organ_list = ['spleen']
#args.pretext = './result_ex/'
method = args.method

for dataset in dataset_list:
    dice_list = []
    dice_list1 = []
    for organ in organ_list:
        folder = args.pretext + dataset + '/' + organ+ '/'+ args.method
        if args.repeat:
            a = []
            a1 = []
            for i in range(1,4):
                file = open(folder+'_run'+str(i)+'/result.txt')
                while(1):
                    line = file.readline()
                    if 'Avg Best Val dice' in line and 'Avg Best Val dice1' in line:
                        a.append(float(line[-33:-29]))
                        a1.append(float(line[-7:]))
                        break
                    elif 'Avg Best Val dice' in line:
                        a.append(float(line[-7:]))
                        break
            if len(a1):
                print('method is {}, dataset is {}, organ is {}. dice is {:.1f},{:.1f},{:.1f}, avg is {:.2f}. dice1 is {:.1f},{:.1f},{:.1f}, avg is {:.2f}'.format(method,dataset,organ,a[0],a[1],
                    a[2],sum(a)/len(a),a1[0],a1[1],a1[2],sum(a1)/len(a1)))
                dice_list.append(sum(a)/len(a))
                dice_list1.append(sum(a1)/len(a1))
            else:
                print('method is {}, dataset is {}, organ is {}. dice is {:.1f},{:.1f},{:.1f}, avg is {:.2f}.'.format(method,dataset,organ,a[0],a[1],a[2],sum(a)/len(a)))
        else:
            file = open(folder+'_'+'/result.txt')
            while(1):
                line = file.readline()
                if 'Avg Best Val dice' in line and 'Avg Best Val dice1' in line:
                    print(method,dataset,organ,float(line[-33:-29]),float(line[-7:]))
                    dice_list.append(float(line[-33:-29]))
                    dice_list1.append(float(line[-7:]))
                    break
                elif 'Avg Best Val dice' in line:
                    print(method,dataset,organ,float(line[-7:]))
                    dice_list.append(float(line[-7:]))
                    break
    if len(dice_list1):
        print(method,dataset,sum(dice_list)/len(dice_list),sum(dice_list1)/len(dice_list1))
    else:
        print(method,dataset,sum(dice_list)/len(dice_list),sum(dice_list1)/len(dice_list1))

















































