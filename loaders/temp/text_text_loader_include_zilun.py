from hashlib import new
import os
from os.path import join as pjoin
import collections
import json
from numpy.lib.histograms import histogram_bin_edges
import torch
import numpy as np
import cv2
import random
import torch.nn.functional as F
from torch.utils import data
import glob

src_list = ['p',"0","1","2","3","4","5","6","7","8","9","0-","1-","2-","3-","4-","5-","6-","7-","8-","9-","0+","1+","2+","3+","4+","5+","6+","7+","8+","9+"]

tgt_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']



class TexttextLoader_zilun(data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        file = open(file_path,'r')
        self.datas = file.readlines()
        file.close()

    def zilun2zhizhen(self,predicts,gts):

        zilun = predicts[0]
        if zilun[-1]=='v':
            first_place = zilun[-2:]
            first_place = first_place.replace('v','')+random.choice(['+','-'])
        else:
            first_place = zilun[-1]

        new_predicts = predicts.copy()
        new_predicts[0] = first_place

        new_gts = gts[-len(predicts):]
        
        return new_predicts[1:],new_gts[1:]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        name,_,pointers,gts = data.split('\t')
        gts = gts.replace('\n','').replace('.','').replace(',','').replace('，','').replace(' ','')
        pointers = pointers.split(';')
        predicts = []
        for pointer in pointers:
            if pointer=='':
                continue
            bbox, predict = pointer.split('/')
            predicts.append(predict.replace('e',''))

        if len(predicts[0])>2:
            predicts,gts = self.zilun2zhizhen(predicts,gts)


        ## fill with 0
        if len(predicts)!=len(gts):
            gts = '0'*(len(predicts)-len(gts))+gts ## padding


        ## get encoder input
        predict_num = len(predicts)
        if len(predicts)<12:
            predicts = ';'.join(predicts)
            predicts = predicts+';p'*(12-predict_num) ## padding
            predicts = predicts.split(';')
        for i,predict in enumerate(predicts):
            predicts[i] = src_list.index(predict)  
        src = torch.from_numpy(np.asarray(predicts))

        ## get decoder input
        tgt = 's' + gts[::-1]
        ### padding
        if len(tgt)<13:
            tgt = tgt + 'p'*(13-len(tgt))
        tgt = tgt + 'e'
        tgt = list(tgt)
        for i,temp in enumerate(tgt):
            tgt[i] = tgt_list.index(temp)
        tgt = torch.from_numpy(np.asarray(tgt))



        return src,tgt

