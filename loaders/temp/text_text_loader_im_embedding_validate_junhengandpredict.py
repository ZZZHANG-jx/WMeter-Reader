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
import math
from math import *

embedding_list = ['p',"0","1","2","3","4","5","6","7","8","9","0-","1-","2-","3-","4-","5-","6-","7-","8-","9-","0+","1+","2+","3+","4+","5+","6+","7+","8+","9+"]

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']


class TexttextLoader_im(data.Dataset):
    def __init__(self, file_paths):
        self.datas = []
        
        if isinstance(file_paths,list):
            for file_path in file_paths:
                self.file_path = file_path
                file = open(file_path,'r')
                self.datas += file.readlines()
                file.close()
        else:
            self.file_path = file_paths
            file = open(file_paths,'r')
            self.datas = file.readlines()
            file.close()

        file = open('./data/datas/20220819-20220825_predict/20220819-20220825_finaltopointer.txt','r')
        self.predict_datas = file.readlines()
        file.close()            

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):


        data = self.datas[index]

        name,_,pointers,gts = data.split('\t')

        if 'train' in self.file_path:
            for predict_data in self.predict_datas:
                pred_name,_,pred_pointers,pred_gts = predict_data.split('\t')
                if pred_name == name:
                    name,_,pointers,gts = predict_data.split('\t')
                    break

        im_path = os.path.join('./data/raw_im',name)
        im = cv2.imread(im_path)
        h,w = im.shape[:2]
        gts = gts.replace('\n','').replace('.','').replace(',','').replace('，','').replace(' ','')
        pointers = pointers.split(';')
        predicts = []
        images = []
        center_xs = []
        center_ys = []
        for pointer in pointers:
            if pointer=='':
                continue
            bbox, predict = pointer.split('/')
            predicts.append(predict.replace('e',''))
            cood = bbox.split(',')
            x0 = min(max(int(cood[0]),0),w)
            y0 = min(max(int(cood[1]),0),h)
            x1 = min(max(int(cood[2]),0),w)
            y1 = min(max(int(cood[3]),0),h)
            x2 = min(max(int(cood[4]),0),w)
            y2 = min(max(int(cood[5]),0),h)
            x3 = min(max(int(cood[6]),0),w)
            y3 = min(max(int(cood[7]),0),h)
            center_x = int(((x0+x2)/2)/w*200)
            center_y = int(((y0+y2)/2)/w*200)
            sub_im = self.rotate(im,[x0,y0],[x3,y3],[x2, y2],[x1,y1])
            sub_im = cv2.resize(sub_im,(48,48))
            # cv2.imshow('before',sub_im)
            if 'train' in self.file_path:
                sub_im = self.randomAugment(sub_im)

            # cv2.imshow('after',sub_im)
            # cv2.waitKey(0)
            images.append(sub_im)
            center_xs.append(center_x)
            center_ys.append(center_y)

        if len(predicts[0])>2:
            predicts,gts,images,center_xs,center_ys = self.zilun2zhizhen(predicts,gts,images,center_xs,center_ys)
        #     continue

        # print(predicts,gts,name)

        predicts = predicts[::-1]
        images = images[::-1]
        center_xs = center_xs[::-1]
        center_ys = center_ys[::-1]
        predict_num = len(predicts)


        ## fill with 0
        if predict_num!=len(gts):
            gts = '0'*(predict_num-len(gts))+gts 


        ## 去掉最高位
        if 'train' in self.file_path and predict_num>=8:
            if random.uniform(0,1)<0:
                rule_out_num = random.choice([4,5])
                predicts = predicts[:-rule_out_num]
                images = images[:-rule_out_num]
                center_xs = center_xs[:-rule_out_num]
                center_ys = center_ys[:-rule_out_num]
                gts = gts[rule_out_num:]
        
        predict_num = len(predicts)

        ## get encoder input
        if len(predicts)<12:
            predicts = ';'.join(predicts)
            predicts = predicts+';p'*(12-predict_num) ## padding
            predicts = predicts.split(';')
            images = images + [np.zeros((48,48,3))]*(12-predict_num)
            center_xs = center_xs + [0]*(12-predict_num)
            center_ys = center_ys + [0]*(12-predict_num)
        for i,predict in enumerate(predicts):
            predicts[i] = embedding_list.index(predict)  
        src = torch.from_numpy(np.asarray(predicts))
        src_im = torch.from_numpy(np.asarray(images))
        src_im = ((src_im/255-0.5)*2).permute(0,3,1,2)
        src_x = torch.from_numpy(np.asarray(center_xs))
        src_y = torch.from_numpy(np.asarray(center_ys))

        # print(src.shape,src_im.shape,src_x.shape,src_y.shape)
        ## get decoder input
        tgt = 's' + gts[::-1]
        ### padding
        if len(tgt)<13:
            tgt = tgt + 'p'*(13-len(tgt))
        tgt = tgt + 'e'
        tgt = list(tgt)
        for i,temp in enumerate(tgt):
            tgt[i] = type_list.index(temp)
        tgt = torch.from_numpy(np.asarray(tgt))

        return src,tgt,src_im,src_x,src_y

    def zilun2zhizhen(self,predicts,gts,images,center_xs,center_ys):
        predicts_num = len(predicts)

        zilun = predicts[0]
        if zilun[-1]=='v':
            first_place = zilun[-2:]
            first_place = first_place.replace('v','')+random.choice(['+','-'])
        else:
            first_place = zilun[-1]

        new_predicts = predicts.copy()
        new_predicts[0] = first_place

        new_gts = gts[-len(predicts):]
        
        # return new_predicts[1:],new_gts[1:]
        return predicts[1:],gts[::-1][:predicts_num-1][::-1],images[1:],center_xs[1:],center_ys[1:]


    def rotate(self, img, pt1, pt2, pt3, pt4):
        # print(pt1,pt2,pt3,pt4)
        withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  
        heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
        # print(withRect,heightRect)
        angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  
        # print(angle)

        if pt4[1]>pt1[1]:
            # print("---")
            pass
        else:
            # print ""
            angle=-angle

        height = img.shape[0]  
        width = img.shape[1]   
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
        # cv2.imshow('rotateImg2',  imgRotation)
        # cv2.waitKey(0)

        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

        if pt2[1]>pt4[1]:
            pt2[1],pt4[1]=pt4[1],pt2[1]
        if pt1[0]>pt3[0]:
            pt1[0],pt3[0]=pt3[0],pt1[0]

        imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        # cv2.imshow("imgOut", imgOut) 
        # cv2.waitKey(0)
        return imgOut

    def transform(self,img):
        if len(img.shape) == 2 :
            img = np.expand_dims(img,-1)
        img = img.astype(np.float)/255. 
        img = (img-0.5)*2
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img
    
    def randomAugment(self,in_img):
        h,w = in_img.shape[:2]
        self.img_size = (h,w)
        # random crop
        if random.uniform(0,1) <= 0.5:
            in_img = in_img[random.randint(0,3):h-random.randint(0,3),random.randint(0,3):h-random.randint(0,3)]
            in_img = cv2.resize(in_img,(w,h))
        # crop_h = random.randint(128,1024)
        ## brightness
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.5:
            high = 1.3
            low = 0.8
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.5:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        
        ## scale and rotate
        if random.uniform(0,1) <= 0:
            y,x = self.img_size
            angle = random.uniform(-5,5)
            scale = random.uniform(0.95,1.05)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            in_img = cv2.warpAffine(in_img,M,(x,y),borderValue=0)
        # add noise
        ## jpegcompression
        if random.uniform(0,1) <= 0.5:
            quanlity_high = 95
            quanlity_low = 45
            quanlity = int(np.random.randint(quanlity_low,quanlity_high))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),quanlity]
            result, encimg = cv2.imencode('.jpg',in_img,encode_param)
            in_img = cv2.imdecode(encimg,1).astype(np.uint8)
        ## gaussiannoise
        if random.uniform(0,1) <= 0.5:
            mean = 0
            sigma = 0.03
            noise_ratio = 0.004
            num_noise = int(np.ceil(noise_ratio*w))
            coords = [np.random.randint(0,i-1,int(num_noise)) for i in [h,w]] 
            gauss = np.random.normal(mean,sigma,num_noise*3)*255
            guass = np.reshape(gauss,(-1,3))
            in_img = in_img.astype(np.float64)
            in_img[tuple(coords)] += guass
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## blur
        if random.uniform(0,1) <= 0.5:
            ksize = np.random.randint(1,2)*2 + 1
            in_img = cv2.blur(in_img,(ksize,ksize))
        return in_img


