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
from PIL import Image, ImageFilter
import lmdb
import pyarrow
import pickle
import copy 

embedding_list = ['p',"0","1","2","3","4","5","6","7","8","9","0-","1-","2-","3-","4-","5-","6-","7-","8-","9-","0+","1+","2+","3+","4+","5+","6+","7+","8+","9+"]

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']


class TexttextLoader_lmdb(data.Dataset):
    def __init__(self, db_path='./data/datas/train_lmdb', db_name='train_1-10_50k', transform=None, target_transform=None, backend='cv2'):
        self.db_name = db_name
        self.datas = []
        self.env = lmdb.open(os.path.join(db_path, '{}.lmdb'.format(db_name)),
                             subdir=False,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin() as txn:
            # self.length = pyarrow.deserialize(txn.get(b'__len__'))
            self.length = pickle.loads(txn.get(b'__len__'))

        self.map_list = [str(i).encode() for i in range(self.length)]
        self.transform = transform
        self.target_transform = target_transform
        self.backend = backend

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        with self.env.begin() as txn:
            byteflow = txn.get(self.map_list[item])
        # unpacked = pyarrow.deserialize(byteflow)
        unpacked = pickle.loads(byteflow)
        ## unpacked is exectly the data list that you convert to lmdb
        data = unpacked[0]
        src,tgt,src_im,src_x,src_y = data

        tgt = torch.from_numpy(np.asarray(tgt))
        src = torch.from_numpy(np.asarray(src))
        src_x = torch.from_numpy(np.asarray(src_x))
        src_y = torch.from_numpy(np.asarray(src_y))
        
        if 'test' in self.db_name:
            aug_src_im = []
            for im in src_im:
                im = np.asarray(im).astype(np.uint8).copy()
                im = cv2.resize(im,(64,64))
                im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
                aug_src_im.append(im)
        else:
            aug_src_im = []
            src_im = src_im
            for im in src_im:
                im = np.asarray(im).astype(np.uint8).copy()
                im = cv2.resize(im,(64,64))
                im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
                im = self.randomAugmentv2(im)
                aug_src_im.append(im)

        src_im = torch.from_numpy(np.asarray(aug_src_im)).unsqueeze(-1)
        src_im = ((src_im/255-0.5)*2).permute(0,3,1,2)

        return src,tgt,src_im,src_x,src_y


    def __shift_padding__(self,image,hor_shift_ratio,ver_shift_ratio):
        h,w = image.shape[:2]
        pad_h = np.int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h))
        pad_w = np.int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w))
        # new_h = h + pad_h
        # new_w = w + pad_w
        # if image.ndim == 2:
        #     new_image = np.ones((new_h,new_w),dtype=image.dtype)
        # else:
        #     new_image = np.ones((new_h,new_w,image.shape[-1]),dtype=image.dtype)
        # new_image[int(np.round(ver_shift_ratio[1]*h)):int(np.round(ver_shift_ratio[1]*h))+h,int(np.round(hor_shift_ratio[1]*w)):int(np.round(hor_shift_ratio[1]*w))+w] = image

        new_image = cv2.copyMakeBorder(image,int(pad_h/2),int(pad_h/2),int(pad_w/2),int(pad_w/2),cv2.BORDER_REPLICATE)

        top = np.random.randint(0,int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h)))
        left = np.random.randint(0,int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w)))
        image = new_image[top:top+h,left:left+w]
        return image
    def __shift_padding_origin__(self,image,hor_shift_ratio,ver_shift_ratio):
        h,w = image.shape[:2]
        pad_h = np.int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h))
        pad_w = np.int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w))
        new_h = h + pad_h
        new_w = w + pad_w
        if image.ndim == 2:
            new_image = np.ones((new_h,new_w),dtype=image.dtype)
        else:
            new_image = np.ones((new_h,new_w,image.shape[-1]),dtype=image.dtype)
        new_image[int(np.round(ver_shift_ratio[1]*h)):int(np.round(ver_shift_ratio[1]*h))+h,int(np.round(hor_shift_ratio[1]*w)):int(np.round(hor_shift_ratio[1]*w))+w] = image

        top = np.random.randint(0,int(np.round((ver_shift_ratio[1]-ver_shift_ratio[0])*h)))
        left = np.random.randint(0,int(np.round((hor_shift_ratio[1]-hor_shift_ratio[0])*w)))
        image = new_image[top:top+h,left:left+w]
        return image
    def __gaussianblur__(self,image,radius):
        image = Image.fromarray(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        image = np.asarray(image)
        return image
    
    def __edgeenhance__(self,image):
        image = Image.fromarray(image)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = np.asarray(image)
        return image

    def __hist_norm__(self, image):
        out = np.zeros(image.shape, np.uint8)
        cv2.normalize(image, out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        return image
    def erase_augment(self,img):
        img = img.copy()
        h,w = img.shape[:2]
        for _ in range(100):
            area = int(np.random.uniform(0.02,0.3)*h*w)
            ration = np.random.uniform(0.3,1/0.3)
            h_shift = int(round(np.sqrt(area*ration)))
            w_shift = int(round(np.sqrt(area/ration)))
            if h_shift < h and w_shift < w:
                h_start = np.random.randint(0,h-h_shift)
                w_start = np.random.randint(0,w-w_shift)
                randm_area = np.random.randint(low=0,high=255,size=(h_shift,w_shift))
                img[h_start:h_start+h_shift,w_start:w_start+w_shift] = randm_area
                return img
            else:
                continue
    def brightness_contrast(self,in_img):
        high = 1.3
        low = 0.3
        ratio = np.random.uniform(low,high)
        in_img = in_img.astype(np.float64)*ratio
        in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        high = 1.3
        low = 0.2
        ratio = np.random.uniform(low,high)
        gray = in_img
        # gray = cv2.cvtColor(in_img,cv2.COLOR_RGB2GRAY)
        mean = np.mean(gray)
        mean_array = np.ones_like(in_img).astype(np.float64)*mean
        in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
        in_img = np.clip(in_img,0,255).astype(np.uint8)       
        return in_img



    def zilun2zhizhen_junheng(self,predicts,gts,images,center_xs,center_ys):
        predicts_num = len(predicts)

        # zilun = predicts[0]
        # if zilun[-1]=='v':
        #     first_place = zilun[-2:]
        #     first_place = first_place.replace('v','')+random.choice(['+','-'])
        # else:
        #     first_place = zilun[-1]

        # new_predicts = predicts.copy()
        # new_predicts[0] = first_place

        # new_gts = gts[-len(predicts):]
        
        # return new_predicts[1:],new_gts[1:]
        return predicts,gts[::-1][:predicts_num][::-1],images,center_xs,center_ys


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


    def randomAugmentv2(self,im):
        im = im.copy()
        self.img_size = (64,64)
        if torch.rand(1) < 0.2:
            probility = torch.rand(1)
            if probility<0.5:
                im = self.__gaussianblur__(im, 1)
            else:
                im = self.__edgeenhance__(im)
            # cv2.imshow('before',im)
            # cv2.imshow('after',im)
            # cv2.waitKey(0)
        if torch.rand(1) < 0.2:
            im = self.__edgeenhance__(im)
        if torch.rand(1) < 0.2:
            im = self.__shift_padding__(im, [-0.2, 0.2], [-0.2, 0.2])
        if torch.rand(1) < 0.2:
            im = self.brightness_contrast(im)
        # if random.uniform(0,1) <= 0.2:
        #     high = 0.2
        #     low = 0.1
        #     ratio = np.random.uniform(0.1,0.3)
        #     random_color = np.random.randint(50,200,3).reshape(1,1,3)
        #     random_color = (random_color*ratio).astype(np.uint8)
        #     random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
        #     in_img = im.astype(np.float64)*(1-ratio) + random_color
        #     in_img = np.clip(in_img,0,255).astype(np.uint8)  
        if torch.rand(1) < 0.2:
            im = self.erase_augment(im)
        return im
