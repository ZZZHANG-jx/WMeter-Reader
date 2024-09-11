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

embedding_list = ['p',"0","1","2","3","4","5","6","7","8","9","0-","1-","2-","3-","4-","5-","6-","7-","8-","9-","0+","1+","2+","3+","4+","5+","6+","7+","8+","9+"]

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']


class TexttextLoader_lmdb_create(data.Dataset):
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

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]

        name,pointers,gts = data.split('\t')

        im_path = os.path.join('./dataset/',name)
        
        im = cv2.imread(im_path)
        if not os.path.exists(im_path):
            print(im_path)
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
            if len(predict)>2:
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
            sub_im = cv2.resize(sub_im,(96,96))
            # cv2.imshow('before',sub_im)
            # if 'train' in self.file_path:
            #     sub_im = self.randomAugment(sub_im)

            # cv2.imshow('after',sub_im)
            # cv2.waitKey(0)
            images.append(sub_im)
            center_xs.append(center_x)
            center_ys.append(center_y)

        ## 基于指针数量对全文读数进行裁减
        if len(predicts)<len(gts):
            predicts,gts,images,center_xs,center_ys = self.zilun2zhizhen_junheng(predicts,gts,images,center_xs,center_ys)




        predicts = predicts[::-1]
        images = images[::-1]
        center_xs = center_xs[::-1]
        center_ys = center_ys[::-1]
        predict_num = len(predicts)



        ## 部分全文读数最高位为0的没有标注，这里进行补0
        if predict_num>len(gts):
            gts = '0'*(predict_num-len(gts))+gts
            # gts = '0'*(predict_num-len(gts))+gts[::-1]  ## from highest to lowest


        ## 去掉最高位
        # if 'train' in self.file_path and predict_num>=8:
        #     if random.uniform(0,1)<0:
        #         rule_out_num = random.choice([4,5])
        #         predicts = predicts[:-rule_out_num]
        #         images = images[:-rule_out_num]
        #         center_xs = center_xs[:-rule_out_num]
        #         center_ys = center_ys[:-rule_out_num]
        #         gts = gts[rule_out_num:]
        
        predict_num = len(predicts)

        ## get encoder input
        if len(predicts)<12:
            predicts = ';'.join(predicts)
            predicts = predicts+';p'*(12-predict_num) ## padding
            predicts = predicts.split(';')
            images = images + [np.zeros((96,96,3))]*(12-predict_num)
            center_xs = center_xs + [0]*(12-predict_num)
            center_ys = center_ys + [0]*(12-predict_num)
        for i,predict in enumerate(predicts):
            predicts[i] = embedding_list.index(predict)  
        # src = torch.from_numpy(np.asarray(predicts))
        src = predicts
        # src_im = torch.from_numpy(np.asarray(images))
        # src_im = ((src_im/255-0.5)*2).permute(0,3,1,2)
        src_im = images
        # src_x = torch.from_numpy(np.asarray(center_xs))
        src_x = center_xs
        # src_y = torch.from_numpy(np.asarray(center_ys))
        src_y = center_ys

        # print(src.shape,src_im.shape,src_x.shape,src_y.shape)
        ## get decoder input
        # tgt = 's' + gts   # for false direction
        tgt = 's' + gts[::-1]
        tgt = tgt + 'e'
        ### padding
        if len(tgt)<14:
            tgt = tgt + 'p'*(14-len(tgt))
        tgt = list(tgt)
        for i,temp in enumerate(tgt):
            tgt[i] = type_list.index(temp)
        # tgt = torch.from_numpy(np.asarray(tgt))
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
        h,w = img.shape[:2]
        for _ in range(100):
            area = int(np.random.uniform(0.02,0.05)*h*w)
            ration = np.random.uniform(0.3,1/0.3)
            h_shift = int(round(np.sqrt(area*ration)))
            w_shift = int(round(np.sqrt(area/ration)))
            if h_shift < h and w_shift < w:
                h_start = np.random.randint(0,h-h_shift)
                w_start = np.random.randint(0,w-w_shift)
                randm_area = np.random.randint(low=0,high=255,size=(h_shift,w_shift,3))
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
        gray = cv2.cvtColor(in_img,cv2.COLOR_RGB2GRAY)
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
        self.img_size = (96,96)
        if torch.rand(1) < 0.5:
            probility = torch.rand(1)
            if probility<0.5:
                im = self.__gaussianblur__(im, 1)
            else:
                im = self.__edgeenhance__(im)

            # cv2.imshow('before',im)
            # cv2.imshow('after',im)
            # cv2.waitKey(0)
        if torch.rand(1) < 0.5:
            im = self.__edgeenhance__(im)
        if torch.rand(1) < 0.5:
            im = self.__shift_padding__(im, [-0.2, 0.2], [-0.2, 0.2])
        if torch.rand(1) < 0.5:
            im = self.brightness_contrast(im)
        if random.uniform(0,1) <= 0.5:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.img_size[0],self.img_size[1],1))
            in_img = im.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)  
        if torch.rand(1) < 0.5:
            im = self.erase_augment(im)
        return im
