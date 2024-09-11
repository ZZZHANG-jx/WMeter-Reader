import cv2 
import os 
import numpy as np
import math
from math import *
from tqdm import tqdm
import cv2 
import numpy as np

def rotate(img, pt1, pt2, pt3, pt4):
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  

    if pt4[1]>pt1[1]:
        pass
    else:
        angle=-angle

    height = img.shape[0]  
    width = img.shape[1]   
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))

    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    if pt2[1]>pt4[1]:
        pt2[1],pt4[1]=pt4[1],pt2[1]
    if pt1[0]>pt3[0]:
        pt1[0],pt3[0]=pt3[0],pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    return imgOut






type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']
with open('label.txt','r') as f:
    corr_num = 0
    half_corr_num = 0
    total_num = 0
    datas = f.readlines()
    for data in tqdm(datas):
        im_name,pointers,gt = data.split('\t')
        gt = gt.replace('\n','').replace('.','')
        pointers = pointers.replace('\n','').split(';')
        im_path = im_name
        im = cv2.imread(im_path)
        show_im = im.copy()
        h, w = im.shape[:2]
        predicts = []
        pointer_num = 0
        sub_ims = []
        for pointer in pointers:
            if not '/' in pointer:
                continue
            cood, predict = pointer.split('/')
            predicts.append(predict)
            cood = cood.split(',')
            x0 = min(max(int(cood[0]),0),w)
            y0 = min(max(int(cood[1]),0),h)
            x1 = min(max(int(cood[2]),0),w)
            y1 = min(max(int(cood[3]),0),h)
            x2 = min(max(int(cood[4]),0),w)
            y2 = min(max(int(cood[5]),0),h)
            x3 = min(max(int(cood[6]),0),w)
            y3 = min(max(int(cood[7]),0),h)
            sub_im = rotate(im,[x0,y0],[x3,y3],[x2, y2],[x1,y1])
            sub_im = cv2.resize(sub_im,(48,48))
            sub_ims.append(sub_im)
            cv2.line(show_im,(x0,y0),(x1,y1),color=(255,0,0),thickness=2)
            cv2.line(show_im,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
            cv2.line(show_im,(x2,y2),(x3,y3),color=(0,255,0),thickness=2)
            cv2.line(show_im,(x3,y3),(x0,y0),color=(255,255,0),thickness=2)

        for i, (predict,sub_im) in enumerate(zip(predicts,sub_ims)):
            sub_im_path = os.path.join('pointers','crop_'+str(i)+'_'+predict+'.jpg')
            cv2.imwrite(sub_im_path,sub_im)

