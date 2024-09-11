import cv2 
import os 
import numpy as np
from tqdm import tqdm 
        
pointer_list_transform = {'0+':'0_1','0':'0_2','1-':'0_3','1+':'1_1','1':'1_2','2-':'1_3','2+':'2_1','2':'2_2','3-':'2_3','3+':'3_1','3':'3_2','4-':'3_3','4+':'4_1','4':'4_2','5-':'4_3','5+':'5_1','5':'5_2','6-':'5_3','6+':'6_1','6':'6_2','7-':'6_3','7+':'7_1','7':'7_2','8-':'7_3','8+':'8_1','8':'8_2','9-':'8_3','9+':'9_1','9':'9_2','0-':'9_3'}



with open('./WMeter5K/label.txt','r') as f:
    datas = f.readlines()
    for i, data in tqdm(enumerate(datas)):
        im_name,pointers,final_readings = data.replace('\n','').split('\t')
        
        im_name = os.path.split(im_name)[-1]
        im_path = os.path.join('./WMeter5K/images/',im_name)
        im = cv2.imread(im_path)
        scale = 1
        im = cv2.resize(im,(0,0),fx=scale,fy=scale)
        new_im_path = im_path.replace('images/','visualize/')
        pointers = pointers.split(';')
        for pointer in pointers:
            cood, predict = pointer.split('/')
            if len(predict)>2:
                color = (255,0,0)
                color1 = (0,255,0)
                color2 = (0,0,255)
                color3 = (255,255,0)
                cood = cood.split(',')
                x0 = int(cood[0])*scale
                y0 = int(cood[1])*scale
                x1 = int(cood[2])*scale
                y1 = int(cood[3])*scale
                x2 = int(cood[4])*scale
                y2 = int(cood[5])*scale
                x3 = int(cood[6])*scale
                y3 = int(cood[7])*scale    
                cv2.line(im,(x0,y0),(x1,y1),color=color,thickness=2)
                cv2.line(im,(x1,y1),(x2,y2),color=color,thickness=2)
                cv2.line(im,(x2,y2),(x3,y3),color=color,thickness=2)
                cv2.line(im,(x3,y3),(x0,y0),color=color,thickness=2)
                predict = '-'.join(list(predict)).strip()
                predict = predict.replace('-v','v')
                cv2.putText(im,predict,(x0,y0-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            else:
                predict = pointer_list_transform[predict]
                color = (255,0,0)
                color1 = (0,255,0)
                color2 = (0,0,255)
                color3 = (255,255,0)
                cood = cood.split(',')
                x0 = int(cood[0])*scale
                y0 = int(cood[1])*scale
                x1 = int(cood[2])*scale
                y1 = int(cood[3])*scale
                x2 = int(cood[4])*scale
                y2 = int(cood[5])*scale
                x3 = int(cood[6])*scale
                y3 = int(cood[7])*scale    
                cv2.line(im,(x0,y0),(x1,y1),color=color2,thickness=2)
                cv2.line(im,(x1,y1),(x2,y2),color=color2,thickness=2)
                cv2.line(im,(x2,y2),(x3,y3),color=color2,thickness=2)
                cv2.line(im,(x3,y3),(x0,y0),color=color2,thickness=2)
                cv2.putText(im,predict,(x2+2,y2),cv2.FONT_HERSHEY_SIMPLEX,0.7,color2,2)

        cv2.putText(im,final_readings,(4,im.shape[0]-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imwrite(new_im_path,im)
        