import torch

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import random

from models import get_model
from utils import convert_state_dict
import utils
from math import *
import math
import Levenshtein

embedding_list = ['p',"0","1","2","3","4","5","6","7","8","9","0-","1-","2-","3-","4-","5-","6-","7-","8-","9-","0+","1+","2+","3+","4+","5+","6+","7+","8+","9+"]

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def zilun2zhizhen_junheng(predicts,gts,images,center_xs,center_ys):
    predicts_num = len(predicts)
    return predicts,gts[::-1][:predicts_num][::-1],images,center_xs,center_ys

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

def val(model,model_im,file_path):

    mape = []
    ar = []
    acc = []
    folder = os.path.split(os.path.split(file_path)[0])[-1]
    f = open(file_path,'r')
    datas = f.readlines()
    f.close()
    for data in tqdm(datas):
        name_new,pointers,gts = data.split('\t')
        im_path = os.path.join('./dataset/',name_new)
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
            if len(predict) >2:
                continue
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
            sub_im = rotate(im,[x0,y0],[x3,y3],[x2, y2],[x1,y1])
            sub_im = cv2.resize(sub_im,(64,64))
            sub_im = cv2.cvtColor(sub_im,cv2.COLOR_RGB2GRAY)
            images.append(sub_im)
            center_xs.append(center_x)
            center_ys.append(center_y)


        if len(predicts)<len(gts):
            predicts,gts,images,center_xs,center_ys = zilun2zhizhen_junheng(predicts,gts,images,center_xs,center_ys)


        ## truncating the pointers
        # pointer_length = 2
        # if len(images)==2:
        #     print(file_path)
        # if len(images)<pointer_length:
        #     continue
        # predicts = predicts[::-1][:pointer_length]
        # # images = images[::-1][:pointer_length]
        # images = images[::-1][:pointer_length][::-1]  ## wrong indexing order
        # center_xs = center_xs[::-1][:pointer_length]
        # center_ys = center_ys[::-1][:pointer_length]
        # gts = gts[::-1][:pointer_length][::-1]
        # predict_num = len(predicts)
        ## did not truncate
        predicts = predicts[::-1]
        images = images[::-1]
        center_xs = center_xs[::-1]
        center_ys = center_ys[::-1]
        predict_num = len(predicts)



        ## fill with 0
        if predict_num!=len(gts):
            gts = '0'*(predict_num-len(gts))+gts 



        ## 去掉最高位
        # rule_out_num = 0
        # predicts = predicts[:-rule_out_num]
        # images = images[:-rule_out_num]
        # center_xs = center_xs[:-rule_out_num]
        # center_ys = center_ys[:-rule_out_num]
        # gts = gts[rule_out_num:]

        predict_num = len(predicts)
        ## get encoder input
        if len(predicts)<12:
            predicts = ';'.join(predicts)
            predicts = predicts+';p'*(12-predict_num) ## padding
            predicts = predicts.split(';')
            images = images + [np.zeros((64,64))]*(12-predict_num)
            center_xs = center_xs + [0]*(12-predict_num)
            center_ys = center_ys + [0]*(12-predict_num)
        for i,predict in enumerate(predicts):
            predicts[i] = embedding_list.index(predict)  
        src = torch.from_numpy(np.asarray(predicts)).unsqueeze(0)
        src_im = torch.from_numpy(np.asarray(images)).unsqueeze(-1)
        src_im = ((src_im/255-0.5)*2).permute(0,3,1,2).unsqueeze(0)
        src_x = torch.from_numpy(np.asarray(center_xs)).unsqueeze(0)
        src_y = torch.from_numpy(np.asarray(center_ys)).unsqueeze(0)

        ## get decoder input
        tgt = 's' + gts[::-1]
        tgt = tgt + 'e'
        ### padding
        if len(tgt)<14:
            tgt = tgt + 'p'*(14-len(tgt))
        tgt = list(tgt)
        for i,temp in enumerate(tgt):
            tgt[i] = type_list.index(temp)
        tgt = torch.from_numpy(np.asarray(tgt)).unsqueeze(0)

        # predict
        model.to(DEVICE)
        model.eval()
        model_im.to(DEVICE)
        model_im.eval()
        with torch.no_grad():
            pred = [11]
            im_embedding,distribution = model_im(src_im.float().cuda())
            src = src.permute(1,0).cuda()
            src_x = src_x.permute(1,0).cuda()
            src_y = src_y.permute(1,0).cuda()
            for j in range(13):
                pred_temp = torch.LongTensor(pred).unsqueeze(0)
                pred_temp = pred_temp.permute(1,0).cuda()
                output_temp = model(src,distribution,pred_temp,im_embedding,src_x,src_y)    # src (12,1) distribution(1,12,30),pred_temp(1,1), im_embedding(12,512)
                out_num = output_temp.argmax(2)[-1].item()
                pred.append(out_num)

            pred_temp = output_temp.argmax(2).reshape(1,-1)
        
        pred = pred[1:][::-1]
        results = []
        for i in pred:
            if type_list[i]=='e' or type_list[i]=='p':
                continue
            results.append(type_list[i])
        results = ''.join(results[::-1][:predict_num][::-1])

        predict_src = torch.argmax(distribution,dim=-1)
        predict_src = predict_src[0].cpu().numpy()+1
        predict_src_list = []
        for temp in predict_src:
            predict_src_list.append(embedding_list[int(temp)])
        predict_src_list = ' '.join(predict_src_list[:predict_num][::-1])


        if results[:-1]==gts[:-1]:
            pass
        if results==gts:
            acc.append(1)
        else:
            acc.append(0)

        try:
            mape.append(abs(int(gts)-int(results))/((int(gts)+int(results))*2))
        except:
            continue
        ar.append(1-Levenshtein.distance(gts,results)/len(gts))
        
        ## for result visualize
        # cv2.putText(im,'pre_src :'+predict_src_list,(0,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        # cv2.putText(im,'pre_final:'+results,(0,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
        # cv2.putText(im,'gt  :'+gts,(0,h-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)
        # if results[:-1]!=gts[:-1]:
        #     new_path = os.path.join('./rule_based_method/ours',os.path.split(im_path)[-1])
            # cv2.imwrite(new_path,im)
            
    print(f'dataset {folder}: acc:{np.mean(acc)}; ar: {np.mean(ar)}; mape: {np.mean(mape)}')

def test_im_sequence(model,model_im,ims):
    num = len(ims)
    new_ims = []
    for im in ims[::-1]:
        im = cv2.resize(im,(64,64))
        im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        new_ims.append(im)
    src_im = torch.from_numpy(np.asarray(new_ims)).unsqueeze(-1)
    src_im = ((src_im/255-0.5)*2).permute(0,3,1,2).unsqueeze(0)    

    model.to(DEVICE)
    model.eval()
    model_im.to(DEVICE)
    model_im.eval()
    with torch.no_grad():
        pred = [11]
        im_embedding,distribution = model_im(src_im.float().cuda())
        src = torch.ones(num,1).cuda()
        for i in range(num):
            pred_temp = torch.LongTensor(pred).unsqueeze(0)
            pred_temp = pred_temp.permute(1,0).cuda()
            output_temp = model(src,distribution,pred_temp,im_embedding)    # src 
            out_num = output_temp.argmax(2)[-1].item()
            pred.append(out_num)

        pred_temp = output_temp.argmax(2).reshape(1,-1)

    pred = pred[1:][::-1]
    results = []
    for i in pred:
        if type_list[i]=='e' or type_list[i]=='p':
            continue
        results.append(type_list[i])
    results = ''.join(results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')


    parser.add_argument('--model_path', nargs='?', type=str, default='./checkpoints/tr.pkl',help='Path to the saved wc model')
    parser.add_argument('--model_path_im', nargs='?', type=str, default='./checkpoints/im.pkl',help='Path to the saved wc model')
    parser.add_argument('--pointer_folder', nargs='?', type=str, default='./demo/pointers/',help='Path of the input pointers')
    parser.add_argument('--mode', nargs='?', type=str, default='test',
                        help='test or val')
    args = parser.parse_args()

    # prepare model
    model = get_model('transformer_im_distribution').cuda()
    model_im = get_model('cnn_extractor_pretrain_distribution').cuda()

    if DEVICE.type == 'cpu':
        state = convert_state_dict(torch.load(args.model_path, map_location='cpu')['model_state'])
    else:
        checkpoint = torch.load(args.model_path)['model_state']
    model.load_state_dict(checkpoint)

    if DEVICE.type == 'cpu':
        state = convert_state_dict(torch.load(args.model_path_im, map_location='cpu')['model_state'])
    else:
        checkpoint = torch.load(args.model_path_im)['model_state']
    model_im.load_state_dict(checkpoint)

    ## val
    if args.mode == 'val':
        val(model,model_im,file_path='./dataset/WMeter5K/label_val.txt')

    ## test
    if args.mode == 'test':
        pointer_paths = glob.glob(os.path.join(args.pointer_folder,'*'))
        pointer_paths.sort()
        pointers = [cv2.imread(pointer_path) for pointer_path in pointer_paths]
        result = test_im_sequence(model,model_im,pointers)
        print('predicted result:', result)