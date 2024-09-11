from math import log
from re import X
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import functools
# from cbam import CBAM
import torch.nn.functional as F
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck


class Net_64(nn.Module):
    def __init__(self, imgH=64, nc=1, nclass=10):
        super(Net_64,self).__init__()
        # input: 1*64*64
        self.conv1 = nn.Conv2d(nc, 64, 3, stride=1, padding=1)  # 64*64*64
        self.pool1 = nn.MaxPool2d(2, 2)  # 64*32*32

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 64*32*32
        self.pool2 = nn.MaxPool2d(2, 2)  # 64*16*16

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 128*16*16
        self.pool3 = nn.MaxPool2d(2, 2)  # 128*8*8

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  # 128*8*8
        self.pool4 = nn.MaxPool2d(2, 2)  # 128*4*4

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 256*4*4
        self.pool5 = nn.MaxPool2d(2, 2)  # 256*4*4

        self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)  # 512*2*2
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2, 2)  # 512*1*1

        self.fc1 = nn.Linear(512, 384)
        self.fc2 = nn.Linear(384, 30)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self,input):

        # with torch.no_grad():
        b,l,c,h,w = input.shape[:]
        input = input.view(b*l,c,h,w)
        input = F.interpolate(input,(64,64))

        x = self.conv1(input)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool6(x)

        x = x.view(-1, 512).detach()

        fc1 = self.fc1(x)
        fc1_relu_drop = self.dropout(self.relu(fc1))
        out = self.fc2(fc1_relu_drop)
        out = out.view(b,l,-1)
        out = torch.argmax(out,dim=-1)+1


        return x

class Net_64_distribution(nn.Module):
    def __init__(self, imgH=64, nc=1, nclass=10):
        super(Net_64_distribution,self).__init__()
        # input: 1*64*64
        self.conv1 = nn.Conv2d(nc, 64, 3, stride=1, padding=1)  # 64*64*64
        self.pool1 = nn.MaxPool2d(2, 2)  # 64*32*32

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 64*32*32
        self.pool2 = nn.MaxPool2d(2, 2)  # 64*16*16

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 128*16*16
        self.pool3 = nn.MaxPool2d(2, 2)  # 128*8*8

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  # 128*8*8
        self.pool4 = nn.MaxPool2d(2, 2)  # 128*4*4

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 256*4*4
        self.pool5 = nn.MaxPool2d(2, 2)  # 256*4*4

        self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1)  # 512*2*2
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(2, 2)  # 512*1*1

        self.fc1 = nn.Linear(512, 384)
        self.fc2 = nn.Linear(384, 30)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self,input):

        # with torch.no_grad():
        b,l,c,h,w = input.shape[:]
        input = input.view(b*l,c,h,w)
        input = F.interpolate(input,(64,64))

        x = self.conv1(input)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool6(x)

        x = x.view(-1, 512).detach()

        fc1 = self.fc1(x)
        fc1_relu_drop = self.dropout(self.relu(fc1))
        out = self.fc2(fc1_relu_drop)
        out = out.view(b,l,-1)
        out = F.softmax(out,dim=-1)
        # out = torch.argmax(out,dim=-1)+1

        return x,out.detach()