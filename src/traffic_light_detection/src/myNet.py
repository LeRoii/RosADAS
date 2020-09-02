#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:23:22 2018
myNet_Res101 is our final results
@author: kongdeqian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from Modules import *
from loss import *
from util import *
import Resnet


class myNet(nn.Module):
    def __init__(self, num_classes, anchors, input_size, CUDA, is_train = True):
        super(myNet, self).__init__()
        self.darknet_53 = darknet_53()
        self.lstm = Resnet_LSTM(16*16*257, 16*16, 3, 1, 16)
        self.conv0 = nn.Conv2d(256,256,8,4,2)
        self.conv1 = nn.Conv2d(512,256,3,2,1)
        self.conv2 = nn.Conv2d(1024,256,1,1,0)
        self.detection = DetectionPart(len(anchors), num_classes)
        
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size
        self.CUDA = CUDA
        self.is_train = is_train
        
    def forward(self, x, target):
        x, output = self.darknet_53(x)
        feature_layers = [output[23], output[14]]
        #print(output[14].size())
        l1 = self.conv0(output[14])
        #print(l1.size())
        l2 = self.conv1(output[23])
        l3 = self.conv2(output[28])        
        lstm = self.lstm(l1, l2, l3, True)
        
        x = lstm * x
        
        loss = self.detection( x, target, self.num_classes, self.anchors, self.input_size, self.CUDA, feature_layers, self.is_train)
#        loss = self.detection( x, target, self.num_classes, self.anchors, self.input_size, self.CUDA, feature_layers, False)
        return loss
    
    def set_test(self):
        self.is_train = False
        
    def set_train(self):
        self.is_train = True


class myNet_Res50(nn.Module):
#    backbone: ResNet50 with pre-trained on ImgNet
    def __init__(self, num_classes, anchors, input_size, CUDA, is_train = True):
        super(myNet_Res50, self).__init__()
        self.res50 = Resnet.resnet50(pretrained=True)
        self.trans0 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.trans1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.trans2 = nn.Conv2d(512, 256, 1, 1, 0)

        self.conv0 = nn.Conv2d(256,256,8,4,2)
        self.conv1 = nn.Conv2d(512,256,3,2,1)
        self.conv2 = nn.Conv2d(1024,256,1,1,0)
        self.detection = DetectionPart(len(anchors), num_classes)
        
        self.num_classes = num_classes
        self.anchors = anchors
        self.input_size = input_size
        self.CUDA = CUDA
        self.is_train = is_train
    def forward(self, x, target):
        x, output = self.res50(x)
        f0 = self.trans0(x)

        f1 = self.trans1(output[3])
#        print(f1.size())
        f2 = self.trans2(output[2])
        feature_layers = [f1, f2]
        l1 = self.conv0(f2)
        l2 = self.conv1(f1)
        l3 = self.conv2(f0)        
        lstm = self.lstm(l1, l2, l3, True)
        
        x = f0
        x = lstm * x
#        print(x.size())
        feature_0 = [f1.new_zeros(f1.size()), f2.new_zeros(f2.size())]
        loss = self.detection( f0, target, self.num_classes, self.anchors, self.input_size, self.CUDA, feature_0, self.is_train)
#        loss = self.detection( x, target, self.num_classes, self.anchors, self.input_size, self.CUDA, feature_layers, False)
        return loss    
    
    def set_test(self):
        self.is_train = False
        
    def set_train(self):
        self.is_train = True
        
class myNet_Res101(nn.Module):
#    backbone: ResNet101 with pre-trained on ImgNet
    def __init__(self, num_cls, num_anchors, anchors, input_size, CUDA, is_train = True):
        super(myNet_Res101, self).__init__()
        self.res101 = Resnet.resnet101(pretrained=True)
        self.trans0 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.trans1 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.trans2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.trans3 = nn.Conv2d(256, 128, 1, 1, 0)
        
        self.detection = Detection_attention(num_anchors, num_cls)
        self.num_cls = num_cls
        self.anchors = anchors
        self.input_size = input_size
        self.CUDA = CUDA
        self.is_train = is_train

    def forward(self, x, target):
        if len(x.size()) > 4:
            batch_size = x.size(0)*x.size(1)
            x = x.view(batch_size, x.size(2), x.size(3), x.size(4))
            target = target.view(batch_size, target.size(2), target.size(3))

        x, output = self.res101(x)
        # f0 1024 16 16
        # f1 512 32 32
        # f2 256 64 64
        # f3 128 128 128
        f0 = self.trans0(x)           
        f1 = self.trans1(output[3])
        f2 = self.trans2(output[2])
        f3 = self.trans3(output[1])
        feature_layers = [f1, f2, f3]

        loss = self.detection(f0, target, self.num_cls, self.anchors, self.input_size, self.CUDA, feature_layers, self.is_train)
        return loss   
    
    def set_test(self):
        self.is_train = False
        
    def set_train(self):
        self.is_train = True
