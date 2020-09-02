# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:00:00 2018
Aim to implemet several feature map extraction method   
input = 
   batch_size * channels * width * height

As for the darknet-53, it concatins 1-74 layers in Darknet model
@author: kongdeqian
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from util import *
from loss import *


class EmptyNet(nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()
    def forward(x):
        return x

class Residual(nn.Module):
    def __init__(self, in_channels, out_conv1, out_conv2):
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_conv1, 1, 1, 0, bias = False),
                nn.BatchNorm2d(out_conv1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_conv1, out_conv2, 3, 1, 1, bias = False),
                nn.BatchNorm2d(out_conv2),
                nn.LeakyReLU(0.1)
                )
        self.empty = EmptyNet()
        
    def forward(self, x):
        F_x = self.residual(x)
        #print(F_x.size(),x.size())
        out = F_x + x
        return out     
        
class convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal, stride, pad):
        super(convblock, self).__init__()
        self.convblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernal, stride, pad, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                )
        
    def forward(self, x):
        x = self.convblock(x)
        return x

            
class darknet_53(nn.Module):
    def __init__(self):
        super(darknet_53, self).__init__()
        '''
        self.conv_0 = nn.Conv2d(in_channels =, out_channels = ,
                                kernel_size =, stride = ,
                                padding = )
        '''
        self.conv0 = convblock(3, 32, 3, 1, 1)
        self.conv1 = convblock(32, 64, 3, 2, 1)
        self.residual0 = Residual(64, 32, 64)
        self.conv2 = convblock(64, 128, 3, 2, 1)
        self.residual1 = Residual(128, 64, 128)
        self.conv3 = convblock(128, 256, 3, 2, 1)
        self.residual2 = Residual(256, 128, 256)
        self.conv4 = convblock(256, 512, 3, 2, 1)
        self.residual3 = Residual(512, 256, 512)
        self.conv5 = convblock(512, 1024, 3, 2, 1)
        self.residual4 = Residual(1024, 512, 1024)
        self.out = []

    # no use only for show     
    def show(self):
        module = nn.ModuleList()
        module.append(self.conv0)
        module.append(self.conv1)
        module.append(self.residual0)
        module.append(self.conv2)
        for i in range(2):
            module.append(self.residual1)      
        module.append(self.conv3)
        for i in range(8):
            module.append(self.residual2)     
        module.append(self.conv4)
        for i in range(8):
            module.append(self.residual3)     
        module.append(self.conv5)
        for i in range(4):
            module.append(self.residual4)     
        return module   
       
    def forward(self, x):
        out = []
        x = self.conv0(x); out.append(x)
        x = self.conv1(x); out.append(x)
        #print('test-----', x.size())
        x = self.residual0(x); out.append(x)
        x = self.conv2(x); out.append(x)
        for i in range(2):
            x = self.residual1(x); out.append(x)
        x = self.conv3(x); out.append(x)
        for i in range(8):
            x = self.residual2(x); out.append(x)
        x = self.conv4(x); out.append(x)
        for i in range(8):
            x = self.residual3(x); out.append(x)
        x = self.conv5(x); out.append(x)
        for i in range(4):
            x = self.residual4(x); out.append(x)
        self.out = out
        return x, out
    
    def show_layers(self):
        if len(self.out)>0:
            for i in range(len(self.out)):
                print ('darknet53 layer %d:'%(i),self.out[i].size())
        else:
            print ('Out has not initialized')
            
        
        

class DetectionLayer(nn.Module):
    def __init__(self):
        super(DetectionLayer, self).__init__()
        # Mainly based on Yolo-v3 protocol
        # DetectionLayer is used to change the format of detection results in
        # different layers in order to combine them together
    def forward(self, prediction, num_classes, anchors, input_size, CUDA):
        # x.size() = batch_size * channels * height * width
        # channels = (bbox_attributes)*num_anchors
        # Aim to change it to batch_size * (height * width * num_anchors) * bbox_attributes
        x = prediction
        batch_size = x.size(0)
        #suppose width == height
        width = x.size(3)
        height = x.size(2)
        channels = x.size(1)
        num_anchors = len(anchors)
        x = x.view(batch_size, channels, width*height)
        x = x.transpose(1,2).contiguous()
        
        #print(channels,width,height)
        x = x.view(batch_size, width*height*num_anchors, channels//num_anchors)
        # from now on, x is (N,507,85) when input is (N,255,13,13)
        # sigmoid the center point (x,y) and the confidence score
        # bx,y = sigmoid(tx,y) + cx,y
        # bw,h = exp(tw,h) * pw,h
        x[:,:,:2] = torch.sigmoid(x[:,:,:2])
        x[:,:,:4] = torch.sigmoid(x[:,:,:4])
        
        grid = np.arange(width)
        a, b = np.meshgrid(grid, grid)
        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)
    
        stride = input_size//width
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
        anchors = torch.FloatTensor(anchors)
        
        if CUDA:
            x_offset = x_offset.cuda()
            y_offset = y_offset.cuda()
            anchors = anchors.cuda()
            
        offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0) 
        x[:,:,:2] += offset
        
        anchors = anchors.repeat(width*width, 1).unsqueeze(0)
        x[:,:,2:4] = torch.exp(x[:,:,2:4]) * anchors
        
        # sigmoid the class score
        x[:,:,5: 5+num_classes] = torch.sigmoid(x[:,:,5: 5+num_classes])
        
        # resize the detection map to the input size
        x[:,:,:4] = x[:,:,:4]*stride
        return x


class AttentionBlock(nn.Module):
    #    return the same size as input x
    def __init__(self, input_size, in_channels):
        super(AttentionBlock, self).__init__()
        self.gmp = nn.MaxPool2d(kernel_size=input_size)
        self.fc1 = nn.Linear(in_channels * 2, input_size)
        self.fc2 = nn.Linear(input_size, input_size * input_size)
        self.softmax2D = nn.Softmax2d()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, feature_layer):
        #       x is high-level feature and feature is low-level feature
        #        print(x.size())
        concat = torch.cat((x, feature_layer), 1)
        concat = self.gmp(concat)
        #        print(concat.size())
        concat = concat.squeeze()
        #        print(concat.size())
        concat = F.relu(self.fc1(concat))
        concat = torch.sigmoid(self.fc2(concat))
        # concat = self.softmax(self.fc2(concat))
        #        print(concat.size())
        mask = concat.view(x.size(0), x.size(2), x.size(2))
        mask = mask.unsqueeze(1)
        #        print(mask.size())

        #        mask = self.softmax(mask)
        result = feature_layer * mask
        result = result + x
        return result
        
class Detection_attention(nn.Module):
    def __init__(self, num_anchors, num_cls):
        super(Detection_attention, self).__init__()
        self.num_anchors = num_anchors
        # input: NxCxHxW
        # 255 = (80+4+1)*3
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        self.detection = DetectionLayer()
        self.conv = nn.Conv2d(512, 256, 1, 1, 0)
        self.detect0_0 = nn.Sequential(
                convblock(1024, 512, 1, 1, 0),
                convblock(512, 1024, 3, 1, 1),
                # convblock(1024, 512, 1, 1, 0),
                # convblock(512, 1024, 3, 1, 1),
                convblock(1024, 512, 1, 1, 0),
                )
        self.detect0_1 = nn.Sequential(
                convblock(512, 1024, 3, 1, 1),
                nn.Conv2d(1024, num_anchors*(num_cls + 5), 1, 1, 0),
#                nn.ReLU(),
                # DetectionLayer(),
                )
        # 85 upsample 
        # 86 route -1 61
        # 87-95  
        self.detect1_0 = nn.Sequential(
                convblock(512, 256, 1, 1, 0),
                convblock(256, 512, 3, 1, 1),
                # convblock(512, 256, 1, 1, 0),
                # convblock(256, 512, 3, 1, 1),
                convblock(512, 256, 1, 1, 0),
                )
        self.detect1_1 = nn.Sequential(
                convblock(256, 512, 3, 1, 1),
                nn.Conv2d(512, num_anchors*(num_cls + 5), 1, 1, 0),
#                nn.ReLU(),
                # DetectionLayer(),
                )
        self.conv96 = nn.Conv2d(256, 128, 1, 1, 0)
        # 97 upsample
        # 98 route -1 36
        # 99-105 106 detectlayer
        self.detect2_0 = nn.Sequential(
                convblock(256, 128, 1, 1, 0),
                convblock(128, 256, 3, 1, 1),
                # convblock(256, 128, 1, 1, 0),
                # convblock(128, 256, 3, 1, 1),
                convblock(256, 128, 1, 1, 0),

#                nn.ReLU(),
                # DetectionLayer(),
                )
        self.detect2_1 = nn.Sequential(
                convblock(128, 256, 3, 1, 1),
                nn.Conv2d(256, num_anchors * (num_cls + 5), 1, 1, 0),
        )
        self.detect3 = nn.Sequential(
                convblock(128, 64, 1, 1, 0),
                convblock(64, 128, 3, 1, 1),
                convblock(128, 64, 1, 1, 0),
                convblock(64, 128, 3, 1, 1),
                nn.Conv2d(128, num_anchors * (num_cls + 5), 1, 1, 0),
        )
        self.detect = []
        self.attention16 = AttentionBlock(input_size = 16, in_channels = 1024)
        self.attention32 = AttentionBlock(input_size = 32, in_channels = 512)
        self.attention64 = AttentionBlock(input_size = 64, in_channels = 256)
        self.attention128 = AttentionBlock(input_size=128, in_channels = 128)


    def forward(self, x, targets, num_classes, anchors, input_size, CUDA, feature_layers, is_train = True):
#        num_route = len(feature_layers)
        out = []
        loss = []
        # deactivate the different detection anchors in different layers
        if self.num_anchors == -1:
            loss_func0 = DetectionLoss(anchors, num_classes, input_size, is_train)
            loss_func1 = loss_func0
            loss_func2 = loss_func0
        else:
            space = len(anchors)//self.num_anchors
            loss_func1 = DetectionLoss(anchors[:space], num_classes, input_size, is_train)
            loss_func2 = DetectionLoss(anchors[space:space*2], num_classes, input_size, is_train)
            loss_func3 = DetectionLoss(anchors[-1-space:-1], num_classes, input_size, is_train)
        
        x_route = self.detect0_0(x)
        x = self.detect0_1(x_route)

#        print(x.size())
        
#       detect in 16*16 without attention
#         loss0 = loss_func0(x, targets)
#         loss.append(loss0)
        
#        x = self.conv(x_route)
        x = self.upsample(x_route)
#        x = torch.cat((x, feature_layers[0]), 1)
#        print(x.size())
        x = self.attention32(x, feature_layers[0])

#        print(x.size())
        x_route = self.detect1_0(x)
        x = self.detect1_1(x_route)
        
#       detect in 32*32 with attention 
        loss1 = loss_func1(x, targets)
        loss.append(loss1)

        x = self.upsample(x_route)
#        x = torch.cat((x, feature_layers[1]), 1)
        x = self.attention64(x, feature_layers[1])
        x_route = self.detect2_0(x)
        x = self.detect2_1(x_route)
#        print(x.size())
#       detect in 64*64 with attention
        loss2 = loss_func2(x, targets)
        loss.append(loss2)

        x = self.upsample(x_route)
        x = self.attention128(x, feature_layers[2])


        x = self.detect3(x)
        loss3 = loss_func3(x, targets)
        loss.append(loss3)


        if is_train:
            return loss
        else:
            return torch.cat((loss1, loss2, loss3), 1)
        
    def show_detect(self):
        if len(self.detect)>0:
            for i in range(len(self.detect)):
                print ('detect layer %d:'%(i),self.detect[i].size())
        else:
            print ('detection has not initialized')
        
        
        
class DetectionPart(nn.Module):
    def __init__(self, num_anchors, num_cls):
        super(DetectionPart, self).__init__()
        
        # input: NxCxHxW
        # 255 = (80+4+1)*3
        #########################################################
        # need to avoid hard coding 
        #########################################################
        # REMEMBER TO CHANGE WHEN TRAINING
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        self.detection = DetectionLayer()
        self.conv84 = nn.Conv2d(512, 256, 1, 1, 0)
        # 75-83, 82 detect 83 route 
        self.detect0_0 = nn.Sequential(
                convblock(1024, 512, 1, 1, 0),
                convblock(512, 1024, 3, 1, 1),
                convblock(1024, 512, 1, 1, 0),
                convblock(512, 1024, 3, 1, 1),
                convblock(1024, 512, 1, 1, 0),
                )
        self.detect0_1 = nn.Sequential(
                convblock(512, 1024, 3, 1, 1),
                nn.Conv2d(1024, num_anchors*(num_cls + 5), 1, 1, 0),
#                nn.ReLU(),
                # DetectionLayer(),
                # route layer to fuse
                # routelayer(), -4
                )
        # 85 upsample 
        # 86 route -1 61
        # 87-95  
        self.detect1_0 = nn.Sequential(
                convblock(256+512, 256, 1, 1, 0),
                convblock(256, 512, 3, 1, 1),
                convblock(512, 256, 1, 1, 0),
                convblock(256, 512, 3, 1, 1),
                convblock(512, 256, 1, 1, 0),
                )
        self.detect1_1 = nn.Sequential(
                convblock(256, 512, 3, 1, 1),
                nn.Conv2d(512, num_anchors*(num_cls + 5), 1, 1, 0),
#                nn.ReLU(),
                # DetectionLayer(),
                #routelayer(),
                )
        self.conv96 = nn.Conv2d(256, 128, 1, 1, 0)
        # 97 upsample
        # 98 route -1 36
        # 99-105 106 detectlayer
        self.detect2 = nn.Sequential(
                convblock(128+256, 128, 1, 1, 0),
                convblock(128, 256, 3, 1, 1),
                convblock(256, 128, 1, 1, 0),
                convblock(128, 256, 3, 1, 1),
                convblock(256, 128, 1, 1, 0),
                convblock(128, 256, 3, 1, 1),
                nn.Conv2d(256, num_anchors*(num_cls + 5), 1, 1, 0),
#                nn.ReLU(),
                # DetectionLayer(),
                )
        self.detect = []

    def forward(self, x, targets, num_classes, num_anchors, anchors, input_size, CUDA, feature_layers, is_train = True):
#        num_route = len(feature_layers)
        out = []
        loss = []
        loss_func = DetectionLoss(anchors, num_classes, input_size, is_train)
        
        x_route = self.detect0_0(x)
        x = self.detect0_1(x_route)
        
        loss0 = loss_func(x, targets)
        loss.append(loss0)
        
        x = self.conv84(x_route)
        x = self.upsample(x)
        x = torch.cat((x, feature_layers[0]), 1)
        
        x_route = self.detect1_0(x)
        x = self.detect1_1(x_route)
        
        loss1 = loss_func(x, targets)
        loss.append(loss1)
        
        
        x = self.conv96(x_route)
        x = self.upsample(x)
        x = torch.cat((x, feature_layers[1]), 1)
        x = self.detect2(x)

        loss2 = loss_func(x, targets)
        loss.append(loss2)
            
        if is_train:
            return loss
        else:
            return torch.cat((loss0, loss1, loss2), 1)
        
    def show_detect(self):
        if len(self.detect)>0:
            for i in range(len(self.detect)):
                print ('detect layer %d:'%(i),self.detect[i].size())
        else:
            print ('detection has not initialized')        




       
if __name__ == '__main__':

    img = cv2.imread("0339.jpg")
    img = cv2.resize(img, (512,512))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  
#    img_ = img_.cuda()
#    
#    darknet_53 = darknet_53()
#    darknet_53 = darknet_53.cuda()
#    m, out = darknet_53(img_)
#    darknet_53.show_layers()
#    print(m.size())

    a =AttentionBlock(512,3)
    out = a(img_, img_)
    print(out.size())
#    print(img_.size())
#    a = nn.MaxPool2d(512)
#    out = a(img_)
#    print(out.size())

    
    
    '''
    detection = DetectionPart()
    detection = detection.cuda()
    
    anchors=[(10,13),(15,20),[28,40]]
    feature_layers = [out[23],out[14]]
    
    num_anchors = 3
    num_classes = 11
    target = torch.rand(1,5,5)
    target[0,0,0] = 0
    target[0,1,0] = 8
    target[0,2:,:] = 0
    target = target.cuda()
    loss = detection(m, num_classes, anchors, 1024, True, feature_layers, target, is_train = True)
    '''
    
    
    
'''
appendix: input : [512,512,3]
    
darknet53 layer 0: torch.Size([1, 32, 512, 512])
darknet53 layer 1: torch.Size([1, 64, 256, 256])
darknet53 layer 2: torch.Size([1, 64, 256, 256])
darknet53 layer 3: torch.Size([1, 128, 128, 128])
darknet53 layer 4: torch.Size([1, 128, 128, 128])
darknet53 layer 5: torch.Size([1, 128, 128, 128])
darknet53 layer 6: torch.Size([1, 256, 64, 64])
darknet53 layer 7: torch.Size([1, 256, 64, 64])
darknet53 layer 8: torch.Size([1, 256, 64, 64])
darknet53 layer 9: torch.Size([1, 256, 64, 64])
darknet53 layer 10: torch.Size([1, 256, 64, 64])
darknet53 layer 11: torch.Size([1, 256, 64, 64])
darknet53 layer 12: torch.Size([1, 256, 64, 64])
darknet53 layer 13: torch.Size([1, 256, 64, 64])
darknet53 layer 14: torch.Size([1, 256, 64, 64])
darknet53 layer 15: torch.Size([1, 512, 32, 32])
darknet53 layer 16: torch.Size([1, 512, 32, 32])
darknet53 layer 17: torch.Size([1, 512, 32, 32])
darknet53 layer 18: torch.Size([1, 512, 32, 32])
darknet53 layer 19: torch.Size([1, 512, 32, 32])
darknet53 layer 20: torch.Size([1, 512, 32, 32])
darknet53 layer 21: torch.Size([1, 512, 32, 32])
darknet53 layer 22: torch.Size([1, 512, 32, 32])
darknet53 layer 23: torch.Size([1, 512, 32, 32])
darknet53 layer 24: torch.Size([1, 1024, 16, 16])
darknet53 layer 25: torch.Size([1, 1024, 16, 16])
darknet53 layer 26: torch.Size([1, 1024, 16, 16])
darknet53 layer 27: torch.Size([1, 1024, 16, 16])
darknet53 layer 28: torch.Size([1, 1024, 16, 16])    
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
