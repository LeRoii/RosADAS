#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:07:51 2018

@author: kongdeqian
"""

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim

from datasets import *
from myNet import *
from test import *
from myTansforms import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='0', help='multi-gpus')
opt = parser.parse_args()
gpus = opt.gpus
ngpus = len(gpus.split(','))
print('device_id:'+gpus)
os.makedirs('weight', exist_ok=True)



# Get data configuration
train_path = '/home/kongdeqian/dKong/Dataset/TLdataset_train/list.txt'
valid_path = '/home/kongdeqian/dKong/Dataset/TLdataset_test/list.txt'

# Get hyper parameters
n_cpu        = 8
momentum     = 0.9
decay        = 0.00005
learning_rate= 0.00005
num_classes  = 11
num_anchors  = 3
input_size   = 512
# anchors 4 bosch data
# anchors      = [ (21, 59), (26, 69), (37, 90), (12, 35), (15, 40), (18, 52), (4, 12), (6, 18), (9, 26)]
# anchors 4 Lisa data
#anchors      = [(15, 23), (25, 40), (32, 50), (35, 65), (47, 66), (49, 87), (59, 103), (60, 84), (87, 136)]
# achors 4 IAIR
anchors      = [(54, 126), (135, 55), (174, 67), (64, 26), (25, 61), (96, 39), (38, 15), (15, 36), (37, 91)]

batch_size   = 6
epochs       = 300
checkpoint_interval = 5

# Initiate model
model = myNet_Res101(num_classes, num_anchors, anchors, input_size, True, is_train = True)
model.load_state_dict(torch.load('./0913_final_param.pkl', map_location='cuda:0'), False)
# model = torch.load('model_0913_34.pkl', map_location= 'cuda:0')
model.set_train()

use_cuda = torch.cuda.is_available()

if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.set_device(int(gpus))

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDatasetraw(train_path, input_size),
    batch_size=batch_size, shuffle=False, num_workers=n_cpu)

# Get dataloader
# transform = my_transform(crops_size=(480, 854), image_size=(720, 1280), max_boxes_per_img=10)
# Get dataloader
# dataloader = torch.utils.data.DataLoader(
#     ListDataset(train_path, input_size,
#                 transform=transform.image_transform_5crops(),
#                 transform2=transform.image_transforms_original_image(),
#                 target_transform=transform.target_transform_crops5,
#                 target_transform2=transform.target_transform_org),
#     batch_size=batch_size, shuffle=False, num_workers=n_cpu)

Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

optimizer0 = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                       weight_decay=decay)
optimizer1 = optim.SGD(model.parameters(), lr=learning_rate / (2 * batch_size), momentum=momentum, dampening=0,
                       weight_decay=decay)

for epoch in range(epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        if epoch < 30:
            optimizer0.zero_grad()
        else:
            optimizer1.zero_grad()

        if use_cuda:
            imgs, targets = imgs.cuda(), targets.cuda()

        loss = model(imgs, targets)
        loss_back = loss[0][0] + loss[1][0] + loss[2][0]

        loss_x = loss[0][1] + loss[1][1] + loss[2][1]
        loss_y = loss[0][2] + loss[1][2] + loss[2][2]
        loss_w = loss[0][3] + loss[1][3] + loss[2][3]
        loss_h = loss[0][4] + loss[1][4] + loss[2][4]
        loss_conf = loss[0][5] + loss[1][5] + loss[2][5]
        loss_noobj = loss[0][6] + loss[1][6] + loss[2][6]
        loss_cls = loss[0][7] + loss[1][7] + loss[2][7]
        recall = loss[0][8] + loss[1][8] + loss[2][8]

        loss_back.backward()
        if epoch < 30:
            optimizer0.step()
        else:
            optimizer1.step()

        print(
            '[Epoch %d/%d, Batch %d/%d]\n[Losses: x %f, y %f, w %f, h %f, conf %f, noobj %f, cls %f, total %f, recall: %.5f]' %
            (epoch, epochs, batch_i, len(dataloader),
             loss_x, loss_y, loss_w, loss_h, loss_conf, loss_noobj, loss_cls,
             loss_back.item(),
             recall / 3.0))

    torch.save(model.state_dict(), './weight/0913_final_param.pkl')
    if (epoch+1) % checkpoint_interval == 0:
        torch.save(model, './weight/model_0913_%d.pkl' % (epoch))
        model.set_test()
        iou_thres = 0.5
        test_batch = 20
        test_all(model, valid_path, epoch, test_batch, num_classes, n_cpu, iou_thres, input_size, use_cuda)
        model.set_train()

torch.save(model, './weight/m_final.pkl')
