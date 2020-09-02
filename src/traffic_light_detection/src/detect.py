#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:52:29 2018
@author: kongdeqian
"""


import os
import time
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
import random
from datasets import ImageFolder
from myNet import *
from util import *

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# for cameras
import PyCapture2
import datetime
from UDP_data_transform import *
from collections import deque

# Get hyper parameters
num_classes  = 11
anchors      = [(38, 15), (15, 36), (174, 67), (37, 91), (64, 26), (54, 126), (25, 61), (96, 39), (135, 55)]
input_size   = 512
batch_size   = 1
image_folder = '/media/iaircv/SSD/ChangShu/test'
save_path = '/home/iaircv/1111'
img_size     = 512
# most important threshold
#------confidence threshold-----------
conf_thresh = 0.5
#------NMS threshold-------
NMS_thresh = 0.05
#------classification threshold-------
cls_thresh = 0.4

detect_queue = deque([[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                      [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                      [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]])

# Get Dataloader using ImageFolder
os.makedirs(save_path, exist_ok = True)
dataloader = DataLoader(ImageFolder(image_folder, img_size=img_size),
                        batch_size= batch_size, shuffle=False, num_workers=8)

# Extracts class labels from file
classes = load_classes('tl.names')

# Initiate model
#model = myNet(num_classes, anchors, input_size, True, is_train = True).cuda()
#model.load_state_dict(torch.load('./weight/final_param.pkl'),False)
model = torch.load('./model_0914_17.pkl', map_location = 'cuda:0')
model.set_test()

cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

# First implement detection
print ('\nPerforming object detection:')
mean_time = 0
prev_time = time.time()
total_num = 1
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    print(dataloader.__len__())
    # Get detections
    input_imgs = input_imgs.cuda()
    detections = model(input_imgs, None)
    detections = non_max_suppression(detections, num_classes, conf_thres = conf_thresh, nms_thres = NMS_thresh, cls_dependent = False)
    output = UDP_data_transform_queue(detect_queue, detections[0], img_size, cls_thresh)
    # Calculate the time
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    mean_time = mean_time + float(current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
    mean_time = mean_time + float(current_time - prev_time)

    total_num = batch_i + 1
    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)
print ('\nMean Inference Time: %s' % ( mean_time/total_num))
print ('\nTotal Inference Time: %s' % ( mean_time))


# Perform plotting
# Bounding-box colors
color = ['orangered', 'orangered', 'orangered', 'orangered', 'orangered', 'g', 'g', 'g', 'g', 'g', 'mediumslateblue']
print ('\nSaving images:')
# Iterate through images and save plot of detections
detect_queue = deque([[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1]])

for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print ("(%d) Image: '%s'" % (img_i, path))
    # Create plot
    img = np.array(Image.open(path))
    plt.figure(figsize=(6.8,4),dpi=100)
    fig, ax = plt.subplots()
    fig.set_size_inches(6.8, 4)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw bounding boxes and labels of detections
    detect_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if detections is None:
        detect_queue.popleft()
        detect_queue.append(detect_data)
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_pred > cls_thresh:

                print ('\t+ Label: %s, Cls_Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                detect_data[int(cls_pred)] = 1
                if detect_queue[2][int(cls_pred)] == 0:
                    continue
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]


                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color[int(cls_pred)],
                                        facecolor='none')

                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)]+' %.2f' % (float(cls_conf.cpu())), color='white', verticalalignment='bottom',
                         bbox={'color': color[int(cls_pred)], 'pad': 0}, fontsize = 'xx-small')
        detect_queue.popleft()
        detect_queue.append(detect_data)
        for i in range(3):
            print(detect_queue[i])
    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    #plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.savefig(save_path + '/%d.png' % (img_i), pad_inches=0.0)
    plt.close()
