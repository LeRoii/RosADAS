#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:25:17 2018
test_all means test all data include Empty ones
@author: kongdeqian
"""
import os
import sys
import torch
from torch.utils.data import DataLoader
from myNet import *
from myTansforms import *

from torchvision import datasets
from torchvision import transforms

from datasets import ImageFolder, ListDataset, ListDatasetraw
from util import non_max_suppression, load_classes, bbox_ious, compute_ap
import numpy as np

def test_all(model, valid_path, epoch, batch_size, num_classes=11, n_cpu=8, iou_thres=0.5, img_size=512, cuda=True):
    # Get dataloader
    # when training the data without multi-scale
    dataset = ListDatasetraw(valid_path, img_size)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    # when training the data with multi-scale
    # transform_test = my_transform(image_size=(1200, 1920))
    # dataset = ListDataset(valid_path, img_size, transform=transform_test.image_transforms_original_image(),
    #                       target_transform=transform_test.target_transform_org)
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    # get hyper-params
    conf_thresh = 0.9
    NMS_thresh = 0.1
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    file_name = './test_log.txt'
    f = open(file_name, 'a')
    f.write('Epoch%d Compute mAP...\n' % (epoch))
    f.write('IoU threshold: %.4f\n' % (iou_thres))
    print('Compute mAP...')

    targets = None
    APs = []
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        targets = targets.type(Tensor)
        imgs = imgs.type(Tensor)

        with torch.no_grad():
            output = model(imgs, None)
            output = non_max_suppression(output, num_classes, conf_thres=conf_thresh, nms_thres=NMS_thresh, cls_dependent=False)

        # Compute average precision for each sample
        for sample_i in range(targets.size(0)):
            correct = []

            # Get labels for sample where width is not zero (dummies)
            annotations = targets[sample_i, targets[sample_i, :, 3] !=0]
            # Extract detections
            detections = output[sample_i]

            if detections is None:
                if annotations.size(0) == 0:
                    continue
                    APs.append(1)
                    print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), 1, np.mean(APs)))
                # If there are no detections but there are annotations mask as zero AP
                if annotations.size(0) != 0:
                    APs.append(0)
                    print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), 0, np.mean(APs)))
                continue

            if detections.size(0) == 0:
                if annotations.size(0) == 0:
                    APs.append(1)
                    print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), 1, np.mean(APs)))
                # If there are no detections but there are annotations mask as zero AP
                if annotations.size(0) != 0:
                    APs.append(0)
                    print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), 0, np.mean(APs)))
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[np.argsort(-detections[:, 4])]

            # If no annotations add number of detections as incorrect
            if annotations.size(0) == 0:
                correct.extend([0 for _ in range(len(detections))])
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
                target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
                target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
                target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
                target_boxes *= img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_ious(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Extract true and false positives
            true_positives = np.array(correct)
            false_positives = 1 - true_positives

            # Compute cumulative false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # Compute recall and precision at all ranks
            recall = true_positives / annotations.size(0) if annotations.size(0) else true_positives
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # Compute average precision
            AP = compute_ap(recall, precision)
            APs.append(AP)
            #            f.write("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), AP, np.mean(APs)))
            #            f.write('\n')
            print("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), AP, np.mean(APs)))
    f.write("Mean Average Precision: %.4f" % np.mean(APs))
    f.write('\n')
    f.close()
    print("Mean Average Precision: %.4f" % np.mean(APs))

if __name__ == '__main__':
    num_classes = 6
    anchors = [(38, 15), (15, 36), (174, 67), (37, 91), (64, 26), (54, 126), (25, 61), (96, 39), (135, 55)]
    num_anchors = 3
    input_size = 512
    Epoch_number = 99999
    valid_path = '/home/kongdeqian/dKong/TLdataset_test/list.txt'

    # model = myNet_Res101(num_classes, num_anchors, anchors, input_size, True, is_train = True).cuda()
    model = torch.load('./model_0914_29.pkl', map_location = 'cuda:0')
    model.set_test()

    test_all(model, valid_path, Epoch_number, batch_size = 20, num_classes = 11, n_cpu = 8, iou_thres = 0.5, img_size = 512, cuda = True)
