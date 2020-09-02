# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:04 2018
Learn to implement Darknet-53 of YOLO-v3
Aim to tranform the output of feature map to make fusions on difft maps  
@author: kongdeqian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
    
def bbox_iou(box1, box2, x1x2y1y2 = True):
    # return IOU of two bbox
    # given 2 boxs : the upper left and the lower right
    b1_xmin ,b1_ymin, b1_xmax, b1_ymax= box1[0], box1[1], box1[2], box1[3]
    b2_xmin ,b2_ymin, b2_xmax, b2_ymax= box2[0], box2[1], box2[2], box2[3]
    
    inter_xmin = max(b1_xmin, b2_xmin)
    inter_xmax = min(b1_xmax, b2_xmax)
    inter_ymin = max(b1_ymin, b2_ymin)
    inter_ymax = min(b1_ymax, b2_ymax)
    
    intersection = torch.clamp(inter_xmax - inter_xmin + 1, min = 0)*\
                   torch.clamp(inter_ymax - inter_ymin + 1, min = 0)
                   
    b1 = (b1_xmax - b1_xmin + 1)*(b1_ymax - b1_ymin + 1)
    b2 = (b2_xmax - b2_xmin + 1)*(b2_ymax - b2_ymin + 1)
    union = b1 + b2 - intersection
    
    iou = intersection/union
    return iou

def bbox_ious(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4, cls_dependent = True):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    #print(prediction.size())
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        if not cls_dependent:
            # Iterate through all predicted boxes
            # Sort the detections by maximum objectness confidence
            detections_class = detections
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_ious(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
        else:         
            # Iterate through all predicted boxes in certain class
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
            for c in unique_labels:
                # Get the detections with the particular class
                detections_class = detections[detections[:, -1] == c]
                # Sort the detections by maximum objectness confidence
                _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                detections_class = detections_class[conf_sort_index]
                # Perform non-maximum suppression
                max_detections = []
                while detections_class.size(0):
                    # Get detection with highest confidence and save as max detection
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(detections_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = bbox_ious(max_detections[-1], detections_class[1:])
                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]
    
                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    return output


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




if __name__ == '__main__':
    img = cv2.imread("0339.jpg")
    img = cv2.resize(img, (1024,1024))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W X C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  
    
    img_ = img_.cuda()
    
    target = torch.rand(1,3,4)
    anchors=[(10,13),(15,20),[28,40]]
    num_anchors = 3
    num_classes = 80








































