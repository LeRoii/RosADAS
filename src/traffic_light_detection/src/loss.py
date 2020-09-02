#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:25:17 2018
loss function of yolo series
@author: kongdeqian
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from util import bbox_ious, bbox_iou
import numpy as np
import numpy as np
def build_targets(pred_boxes, target, anchors, num_anchors, num_classes = 11, dim = 32, ignore_thres = 0.5, input_size = 1024):
    '''
   pre_boxes: result from detection part [batch_size, c, h, w] eg: [1,48,32,32],[1,48,64,64],[1,48,128,128]
   then view it as [1,3,32,32,16]
   dim as 4 grid index
   '''
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    dim = dim
    mask        = torch.zeros(nB, nA, dim, dim)
    conf_mask   = torch.ones(nB, nA, dim, dim)
    tx          = torch.zeros(nB, nA, dim, dim)
    ty          = torch.zeros(nB, nA, dim, dim)
    tw          = torch.zeros(nB, nA, dim, dim)
    th          = torch.zeros(nB, nA, dim, dim)
    tconf       = torch.zeros(nB, nA, dim, dim)
    tcls        = torch.zeros(nB, nA, dim, dim, num_classes)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to current dimxdim map
            gx = float(target[b, t, 1] * dim)
            gy = float(target[b, t, 2] * dim)
            gw = float(target[b, t, 3] * dim)
            gh = float(target[b, t, 4] * dim)
            # print(gx,gy,gw,gh)
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_ious(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres] = 0
            # Find the best matching anchor box
           # best_n = np.argmax(anch_ious)
            n = np.argwhere(anch_ious>0.3)
            if len(n)>0:
                best_n = n
                best_n = best_n.squeeze(0)
            else:
                best_n = np.argmax(anch_ious)
                best_n = best_n.unsqueeze(0)
#             best_n = np.argmax(anch_ious)
#             best_n = best_n.unsqueeze(0)
#                break
#            print(best_n)

            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            #print(dim, b, best_n, gj, gi)
#            pred_box = pred_boxes[b, best_n, gj, gi,:].unsqueeze(0)
            pred_box = pred_boxes[b, best_n, gj, gi,:]
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = float(gx - gi)
            ty[b, best_n, gj, gi] = float(gy - gj)
            # Width and height
            anchors_w = torch.FloatTensor([x[0] for x in anchors])
            anchors_h = torch.FloatTensor([x[1] for x in anchors])
            tw[b, best_n, gj, gi] = torch.log(gw/anchors_w[best_n] + 1e-16)
#            print(torch.log(gw/anchors_w[best_n] + 1e-16))
            th[b, best_n, gj, gi] = torch.log(gh/anchors_h[best_n] + 1e-16)
#            print(torch.log(gh/anchors_h[best_n] + 1e-16))
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
            # Calculate iou between ground truth and best matching prediction
#            print()
            iou = bbox_ious(gt_box, pred_box, x1y1x2y2=False)

            tconf[b, best_n, gj, gi] = 1

            # if iou > 0.5:
            #    nCorrect += 1
            if (sum((iou>0.5)==1)) >= 1:
                nCorrect += 1
                
           

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


class DetectionLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_size, is_train):
        super(DetectionLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_size = input_size
        self.is_train = is_train
        self.ignore_thres = 0.5
        self.lambda_coord = 100
        self.lambda_noobj = .5

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.l1loss = nn.L1Loss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, targets):
        # x.size()      = [bs,C,W,H]
        # traget.size() = [bs,max_obj,1+4]
        # print self.anchors
        bs = x.size(0)
        g_dim = x.size(2)
        stride =  float(self.input_size / float(g_dim))
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # prediction.size() = [1,3,32,32,11+5]
        prediction = x.view(bs,  self.num_anchors, self.bbox_attrs, g_dim, g_dim).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid((prediction[..., 4]))         # Conf
#        print(conf)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
#        print(pred_cls.size())
        pred_cls = F.softmax(prediction[..., 5:],dim=4)

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = (x.data + grid_x) #* stride
        pred_boxes[..., 1] = (y.data + grid_y) #* stride
        pred_boxes[..., 2] = (torch.exp(w.data) * anchor_w) #* stride
        pred_boxes[..., 3] = (torch.exp(h.data) * anchor_h) #* stride

        # Training
        if self.is_train:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()


            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes.cpu().data,
                                                                            targets.cpu().data,
                                                                            scaled_anchors,
                                                                            self.num_anchors,
                                                                            self.num_classes,
                                                                            g_dim,
                                                                            self.ignore_thres,
                                                                            self.input_size)


            
            if x.is_cuda:
                tx, ty, tw, th, tconf, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), tconf.cuda(), tcls.cuda()

            nProposals = int((conf > 0.25).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1

            # Handle masks
            mask = (mask.type(FloatTensor))
            cls_mask = (mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor))
            conf_mask = (conf_mask.type(FloatTensor))


            # Mask outputs to ignore non-existing objects

            loss_x = self.lambda_coord * self.mse_loss(x * mask, tx * mask)
            loss_y = self.lambda_coord * self.mse_loss(y * mask, ty * mask)
            loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask)
            loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask)
            loss_conf = self.mse_loss(conf * tconf, tconf)
            # loss_x = self.l1loss(x * mask, tx * mask)
            # loss_y = self.l1loss(y * mask, ty * mask)
            # loss_w = self.l1loss(w * mask, tw * mask)
            # loss_h = self.l1loss(h * mask, th * mask)
            # loss_conf = self.l1loss(conf * tconf, tconf)

            loss_conf_no = self.lambda_noobj*self.mse_loss(conf * (1-tconf), tconf * (1-tconf))
            loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls + loss_conf_no

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_conf_no.item(), loss_cls.item(), recall

        if not self.is_train and targets is None:
            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data
