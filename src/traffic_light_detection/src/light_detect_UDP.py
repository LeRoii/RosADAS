#!/usr/bin/python3
#!coding=utf-8
from __future__ import division
import sys
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2

from util import *
import random

import os
import time
import datetime

import torch
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms
import numpy as np
import random
from myNet import *
from util import *
# import UDP_trans
# from save_result import *
# from UDP_data_transform import *
# import socket
# from collections import deque

from yolo_detection.msg import BoundingBox, BoundingBoxes


class YOLO_detection:
    def __init__(self):
        self.image_pub = rospy.Publisher("traffic_light_detect_result", Image, queue_size = 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/wideangle/image_color", Image, self.callback)
        self.batch_size = 1
        self.reso = 512
        self.confidence = 0.5
        self.nms_thesh = 0.05
        self.cls_conf = 0.4
        self.CUDA = torch.cuda.is_available()
        self.num_classes = 11
        # self.classes = load_classes("/home/iairiv/code/roslightdet/src/traffic_light_detection/src/tf.names")
        # self.classes = load_classes("/space/code/rosTSR/src/traffic_light_detection/src/tf.names")
        self.classes = load_classes(rospy.get_param("lightdet_classname"))
        self.colors = random_color(self.num_classes)
        # self.weights_file = "/home/iairiv/code/roslightdet/src/traffic_light_detection/src/model_1105_99.pkl"
        # self.weights_file = "/space/code/rosTSR/src/traffic_light_detection/src/model_1105_99.pkl"

        self.weights_file = rospy.get_param("lightdet_weight")
        self.anchors = [(38, 15), (15, 36), (25, 61), (37, 91), (54, 126), (64, 26), (96, 39), (135, 55), (174, 67)]
        self.model = myNet_Res101(self.num_classes, 3, self.anchors, self.reso, self.CUDA, is_train = False)
        self.model.load_state_dict(torch.load(self.weights_file, map_location='cuda:0'))
        if self.CUDA: self.model.cuda()
        self.model.set_test()
        # self.UDP = UDP_trans.TrafficLight_UDP('10.0.0.14', 7800)
        self.i = 0

        self.boxes_pub = rospy.Publisher("tlight_detect_result_boxes", BoundingBoxes, queue_size = 1)

    def transform_input(self, img):
        return prep_image(img, self.reso)

    def detection(self, input):
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            prediction = self.model(input, None)
        prediction = non_max_suppression(prediction, self.num_classes, conf_thres = self.confidence, nms_thres = self.nms_thesh, cls_dependent = False)
        return prediction

    def write(self, output, img):
        # im_dim_list = [(img.shape[1], img.shape[0])]
        # im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        # print im_dim_list
        output = output[0]
        # print output
        # if self.CUDA:
        #     im_dim_list = im_dim_list.cuda()
        # scaling_factor = torch.min(self.reso / im_dim_list, 1)[0].view(-1, 1)
        # print(scaling_factor)
        # output[:, [0,2]] -= (self.reso - scaling_factor * im_dim_list[:, 0]) / 2
        # output[:, [1,3]] -= (self.reso - scaling_factor * im_dim_list[:, 1]) / 2
        # output[:, 0:4] /= scaling_factor
        for x in output:
            c1 = tuple(x[0:2].int())
            c2 = tuple(x[2:4].int())
            cls = int(x[-1])
            color = self.colors[cls]
            label = "{0}".format(self.classes[cls])
            cv2.rectangle(img, c1, c2, color, 3)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)


        # cv2.imwrite(str(self.i)+'.png', img)
        self.i+=1
        return img


    def callback(self, data):
        print('callback!!!!')
        startt = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            input_image = self.transform_input(cv_image)
            prediction = self.detection(input_image)
            # self.UDP.UDP_data_transform_queue(prediction[0], self.reso, self.cls_conf)


            
            if prediction[0] is None:
                result = cv_image
            else:
                # print prediction
                im_dim_list = [(cv_image.shape[1], cv_image.shape[0])]
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                if self.CUDA:
                    im_dim_list = im_dim_list.cuda()
                scaling_factor = torch.min(self.reso / im_dim_list, 1)[0].view(-1, 1)
                print(scaling_factor)

                tlightRet = prediction[0]
                tlightRet[:, [0,2]] -= (self.reso - scaling_factor * im_dim_list[:, 0]) / 2
                tlightRet[:, [1,3]] -= (self.reso - scaling_factor * im_dim_list[:, 1]) / 2
                tlightRet[:, 0:4] /= scaling_factor

                result = self.write(prediction, cv_image)

                tlightRet[:, [0,2]] *= 2
                tlightRet[:, [1,3]] *= 2
        except CvBridgeError as e:
            print('CvBridgeError;',e)

        # cv2.imshow("image windows", result)
        # cv2.waitKey(3)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
        except CvBridgeError as e:
            print(e)

        boxes = BoundingBoxes()
        if prediction[0] is None:
            detec_len = 0
        else:
            detec_len = len(prediction[0])

        
        for i in range(detec_len):
            box = BoundingBox()
            # box.num = prediction[i][0]
            box.xmin = tlightRet[i][0]
            box.ymin = tlightRet[i][1]
            box.xmax = tlightRet[i][2]
            box.ymax = tlightRet[i][3]
            # box.probability = prediction[i][6]
            box.id = "{0}".format(self.classes[int(tlightRet[i][-1])])
            #self.box_pub.publish(self.box)
        
            boxes.bounding_boxes.append(box)

        boxes.objNum = detec_len
        boxes.header.stamp = rospy.Time.now()
        
        self.boxes_pub.publish(boxes)
        
        print('light det all use：', time.time()-startt)


def random_color(num):
    colors = []
    for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append([b, g, r])
    return colors


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def main(args):
    ic = YOLO_detection()
    print ("light model init end")
    rospy.init_node("image_converter", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)

