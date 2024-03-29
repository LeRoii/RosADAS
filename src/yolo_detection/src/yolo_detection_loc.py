#!/usr/bin/python2.7
#!coding=utf-8
from __future__ import division
import sys
# import rospy
import numpy as np
# from sensor_msgs.msg import Image
from std_msgs.msg import String
# from std_msgs.msg import Int16
# from cv_bridge import CvBridge, CvBridgeError
import cv2

import torch
import time
from util_eight_labels import *
from darknet import Darknet
import random
# import UDPtrans

class YOLO_detection:
    def __init__(self):
        self.start=False
        # self.image_pub = rospy.Publisher("YOLO_detect_result", Image)
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("wideangle/image_color", Image, self.callback)
        # self.loc_sub = rospy.Subscriber("/yolo_state", Int16, self.loc_callback)
        self.batch_size = 1
        self.reso = 416
        self.confidence = 0.5
        self.nms_thesh = 0.4
        self.CUDA = torch.cuda.is_available()
        self.num_classes = 80
        self.classes = load_classes("/home/iairiv/code/yolo/src/yolo_detection/src/data/coco.names")
        self.colors = random_color()
        self.cfg_file = "/home/iairiv/code/yolo/src/yolo_detection/src/cfg/yolov3.cfg"
        self.weights_file = "/home/iairiv/code/yolo/src/yolo_detection/src/yolov3.weights"
        self.model = Darknet(self.cfg_file)
        self.model.load_weights(self.weights_file)
        self.model.net_info["height"] = self.reso
        if self.CUDA: self.model.cuda()
        self.model.eval()
        self.send_by_UDP = True
        self.draw_res = True
        # if self.send_by_UDP:
        #     self.UDP = UDPtrans.YOLO_UDP('195.0.0.5', 7800)

    def transform_input(self, img):
        return prep_image(img, self.reso)

    def yolo_detection(self, input):
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            prediction = self.model(Variable(input), self.CUDA)
        # print prediction
        prediction = write_results(prediction, self.confidence, self.num_classes, nms_conf=self.nms_thesh)
        return prediction

    def write(self, output, img):
        # im_dim_list = [(img.shape[1], img.shape[0])]
        # im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        # if self.CUDA:
        #     im_dim_list = im_dim_list.cuda()
        # scaling_factor = torch.min(self.reso / im_dim_list, 1)[0].view(-1, 1)
        # # print output
        # output[:, [1, 3]] -= (self.reso - scaling_factor * im_dim_list[:, 0]) / 2
        # output[:, [2, 4]] -= (self.reso - scaling_factor * im_dim_list[:, 1]) / 2
        # output[:, 1:5] /= scaling_factor
        for x in output:
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            cls = int(x[-1])
            color = self.colors[cls]
            label = "{0}".format(self.classes[cls])
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    def callback(self):
        # if self.start:
            # try:
            print(time.time())
            # start_time = rospy.Time.now()
            # start_time_second = start_time.to_sec()
            # timeArray = time.localtime(start_time_second)
            # timeArray_H_M_S = time.strftime("%H_%M_%S", timeArray)
            # nano_seconds = str(int(start_time.to_nsec() - int(start_time_second) * 1e9)).zfill(9)
            # timeArray_H_M_S_MS = timeArray_H_M_S + "_" + nano_seconds[:3]
            # print timeArray_H_M_S_MS
            # YOLO detect
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.imread('/media/iairiv/My%20Passport/DATASET/yolo_detect/M_H_m22_7_37/1.jpg')
            # cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            input_image = self.transform_input(cv_image)
            prediction = self.yolo_detection(input_image)
                # print prediction
                # coordinate transformation
                # if type(prediction) == int:
                #     if self.draw_res == True:
                #         result = cv_image
                #     if self.send_by_UDP:
                #         self.UDP.send_message(timeArray_H_M_S_MS, None)
                # else:
                #     # image size should be the same with the size when we calibrate
                #     im_dim_list_list = [(cv_image.shape[1], cv_image.shape[0])]
                #     # print im_dim_list_list
                #     im_dim_list = torch.FloatTensor(im_dim_list_list).repeat(1, 2)
                #     if self.CUDA:
                #         im_dim_list = im_dim_list.cuda()
                #     scaling_factor = torch.min(self.reso / im_dim_list, 1)[0].view(-1, 1)
                #     prediction[:, [1, 3]] -= (self.reso - scaling_factor * im_dim_list[:, 0]) / 2
                #     prediction[:, [2, 4]] -= (self.reso - scaling_factor * im_dim_list[:, 1]) / 2
                #     prediction[:, 1:5] /= scaling_factor
                #     prediction[:, [1, 3]] = torch.clamp(prediction[:,[1,3]], 0.0, im_dim_list_list[0][0])
                #     prediction[:, [2, 4]] = torch.clamp(prediction[:,[2,4]], 0.0, im_dim_list_list[0][1])
                #     # print prediction
                #     # UDP send
                #     if self.send_by_UDP:
                #         self.UDP.send_message(timeArray_H_M_S_MS, prediction.cpu().numpy().tolist())
                #     # draw Image
                #     if self.draw_res:
            result = self.write(prediction, cv_image)
            cv2.imshow('2',result)
            cv2.waitKey(0)

            # except CvBridgeError as e:
            #     print (e)

            # cv2.imshow("image windows", result)
            # cv2.waitKey(3)

        #     try:
        #         self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
        #     # except CvBridgeError as e:
        #     #     print e
        # else:
        #     time.sleep(0.05)


    def loc_callback(self, start_flag):
        print(start_flag)
        if start_flag.data == 1:
            self.start = True
        else:
            self.start = False


def random_color():
    colors = []
    for i in range(81):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append([b, g, r])
    return colors


# def main(args):
#     ic = YOLO_detection()
#     rospy.init_node("image_converter", anonymous=True)
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print ("Shutting down")
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    # main(sys.argv)
    ic = YOLO_detection()
    ic.callback()

