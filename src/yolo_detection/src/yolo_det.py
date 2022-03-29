#!/usr/bin/python3
#!coding=utf-8
from __future__ import division
import sys
# sys.path.append('/home/iairiv/code/yolo/src/yolo_detection/src')
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from std_msgs.msg import Float64MultiArray
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import torch
import time
from util_eight_labels import *
from darknet import Darknet
import random
# import UDPtrans

from yolo_detection.msg import BoundingBox, BoundingBoxes

class YOLO_detection:
    def __init__(self):
        self.boxes = BoundingBoxes()
        self.box = BoundingBox()
        self.image_pub = rospy.Publisher("YOLO_detect_result", Image, queue_size = 1)
        self.boxes_pub = rospy.Publisher("YOLO_detect_result_boxes", BoundingBoxes, queue_size = 1)
        # self.result = rospy.Publisher('YOLO_detect_result', Float64MultiArray, queue_size=10)
        self.bridge = CvBridge()
        #self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/wideangle/image_color", Image, self.callback)
        self.image_sub = rospy.Subscriber('/cam_front/csi_cam/image_raw', Image, self.callback)
        self.batch_size = 1
        self.reso = 416
        self.confidence = 0.5
        self.nms_thesh = 0.4
        self.CUDA = torch.cuda.is_available()
        self.num_classes = 80

        # self.classes = load_classes("/home/iairiv/code/yolo/src/yolo_detection/src/data/coco.names")
        # self.cfg_file = "/home/iairiv/code/yolo/src/yolo_detection/src/cfg/yolov3.cfg"
        # self.weights_file = "/home/iairiv/code/yolo/src/yolo_detection/src/yolov3.weights"
        
        self.colors = random_color()

        # self.classes = load_classes("/space/code/rosadas/src/yolo_detection/src/data/coco.names")
        # self.cfg_file = "/space/code/rosadas/src/yolo_detection/src/cfg/yolov3.cfg"
        # self.weights_file = "/space/code/rosadas/src/yolo_detection/src/yolov3.weights"

        self.classes = load_classes(rospy.get_param("yolo_classname", "/space/code/rosadas/src/yolo_detection/src/data/coco.names" ))
        self.cfg_file = rospy.get_param("yolo_cfg","/space/code/rosadas/src/yolo_detection/src/cfg/yolov3.cfg")
        self.weights_file = rospy.get_param("yolo_weight","/space/code/rosadas/src/yolo_detection/src/yolov3.weights")

        self.model = Darknet(self.cfg_file)
        self.model.load_weights(self.weights_file)
        self.model.net_info["height"] = self.reso
        if self.CUDA: self.model.cuda()
        self.model.eval()
        self.send_by_UDP = False
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
        
        
        # for x in output:
        #     c1 = tuple(x[1:3].int())
        #     c2 = tuple(x[3:5].int())
        #     cls = int(x[-1])
        #     color = self.colors[cls]
        #     label = "{0}".format(self.classes[cls])
        #     print(label)
        #     color = [255,255,0]
        #     cv2.rectangle(img, c1, c2, color, 2)
        #     t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        #     c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        #     cv2.rectangle(img, c1, c2, color, 2)
        #     cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 1)
        # return img
        for x in output:
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            cls = int(x[-1])
            color = (0, 255, 0)#self.colors[cls]
            label = "{0}".format(self.classes[cls])
            print(label)
            cv2.rectangle(img, c1, c2, color, 4)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img



    def callback(self, data):
        startt = time.time()
        try:
            start_time = rospy.Time.now()
            start_time_second = start_time.to_sec()
            timeArray = time.localtime(start_time_second)
            timeArray_H_M_S = time.strftime("%H_%M_%S", timeArray)
            nano_seconds = str(int(start_time.to_nsec() - int(start_time_second) * 1e9)).zfill(9)
            timeArray_H_M_S_MS = timeArray_H_M_S + "_" + nano_seconds[:3]
            print(timeArray_H_M_S_MS)
            # YOLO detect
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            input_image = self.transform_input(cv_image)
            prediction = self.yolo_detection(input_image)
            # print(type(prediction))
            # coordinate transformation
            if type(prediction) == int:
                if self.draw_res == True:
                    result = cv_image
                # if self.send_by_UDP:
                #     self.UDP.send_message(timeArray_H_M_S_MS, None)
            else:
                # image size should be the same with the size when we calibrate
                im_dim_list_list = [(cv_image.shape[1], cv_image.shape[0])]
                # print im_dim_list_list
                im_dim_list = torch.FloatTensor(im_dim_list_list).repeat(1, 2)
                if self.CUDA:
                    im_dim_list = im_dim_list.cuda()
                scaling_factor = torch.min(self.reso / im_dim_list, 1)[0].view(-1, 1)
                prediction[:, [1, 3]] -= (self.reso - scaling_factor * im_dim_list[:, 0]) / 2
                prediction[:, [2, 4]] -= (self.reso - scaling_factor * im_dim_list[:, 1]) / 2
                prediction[:, 1:5] /= scaling_factor
                prediction[:, [1, 3]] = torch.clamp(prediction[:,[1,3]], 0.0, im_dim_list_list[0][0])
                prediction[:, [2, 4]] = torch.clamp(prediction[:,[2,4]], 0.0, im_dim_list_list[0][1])
                # print prediction
                # UDP send
                # if self.send_by_UDP:
                #     self.UDP.send_message(timeArray_H_M_S_MS, prediction.cpu().numpy().tolist())
                # draw Image
                if self.draw_res:
                    result = self.write(prediction, cv_image)
                # pub.publish(self.boxes)


        except CvBridgeError as e:
            print(e)
        # return prediction
        #
        # # cv2.imshow("image windows", result)
        # # cv2.waitKey(3)
        #

        try:
            # prediction = prediction.cpu().numpy().tolist()
            # boxes = self.boxes.bounding_boxes()
            # print(type(prediction))
            boxes = BoundingBoxes()
            if type(prediction) == int:
                detec_len = 0
            else:
                detec_len = len(prediction)
            for i in range(detec_len):
                box = BoundingBox()
                box.num = prediction[i][0]
                box.xmin = prediction[i][1]
                box.ymin = prediction[i][2]
                box.xmax = prediction[i][3]
                box.ymax = prediction[i][4]
                box.probability = prediction[i][6]
                box.id = "{0}".format(self.classes[int(prediction[i][7])])
                #self.box_pub.publish(self.box)
            
                boxes.bounding_boxes.append(box)
            boxes.objNum = detec_len
            boxes.header.stamp = rospy.Time.now()
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
            
            self.boxes_pub.publish(boxes)

        except CvBridgeError as e:
            print(e)
        # time.sleep(0.05)
        print('yolo use：', time.time()-startt)


def random_color():
    colors = []
    for i in range(81):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append([b, g, r])
    return colors


def main(args):
    ic = YOLO_detection()
    print ("yolo init end")
    rospy.init_node("image_converter", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)

