#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import playsound
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import cv2
import threading

import time
from PIL import Image as PIL_IMG

sys.path.append("/home/iairiv/code/adas0903/src/lanedet/Ultra-Fast-Lane-Detection/configs")
print(sys.path)
from ldw_algo_instance import laneDepartureWarning
from adas_algo_instance import adasDetector
from yolo_detection.msg import BoundingBox, BoundingBoxes

g_frameCnt = 0
g_videoPlay = True
g_keyboardinput = ''
g_writevideo = False

def listenkeyboard():
    global g_videoPlay, g_keyboardinput, g_writevideo
    while True:
        g_keyboardinput = input()
        if g_keyboardinput == 'a':
            g_videoPlay = not g_videoPlay
        elif g_keyboardinput == 'r':
            print('g_keyboardinput rrrrrr')
            g_writevideo = True

def playWarningSound():
    playsound.playsound('/space/warn.mp3')


def testImg():
    lanedet = LaneDetect()
    img = PIL_IMG.open('/space/data/cxg0903/ok/img4/22.png')
    lanes = lanedet.process(img)
    img = cv2.imread('/space/data/cxg0903/ok/img4/22.png')
    for lane in lanes:
        for n in range(11):
            cv2.line(img, (lane[n][0], lane[n][1]), (lane[n+1][0], lane[n+1][1]),  (0,255,0), 10) 

    cv2.imwrite('111.png', img)


def testVideo():
    global g_videoPlay, g_keyboardinput, g_writevideo, g_frameCnt
    rospy.init_node("lanedetnode", anonymous=True)
    image_pub = rospy.Publisher("lanedetframe", Image,queue_size = 1)
    cap = cv2.VideoCapture('/space/data/cxg0903/ok/4.avi')
    ret, frame = cap.read()
    rate = rospy.Rate(15)
    rospy.loginfo('video frame cnt:%d', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    listernerTh = threading.Thread(target=listenkeyboard)
    listernerTh.setDaemon(True)
    listernerTh.start()

    bridge = CvBridge()

    ldw_inst = laneDepartureWarning()

    # out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (1920,1200),True)
    
    while not rospy.is_shutdown():
        while not g_videoPlay:
            time.sleep(1)
            if g_keyboardinput == "'":
                g_keyboardinput = ''
                break

        ret, frame = cap.read()
        if not ret:
            rospy.loginfo('frame end')
            break

        g_frameCnt = g_frameCnt + 1
        rospy.loginfo('frame cnt:%d', g_frameCnt)
        cv_image = frame.copy()
        startt = time.time()

        pilImg = PIL_IMG.fromarray(np.uint8(cv_image))

        ret = ldw_inst.process(pilImg)
        llane = ret['leftlane']
        rlane = ret['rightlane']
        warningret = ret['warningret']
        color = (0, 0, 255) if warningret == 1 else (0, 255, 0)
        if warningret == 1:
            soundplayTh = threading.Thread(target=playWarningSound)
            soundplayTh.start()

        for idx in range(11):
            cv2.line(cv_image, (int(llane.points[idx][0]), int(llane.points[idx][1])), (int(llane.points[idx+1][0]), int(llane.points[idx+1][1])), color, 10)
            cv2.line(cv_image, (int(rlane.points[idx][0]), int(rlane.points[idx][1])), (int(rlane.points[idx+1][0]), int(rlane.points[idx+1][1])), color, 10)
        
        image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #print('lanedet all use：', time.time()-startt)


        # # if g_writevideo:
        # #     print('write video')
        # #     out.write(cv_image)

        # ic.process(cv_image)
        # ic.image_pub.publish(ic.bridge.cv2_to_imgmsg(ic.img, "bgr8"))
        rate.sleep()

    # out.release()


def main():
    detector = adasDetector()
    # self.bridge = CvBridge()
    yolo_sub = rospy.Subscriber("YOLO_detect_result_boxes", BoundingBoxes, detector.callbackyolo)
    tlight_sub = rospy.Subscriber("tlight_detect_result_boxes", BoundingBoxes, detector.callbackTrafficLight)

    image_sub = rospy.Subscriber("/wideangle/image_color", Image, detector.callbackImg)
    rospy.init_node("lanedetnode", anonymous=True)
    rospy.loginfo('lanedet model init end')
    rospy.spin()




if __name__ == "__main__":
    main()
    # testVideo()