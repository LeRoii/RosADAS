#!/usr/bin/env python3
#coding:utf-8

import rospy
import sys
sys.path.append('.')
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import glob

videoPath = '/space/data/road/1.avi'

def pubVideo():
    rospy.init_node('videopub',anonymous = True)
    pub = rospy.Publisher('zed', Image, queue_size = 1)
    rate = rospy.Rate(30)
    bridge = CvBridge()
    cap = cv2.VideoCapture(videoPath)
    ret, frame = cap.read()
    cnt = 0
    rospy.loginfo('video frame cnt:%d', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while not rospy.is_shutdown():
        ret, frame = cap.read()

        if not ret:
            rospy.loginfo('frame end')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        pub.publish(bridge.cv2_to_imgmsg(frame,"bgr8"))
        rospy.loginfo('frame cnt:%d', cnt)
        cnt = cnt+1
        # cv2.imshow("lala",frame)
        # cv2.waitKey(0)
        # if cnt == 224:
        # str = input()
        # if str == 'a':
        #     pass
        rate.sleep()



def pubImg():
    image_paths = glob.glob('/space/code/calibration/image/*.jpg')
    image_paths.sort()
    print(image_paths)

    rospy.init_node('videopub',anonymous = True)
    pub = rospy.Publisher('/wideangle/image_color', Image, queue_size = 1)
    rate = rospy.Rate(10)
    bridge = CvBridge()

    imgNum = len(image_paths)
    print('img num:', imgNum)
    i=0
    while not rospy.is_shutdown():
        if i >= imgNum:
            rospy.loginfo('frame end')
            break
        image = cv2.imread(image_paths[i])
        # cv2.imwrite(str(i)+'.png',image)
        # image = image[:,:,0:800]
        # image = cv2.resize(image,(1280,720))
        # cv2.imshow("lala",image)
        # cv2.waitKey(0)
        
        
        
        pub.publish(bridge.cv2_to_imgmsg(image,"bgr8"))
        rospy.loginfo('frame cnt:%d, img name:%s', i, image_paths[i])
        #cnt = cnt+1
        cv2.imshow("lala",image)
        cv2.waitKey(0)
        i = i+1

        rate.sleep()


if __name__ == '__main__':
    try:
        # pubVideo()
        pubImg()
    except rospy.ROSInterruptException:
        pass
