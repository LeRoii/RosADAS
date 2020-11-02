from ldw_algo_instance import laneDepartureWarning
from yolo_detection.msg import BoundingBox, BoundingBoxes

from PIL import Image as PIL_IMG
import configs.testconfig

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import threading
import playsound
import numpy as np

CFG = configs.testconfig.cfg

def playWarningSound():
    playsound.playsound('/home/iairiv/code/model/warn.mp3')

class adasDetector:
    def __init__(self):
        self.yoloBoxes = BoundingBoxes()
        self.trafficLightBoxes = BoundingBoxes()
        self.ldw_inst = laneDepartureWarning()
        self.image_pub = rospy.Publisher("adasresult", Image,queue_size = 1)
        self.img = np.zeros([CFG.imgWidth, CFG.imgHeight, 3],np.uint8)
        self.bridge = CvBridge()



    def callbackyolo(self, boxmsg):
        print('callbackyolo, boxes len:', boxmsg.objNum)
        self.yoloBoxes.objNum = boxmsg.objNum
        self.yoloBoxes.bounding_boxes = []
        for i in range(boxmsg.objNum):
            print('box id:', boxmsg.bounding_boxes[i].id)
            box = boxmsg.bounding_boxes[i]
            self.yoloBoxes.bounding_boxes.append(box)

            # cv2.rectangle(self.img, (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax)), (0, 255, 0), 4)

    def callbackTrafficLight(self, boxmsg):
        print('callbackTrafficLight, boxes len:', boxmsg.objNum)
        self.trafficLightBoxes.objNum = boxmsg.objNum
        self.trafficLightBoxes.bounding_boxes = []
        for i in range(boxmsg.objNum):
            print('box id:', boxmsg.bounding_boxes[i].id)
            box = boxmsg.bounding_boxes[i]
            self.trafficLightBoxes.bounding_boxes.append(box)

    def drawYoloResult(self, data):
        for i in range(data.objNum):
            box = data.bounding_boxes[i]
            cv2.rectangle(self.img, (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax)), (0, 255, 0), 3)
            t_size = cv2.getTextSize(box.id, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = int(box.xmin) + t_size[0] + 3, int(box.ymin) + t_size[1] + 4
            cv2.rectangle(self.img, (int(box.xmin), int(box.ymin)), c2, (0, 255, 0), -1)
            cv2.putText(self.img, box.id, (int(box.xmin), int(box.ymin)+10), cv2.FONT_HERSHEY_PLAIN, 1, (225, 255, 255), 1)


    
    def callbackImg(self, imgMsg):
        cv_image = self.bridge.imgmsg_to_cv2(imgMsg, "bgr8")
        self.img = cv_image.copy()
        pilImg = PIL_IMG.fromarray(np.uint8(cv_image))
        ret = self.ldw_inst.process(pilImg)

        llane = ret['leftlane']
        rlane = ret['rightlane']
        warningret = ret['warningret']
        color = (0, 0, 255) if warningret == 1 else (0, 255, 0)
        if warningret == 1:
            soundplayTh = threading.Thread(target=playWarningSound)
            soundplayTh.start()

        for idx in range(11):
            cv2.line(self.img, (int(llane.points[idx][0]), int(llane.points[idx][1])), (int(llane.points[idx+1][0]), int(llane.points[idx+1][1])), color, 10)
            cv2.line(self.img, (int(rlane.points[idx][0]), int(rlane.points[idx][1])), (int(rlane.points[idx+1][0]), int(rlane.points[idx+1][1])), color, 10)

        self.drawYoloResult(self.yoloBoxes)
        self.drawYoloResult(self.trafficLightBoxes)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "bgr8"))

        
        pass