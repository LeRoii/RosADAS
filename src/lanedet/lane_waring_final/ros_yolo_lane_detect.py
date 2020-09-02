#!/usr/bin/env python3

import os
# import xml.dom.minidom
import cv2
import sys
import time
import threading

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import global_config
from config import *
from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
import torch.backends.cudnn as cudnn
from utils.prob2lines import getLane
from utils.lanenet_postprocess import LaneNetPostProcessor
# import matplotlib.pyplot as plt
from Detection import Detection

from lane_obj import Lane
from lane_tracker import LaneTracker

from yolo_detection.msg import BoundingBox, BoundingBoxes
import glob

CFG = global_config.cfg

g_frameCnt = 0
g_videoPlay = True
g_keyboardinput = ''
g_writevideo = False

fp = 0
tp = 0
fn = 0

class Lane_warning:
    def __init__(self):
        self.image_pub = rospy.Publisher("lanedetframe", Image,queue_size = 1)
        self.maskimg_pub = rospy.Publisher("lanedetmask", Image,queue_size = 1)
        self.binimg_pub = rospy.Publisher("lanedetbin", Image,queue_size = 1)
        self.morphoimg_pub = rospy.Publisher("lanedetmorph", Image,queue_size = 1)
        # self.bridge = CvBridge()
        self.yolo_sub = rospy.Subscriber("YOLO_detect_result_boxes", BoundingBoxes, self.callbackyolo)
        self.tlight_sub = rospy.Subscriber("tlight_detect_result_boxes", BoundingBoxes, self.callbackTrafficLight)

        # self.image_sub = rospy.Subscriber("YOLO_detect_result", Image, self.callbackRos)
        # self.image_sub = rospy.Subscriber("/camera/image_color", Image, self.callbackRos)
        self.image_sub = rospy.Subscriber("/wideangle/image_color", Image, self.callbackRos)
        # self.image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image，queue_size=1, buff_size=110592*6)
        self.weights_file = rospy.get_param("lanenet_weight")
        self.CUDA = torch.cuda.is_available()
        self.postprocessor = LaneNetPostProcessor()
        self.warning = Detection()
        self.band_width = 1.5
        self.image_X = CFG.IMAGE_WIDTH
        self.image_Y = CFG.IMAGE_HEIGHT
        self.car_X = self.image_X/2
        self.car_Y = self.image_Y
        self.model = LaneNet(pretrained=False, embed_dim=4, delta_v=.5, delta_d=3.)
        self.save_dict = torch.load(self.weights_file, map_location='cuda:0')
        self.model.load_state_dict(self.save_dict['net'])
        # self.model.load_state_dict(torch.load(self.weights_file, map_location='cuda:0'))
        if self.CUDA: self.model.cuda()
        self.model.set_test()
        self.lastlane = np.ndarray(4,)
        self.bridge = CvBridge()

        self.leftlane = Lane('left')
        self.rightlane = Lane('right')
        self.tracker = LaneTracker()

        # self.out = cv2.VideoWriter(str(time.time())+'testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (CFG.IMAGE_WIDTH, CFG.IMAGE_HEIGHT),True)

        self.img = np.zeros([CFG.IMAGE_WIDTH, CFG.IMAGE_HEIGHT, 3],np.uint8)
        self.yoloBoxes = BoundingBoxes()
        self.trafficLightBoxes = BoundingBoxes()

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


    def transform_input(self, img):
        _set = "IMAGENET"
        mean = IMG_MEAN[_set]
        std = IMG_STD[_set]
        # transform_img = Resize((800, 288))
        transform_img = Resize((512, 256))
        transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
        #img_org = img[255:945, :, :]
        img = transform_img({'img': img})['img']
        x = transform_x({'img': img})['img']
        # print(x)
        x.unsqueeze_(0)
        x = x.to('cuda')
        return x

    def detection(self, input):

        #startt = time.time()
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            output = self.model(input, None)

       # print('detection use：', time.time()-startt)
        return self.cluster(output)

    def cluster(self,output):
        #startt = time.time()

        global g_frameCnt

        embedding = output['embedding']
        embedding = embedding.detach().cpu().numpy()
        embedding = np.transpose(embedding[0], (1, 2, 0))
        binary_seg = output['binary_seg']
        bin_seg_prob = binary_seg.detach().cpu().numpy()
        bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]
        # seg = bin_seg_pred * 255

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=bin_seg_pred,
            instance_seg_result=embedding)

        if postprocess_result['mask_image'] is None:
            print('cant find any lane!!!')
        else:
            self.maskimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['mask_image'], "bgr8"))
            self.binimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['binary_img'], "mono8"))
            self.morphoimg_pub.publish(self.bridge.cv2_to_imgmsg(postprocess_result['morpho_img'], "mono8"))

        # cv2.imwrite(str(g_frameCnt)+'_mask.png', postprocess_result['mask_image'])
        # cv2.imwrite(str(g_frameCnt)+'_binary.png', postprocess_result['binary_img'])

        return postprocess_result


    def color(self, signal):
        if signal == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return color

    def process(self, frame):
        startt = time.time()
        cropImg = cropRoi(frame)
        input_image = self.transform_input(cropImg)
        # startt = time.time()
        postProcResult = self.detection(input_image)
        
        # debugImg = frame.copy()
        self.img = frame.copy()
        # cv2.imwrite(str(g_frameCnt)+'.png', debugImg)

        # if len(postProcResult['fit_params']) > 0:
        #     self.leftlane.updateLane(postProcResult['fit_params'])
        #     self.rightlane.updateLane(postProcResult['fit_params'])
        #     if self.leftlane.detectedLostCnt > 3 and self.rightlane.detectedLostCnt > 3:
        #         self.leftlane.initLane()
        #         self.rightlane.initLane()
        #         print('!!!!!!! detected not fit')
        #     lanePoints = {
        #         'lanes':[self.leftlane.points,self.rightlane.points]
        #     }
        #     signal = self.warning.detect(lanePoints)
        #     color = (0, 0, 255) if signal == 1 else (0, 255, 0)
        #     # color = (0, 255, 0)
        #     #draw lane
        #     for idx in range(11):
        #         cv2.line(self.img, (int(self.leftlane.points[idx][0]), int(self.leftlane.points[idx][1])), (int(self.leftlane.points[idx+1][0]), int(self.leftlane.points[idx+1][1])), color, 10)
        #         cv2.line(self.img, (int(self.rightlane.points[idx][0]), int(self.rightlane.points[idx][1])), (int(self.rightlane.points[idx+1][0]), int(self.rightlane.points[idx+1][1])), color, 10)

        self.tracker.process(postProcResult['detectedLanes'])
        llane = self.tracker.leftlane
        rlane = self.tracker.rightlane

        lanePoints = {
                'lanes':[llane.points, rlane.points]
        }
        signal = self.warning.detect(lanePoints)
        color = (0, 0, 255) if signal == 1 else (0, 255, 0)
        for idx in range(11):
            cv2.line(self.img, (int(llane.points[idx][0]), int(llane.points[idx][1])), (int(llane.points[idx+1][0]), int(llane.points[idx+1][1])), color, 10)
            cv2.line(self.img, (int(rlane.points[idx][0]), int(rlane.points[idx][1])), (int(rlane.points[idx+1][0]), int(rlane.points[idx+1][1])), color, 10)
        
        # for lane in postProcResult['detectedLanes']:
        #     if lane[0][0] == llane.detectedLane[0][0]:
        #         color = (255, 0, 0)
        #     elif lane[0][0] == rlane.detectedLane[0][0]:
        #         color = (255, 255, 0)
        #     else:
        #         color = (0,255,0)

        #     for j in range(11):
        #         cv2.line(debugImg, (lane[j][0], lane[j][1]), (lane[j+1][0], lane[j+1][1]), color, 10)
        #         cv2.line(debugImg, (0,lane[j][1]), (int(self.image_X),lane[j][1]), (0,0,255), 3)
        #         # cv2.putText(debugImg, '{}'.format(abs(int(fit_x[5])-960)), (int(fit_x[0]), int(plot_y[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
        # cv2.imwrite(str(1)+'debug.png', debugImg)



        # debug
        # plot_y = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)
        # for fit_param in postProcResult['fit_params']:
        #     fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2]
            
        #     if self.leftlane.detectedLeftLane[0][0] == int(fit_x[0]):
        #         color = (255,0,0)
        #     elif self.leftlane.detectedRightLane[0][0] == int(fit_x[0]):
        #         color = (255,255,0)
        #     else:
        #         color = (0,255,0)

        #     for j in range(11):
        #         cv2.line(debugImg, (int(fit_x[j]), int(plot_y[j])), (int(fit_x[j+1]), int(plot_y[j+1])), color, 10)
        #         cv2.line(debugImg, (0,int(plot_y[j])), (int(self.image_X),int(plot_y[j])), (0,0,255), 3)
        #     cv2.putText(debugImg, '{}'.format(abs(int(fit_x[5])-960)), (int(fit_x[0]), int(plot_y[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
            
        # cv2.imwrite(str(g_frameCnt)+'debug.png', debugImg)
        # cv2.imwrite(str(g_frameCnt)+'input_image.png', frame)

        # cv2.imwrite(str(1)+'debug.png', debugImg)
        #cv2.imwrite(str(1)+'input_image.png', cropImg)
        # debug end

        print('all use：', time.time()-startt)

    def drawYoloResult(self, data):
        for i in range(data.objNum):
            box = data.bounding_boxes[i]
            cv2.rectangle(self.img, (int(box.xmin), int(box.ymin)), (int(box.xmax), int(box.ymax)), (0, 255, 0), 3)
            t_size = cv2.getTextSize(box.id, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = int(box.xmin) + t_size[0] + 3, int(box.ymin) + t_size[1] + 4
            cv2.rectangle(self.img, (int(box.xmin), int(box.ymin)), c2, (0, 255, 0), -1)
            cv2.putText(self.img, box.id, (int(box.xmin), int(box.ymin)+10), cv2.FONT_HERSHEY_PLAIN, 1, (225, 255, 255), 1)



    def callbackRos(self, data):
        print('callbackros')

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img = cv_image.copy()
        # self.out.write(cv_image)
        
        self.process(cv_image)
        #cv2.imwrite('cvimage.png', cv_image)
        
        # cv2.imwrite('result.png', cv_image)

        self.drawYoloResult(self.yoloBoxes)
        self.drawYoloResult(self.trafficLightBoxes)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.img, "bgr8"))


def listenkeyboard():
    global g_videoPlay, g_keyboardinput, g_writevideo
    while True:
        g_keyboardinput = input()
        if g_keyboardinput == 'a':
            g_videoPlay = not g_videoPlay
        elif g_keyboardinput == 'r':
            print('g_keyboardinput rrrrrr')
            g_writevideo = True

def test():
    global g_videoPlay, g_keyboardinput, g_writevideo, g_frameCnt
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    cap = cv2.VideoCapture('/space/data/cxg/3.avi')
    ret, frame = cap.read()
    rate = rospy.Rate(10)
    rospy.loginfo('video frame cnt:%d', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    listernerTh = threading.Thread(target=listenkeyboard)
    listernerTh.setDaemon(True)
    listernerTh.start()

    # out = cv2.VideoWriter('testwrite.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (1920,1200),True)
    
    while not rospy.is_shutdown():
        while not g_videoPlay:
            time.sleep(1)
            if g_keyboardinput == 's':
                g_keyboardinput = ''
                break

        ret, frame = cap.read()
        if not ret:
            rospy.loginfo('frame end')
            break

        g_frameCnt = g_frameCnt + 1
        rospy.loginfo('frame cnt:%d', g_frameCnt)
        cv_image = frame.copy()

        # # if g_writevideo:
        # #     print('write video')
        # #     out.write(cv_image)

        ic.process(cv_image)
        ic.image_pub.publish(ic.bridge.cv2_to_imgmsg(ic.img, "bgr8"))
        rate.sleep()

    # out.release()

def cropRoi(img):

    # return img[int(vanishPtY-cropedImgHeight/2):int(vanishPtY+cropedImgHeight/2), int(vanishPtX-cropedImgWidth/2):int(vanishPtX+cropedImgWidth/2), :].copy()

    return img[CFG.CROP_IMG_Y:CFG.CROP_IMG_Y+CFG.CROP_IMG_HEIGHT, CFG.CROP_IMG_X:CFG.CROP_IMG_X+CFG.CROP_IMG_WIDTH, :].copy()

def evaluateImage(model, imagePath, outputRoot):
    global tp, fp, fn
    # imagePath = os.path.join(imageDir,img)
    labelPath = imagePath[:-3]+'lines.txt'
    frameIdx = imagePath.find('frame')
    subDirIdx = imagePath.find('.MP4')
    # imagePath[frameIdx+7:subDirIdx]
    outputPath = outputRoot + imagePath[frameIdx+7:subDirIdx] + '_' + imagePath[imagePath.rfind('/')+1:-4] + '_result.png'
    print('imagePath:',imagePath)
    print('outputpath:',outputPath)
    detectedImage = cv2.imread(imagePath)
    gtImage = detectedImage.copy()

    grayDetectedImg = np.zeros([detectedImage.shape[0], detectedImage.shape[1]],np.uint8)
    grayGtImg = np.zeros([detectedImage.shape[0], detectedImage.shape[1]],np.uint8)
    # print(imagePath)
    cropImg = cropRoi(detectedImage)
    input_image = model.transform_input(cropImg)
    postProcResult = model.detection(input_image)

    # for detectedLane in postProcResult['detectedLanes']:
    #     for i in range(11):
    #         cv2.line(detectedImage, (detectedLane[i][0], detectedLane[i][1]), (detectedLane[i+1][0], detectedLane[i+1][1]), (0,255,0), 2)
    #         #cv2.line(image, (0, detectedLane[i][1]), (int(lanedet.image_X), detectedLane[i][1]), (0,0,255), 3)
    # cv2.imwrite(imagePath[:-4]+'deteced.png', detectedImage)

    detectedLanes = []
    gtLanes = []

    curFN = 0
    curFP = 0
    curTP = 0

    # detectedLanes = np.array(())
    # gtLanes = np.array(())
    with open(labelPath, 'r') as f, open('/space/data/lane/ret_acc20_test.txt', 'a') as ret:
        lines = f.readlines()
        findMatch = 0
        if len(lines) == 0:
            print('no lines')
            return
        for line in lines:
            print('line:')
            lane = np.array(line.split())
            lane = lane.astype(np.float)
            lane = lane.astype(np.int)
            lane = lane.reshape(int(lane.size/2),2)
            # print('gt point:', lane)

            detectedLane = lane.copy()
            minDist = 500
            bestFit = []
            for fit_param in postProcResult['fit_params']:
                detectedLane[:,0] = fit_param[0] * detectedLane[:,1] ** 2 + fit_param[1] * detectedLane[:,1] + fit_param[2]
                # fit_X = fit_param[0] * detectedLane[:1] ** 2 + fit_param[1] * detectedLane[:1] + fit_param[2]
                dist = np.linalg.norm(detectedLane - lane)
                print('dist:',dist)
                # print('detect point:', detectedLane)
                if dist < minDist:
                    bestFit = detectedLane.copy()
                    minDist = dist

            if len(bestFit) > 0:
                # print('detect point:', bestFit)
                for i in range(len(lane)-1):
                    cv2.line(gtImage, (bestFit[i][0], bestFit[i][1]), (bestFit[i+1][0], bestFit[i+1][1]), (255, 0, 0), 2)
                    cv2.line(grayDetectedImg, (bestFit[i][0], bestFit[i][1]), (bestFit[i+1][0], bestFit[i+1][1]), (1), 2)

                # calculate accuracy
                for pt in bestFit:
                    detectedLanes.append(pt.tolist())

                for pt in lane:
                    gtLanes.append(pt.tolist())

                # calculate 
                findMatch += 1
            else:
                curFN += 1

            for i in range(len(lane)-1):
                cv2.line(gtImage, (lane[i][0], lane[i][1]), (lane[i+1][0], lane[i+1][1]), (0, 255, 0), 2)
                cv2.line(grayGtImg, (lane[i][0], lane[i][1]), (lane[i+1][0], lane[i+1][1]), (1), 2)
        
        curTP = findMatch
        curFP = len(postProcResult['fit_params']) - findMatch


        detectedLanes = np.array(detectedLanes)
        gtLanes = np.array(gtLanes)
        tmp = np.sum(np.where(np.abs(detectedLanes - gtLanes) < 20, 1., 0.)) / gtLanes.size
        cv2.putText(gtImage, 'pixel wise accuracy:{}'.format(tmp), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), thickness=2)


        print('acc:',tmp)
        if tmp > 0.96:
            ret.write(imagePath+'\n')
            if curFN == 0:
                cv2.imwrite(outputPath, gtImage)

        fp += curFP
        tp += curTP
        fn += curFN

        if tp+fp == 0:
            precision = 0
        else:
            precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        print('precision:',precision,',recall:',recall)


        sumGt = grayGtImg.sum()
        sumDetected = grayDetectedImg.sum()
        inter = (grayGtImg * grayDetectedImg).sum()
        union = sumGt+sumDetected-inter
        iou = inter/union
        print('iou:',iou)
        


def testImage():
    lanedet = Lane_warning()
    imgs = glob.glob('/space/data/lane/culane/*.jpg')
    rootPath = '/space/data/CUlane'
    outputRoot = '/space/data/lane/culane/acc20_test/'
    listImagePath = '/space/data/lane/culane/test.txt'
    print(imgs)
    print(float('-0.534045'))
    with open(listImagePath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            imagePath = rootPath + line[:-1]
            evaluateImage(lanedet, imagePath, outputRoot)

def testOneFrame():
    lanedet = Lane_warning()
    imgs = glob.glob('/space/code/rosadas/tmp/*.png')
    imgnum = len(imgs)
    for i in range(imgnum):
        frame = cv2.imread(imgs[i])
        cropImg = cropRoi(frame)
        input_image = lanedet.transform_input(cropImg)
        postProcResult = lanedet.detection(input_image)
        cv2.imwrite(str(i)+'mask.png', postProcResult['mask_image'])
        cv2.imwrite(str(i)+'morph.png', postProcResult['morpho_img'])
        cv2.imwrite(str(i)+'bin.png', postProcResult['binary_img'])



def main(args):
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    rospy.loginfo('model init end')
    rospy.spin()


if __name__ == "__main__":
    # main(sys.argv)
    test()
    # testOneFrame()








