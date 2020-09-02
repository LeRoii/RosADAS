#!/usr/bin/env python3
import os
# import xml.dom.minidom
import cv2
import sys
import rospy
from sensor_msgs.msg import Image
# from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import time
from config import *
from model import LaneNet
from utils.transforms import *
from utils.postprocess import *
import time
import torch.backends.cudnn as cudnn
import os
from utils.prob2lines import getLane
from utils.lanenet_postprocess import LaneNetPostProcessor
import matplotlib.pyplot as plt
from Detection import Detection

# from lanedet.msg import BoundingBoxes
# from lanedet.msg import BoundingBox
i=1
class Lane_warning:
    def __init__(self):
        self.image_pub = rospy.Publisher("lanedetframe", Image,queue_size = 1)
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("YOLO_detect_result", Image, self.callbackRos)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callbackRos)
        # self.yolobbox_sub = rospy.Subscriber("publishers/bounding_boxes/topic", BoundingBoxes, self.callbackRos)
        self.weights_file = '/home/iairiv/code/lane_waring_final/experiments/exp1/exp1_best.pth'
        self.CUDA = torch.cuda.is_available()
        self.postprocessor = LaneNetPostProcessor()
        self.warning = Detection()
        self.band_width = 1.5
        self.image_X = 1920
        self.image_Y = 1200
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

    def transform_input(self, img):
        return prep_image(img)

    def detection(self, input,raw):
        
        if self.CUDA:
            input = input.cuda()
        with torch.no_grad():
            output = self.model(input, None)

        
        return self.cluster(output,raw)

    def cluster(self,output,raw):
        global i
        embedding = output['embedding']
        embedding = embedding.detach().cpu().numpy()
        embedding = np.transpose(embedding[0], (1, 2, 0))
        binary_seg = output['binary_seg']
        bin_seg_prob = binary_seg.detach().cpu().numpy()
        bin_seg_pred = np.argmax(bin_seg_prob, axis=1)[0]

        
        # plt.savefig('a.png')
        # i = time.time()
        cv2.imwrite(str(i)+'a.png',bin_seg_pred)
        i=i+1
        # cv2.waitKey(0)
        #plt.show()
        seg = bin_seg_pred * 255
        # print('postprocess_result')
        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=bin_seg_pred,
            instance_seg_result=embedding
        )

        prediction = postprocess_result
        prediction = np.array(prediction)
        
        return prediction
    """"没加预警"""
    # def write(self, output, img):
    #     # output[:,:,1] = output[:,:,1]+255
    #     for i in range(len(output)):
    #         line = np.array(output[i])
    #         line[:,1] = line[:,1]+255
    #         output[i] = line.tolist()
    #         # for j in range(len(output[i])):
    #             # output[i][j][1] = output[i][j][1] + 255
    #             # print(arr[i][j])
    #             # cv.circle(image, (int(arr[i][j][0]),int(arr[i][j][1])), 5, (0, 0, 213), -1) #画成小圆点
    #         cv.line(img, (int(output[i][0][0]), int(output[i][0][1])), (int(output[i][-1][0]), int(output[i][-1][1])),
    #                 (0,0,255), 3)
    #     # if signal == 1:
    #     #     cv2.putText(img, "WARNING", (1300, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, thickness=10)
    #     #plt.imshow(img)
    #     #plt.show()
    #     return img
    """""加了预警"""""
    # def write(self, output, img,signal,color):
    #     # output[:,:,1] = output[:,:,1]+255
    #     for i in range(len(output)):
    #         line = np.array(output[i])
    #         # line[:,1] = line[:,1]+255
    #         output[i] = line.tolist()
    #         #for j in range(len(output[i])):
    #             #output[i][j][1] = output[i][j][1] + 255
    #             #print(output[i][j])
    #             #cv2.circle(img, (int(output[i][j][0]),int(output[i][j][1])), 5, color, -1) #画成小圆点
    #         cv2.line(img, (int(output[i][0][0]), int(output[i][0][1])), (int(output[i][-1][0]), int(output[i][-1][1])),color, 3)
    #     if signal == 1:
    #         cv2.putText(img, "WARNING", (1300, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, thickness=10)
    #     #plt.imshow(img)
    #     #plt.show()
    #     return img

    def write_nowarning(self, output, img):
        # output[:,:,1] = output[:,:,1]+255
        for i in range(len(output)):
            line = np.array(output[i])
            # line[:,1] = line[:,1]+255
            output[i] = line.tolist()
            #for j in range(len(output[i])):
                #output[i][j][1] = output[i][j][1] + 255
                #print(output[i][j])
                #cv2.circle(img, (int(output[i][j][0]),int(output[i][j][1])), 5, color, -1) #画成小圆点
            cv2.line(img, (int(output[i][0][0]), int(output[i][0][1])), (int(output[i][-1][0]), int(output[i][-1][1])),(0,0,255), 3)
        #plt.imshow(img)
        #plt.show()
        return img

    def color(self, signal):
        if signal == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        return color
        
    #ros下的代码，还没测试过。无ros用另一个测。
    def callbackRos(self, data):
        # print('callbackros')
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            input_image = self.transform_input(cv_image)
            prediction = self.detection(input_image, cv_image)
            if len(prediction) == 0:
                result = cv_image
            else:
                print(prediction)
                # signal = self.warning.detect(prediction)
                # color = self.color(signal)
                # result = self.write(prediction, cv_image, signal, color)
                # result = self.write_nowarning(prediction, cv_image)
        except CvBridgeError as e:
            print(e)
    
        # cv2.imshow("image windows", result)
        # cv2.waitKey(3)
        # try:
        #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)

        
    def callback(self, data):
        # try:
            # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.resize(cv_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        time3 = time.time()
        input_image = self.transform_input(data)
        time4 = time.time()-time3
        print('数据预处理时间:',time4)

        #lane_detect
        time5 = time.time()
        prediction = self.detection(input_image,data)
        # if len(prediction) == 0:
        #     prediction = self.lastlane
        # else:
        #     self.lastlane = prediction


        #print(prediction)
        time6 = time.time()-time5
        print('检测时间：', time6)

        #warning
        time7 = time.time()
        signal = self.warning.detect(prediction)
        color = self.color(signal)
        time8 = time.time()-time7
        print('预警时间：',time8)


        #draw_line
        time1 = time.time()
        # img = self.write(prediction, data)
        img = self.write(prediction, data, signal, color)
        time2 = time.time()-time1
        print('画图时间：',time2)

        cv2.imshow("final",img)
        cv2.waitKey(0)
        #plt.show()


def prep_image(img):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    _set = "IMAGENET"
    mean = IMG_MEAN[_set]
    std = IMG_STD[_set]
    # transform_img = Resize((800, 288))
    transform_img = Resize((512, 256))
    transform_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
    # img_org = img[255:945, :, :]
    img_org = img[:, :,0:800]
    img_org = img
    img = transform_img({'img': img_org})['img']
    x = transform_x({'img': img})['img']
    # print(x)
    x.unsqueeze_(0)
    x = x.to('cuda')
    return x

def main(args):
    ic = Lane_warning()
    rospy.init_node("lanedetnode", anonymous=True)
    rospy.loginfo('model init end')
    rospy.spin()
    # while not rospy.is_shutdown():
    #     print("laneeemain")
    #     try:
    #         rospy.spin()
    #     except KeyboardInterrupt:
    #         print("Shutting down")
    #     rate.sleep()
    print('endddd')

if __name__ == "__main__":
    # imgPath = '/home/iairiv/data/fc2_save_2020-06-03-110140-0000/images/20.jpg'
    # image = cv2.imread(imgPath)
    # cv2.imshow("12313",image)
    # cv2.waitKey(0)
    
    main(sys.argv)
    # img = cv2.imread('/home/iairiv/Desktop/1.png')
    # #cv2.imshow('111',img)
    
    # ic = Lane_warning()
    # #ic.callback(image)
    # image_path = '/home/iairiv/data/fc2_save_2020-06-03-110140-0000/images'
    # imagedir = os.listdir(image_path)
    # imagedir.sort()
    # for i in imagedir:
    #     print(i)
    #     image = os.path.join(image_path,i)
    #     image = cv2.imread(image)
    #     ic.callback(image)








