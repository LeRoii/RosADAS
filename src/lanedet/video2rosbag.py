import time, sys, os
from ros import rosbag
import roslib, rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

from pynput import keyboard
import threading,time

TOPIC = '/camera/rgb/image_raw'
videopath = '/space/data/road/3.mp4'
bagname = '3.bag'

def CreateVideoBag(videopath, bagname):
    '''Creates a bag file with a video file'''
    print(videopath)
    print(bagname)
    bag = rosbag.Bag(bagname, 'w')
    cap = cv2.VideoCapture(videopath)
    cb = CvBridge()
    # prop_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)  # 源代码是这个,不能正常运行
    prop_fps = cap.get(cv2.CAP_PROP_FPS)  # 我该成了这个
    #prop_fps = 24 # 我手机拍摄的是29.78，我还是转成24的。
    print(prop_fps)
    ret = True
    frame_id = 0
    while(ret):
        ret, frame = cap.read()
        if not ret:
            break
        stamp = rospy.rostime.Time.from_sec(float(frame_id) / prop_fps)
        frame_id += 1
        image = cb.cv2_to_imgmsg(frame, encoding='bgr8')
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        bag.write(TOPIC, image, stamp)
    cap.release()
    bag.close()

def on_press(key):
    '按下按键时执行。'
    # try:
    #     print('alphanumeric key {0} pressed'.format(
    #         key.char))
    # except AttributeError:
    #     print('special key {0} pressed'.format(
    #         key))
    #通过属性判断按键类型。

    if key == keyboard.Key.space:
        print('key {0} pressed'.format(key))

def on_release(key):
    '松开按键时执行。'
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def listenkeyboard():
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

def testlistener():
    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

    print('asdadas')


if __name__ == "__main__":
    # CreateVideoBag(videopath, bagname)
    # testlistener()
    t1 = threading.Thread(target=listenkeyboard)
    t1.setDaemon(True)
    t1.start()
    while True:
        print('asdadas')
        time.sleep(1)
