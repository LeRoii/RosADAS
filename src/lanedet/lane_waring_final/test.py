import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
from PIL import Image

def PicToVideo(imgPath, videoPath):
    images = os.listdir(imgPath)
    images.sort()
    # images.sorted()
    print(images)
    fps = 6
    fourcc = VideoWriter_fourcc(*"MJPG")
    im = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, im.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])
        print(im_name)
        videoWriter.write(frame)
    videoWriter.release()


def unlock_mv(sp):
    """ 将视频转换成图片
        sp: 视频路径 """
    cap = cv2.VideoCapture(sp)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
        print(frame_count)
        frame_count += 1
        suc, frame = cap.read()
        params = []
        params.append(2)  # params.append(1)
        cv2.imwrite('1_%d.jpg' % frame_count, frame, params)

    cap.release()
    print('unlock image: ', frame_count)


imgPath = "/home/iairiv/data/kitti/"
videoPath = "/home/iairiv/data/kitti.avi"
# unlock_mv(videoPath)
PicToVideo(imgPath, videoPath)

