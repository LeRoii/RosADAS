import os
# import socket
import struct
from util import *
#------UDP IP and Port-------
# Destination_IP   = "192.168.0.102"
# Destination_Port = 9002
# udpSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sendArr   = (Destination_IP, Destination_Port)

UDP_class_id = [5, 4, 2, 2, 10, 9, 8, 6, 6, 11, 12]
classes = load_classes('/home/iairiv/ros_pkgs/src/traffic_light_detection/src/tf.names')


def UDP_data_transform(detections_result, cls_thresh):
    UDP_package = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    # UDP_package = ["FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF"]
    if detections_result is None:
        # print('aaa')
        # output = ''.join(UDP_package)
        print(UDP_package)
        # output = struct.pack('B', int(UDP_package[0], 16))
        # for i in range(13):
        #     output += struct.pack('B', int(UDP_package[i+1], 16))
        output = bytearray(UDP_package)
        # print(output)
    else:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections_result:
            if cls_pred > cls_thresh:
                UDP_package[UDP_class_id[int(cls_pred)]] = 0xff
            # print(UDP_package)
            # output = struct.pack('B', int(UDP_package[0], 16))
            # for i in range(13):
            #     output += struct.pack('B',int(UDP_package[i+1], 16))
            output = bytearray(UDP_package)
    print(output)
    return output


def UDP_data_transform_queue(detect_queue, detections_result, img_size, cls_thresh):
    UDP_package = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    UDP_package_queue_data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    # UDP_package = ["FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF"]
    x_max = 0
    x_min = img_size
    x_max_color = 0x00
    x_min_color = 0x00
    count = 0
    if detections_result is None:
        # print('aaa')
        # output = ''.join(UDP_package)
        # print(UDP_package_queue_data)
        # output = struct.pack('B', int(UDP_package[0], 16))
        # for i in range(13):
        #     output += struct.pack('B', int(UDP_package[i+1], 16))
        output = bytearray(UDP_package)
        # print(output)
    else:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections_result:
            if cls_conf > cls_thresh and y2 < img_size / 2 and abs(y2-y1) > 15:
                # UDP_package_queue_data[UDP_class_id[int(cls_pred)]] = 0xff
                # if detect_queue[2][UDP_class_id[int(cls_pred)]] == 0xff or detect_queue[1][UDP_class_id[int(cls_pred)]] == 0xff:
                #     print('\t+ Label: %s, Cls_Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                #     UDP_package[UDP_class_id[int(cls_pred)]] = 0xff
                if cls_pred < 5: # red
                    color = 0x0f
                if 5<= cls_pred < 10: # green
                    color = 0xff
                if x_max < x1:
                    x_max = x1
                    x_max_color = color
                if x1 < x_min:
                    x_min = x1
                    x_min_color = color
                count = count + 1
        if count > 1:
            if x_min_color == 0xff:
                UDP_package[4] = 0xff
            if x_min_color == 0x0f:
                UDP_package[2] = 0xff
            if x_max_color == 0xff:
                UDP_package[5] = 0xff
            if x_max_color == 0x0f:
                UDP_package[3] = 0xff
            # print(UDP_package)
            # output = struct.pack('B', int(UDP_package[0], 16))
            # for i in range(13):
            #     output += struct.pack('B',int(UDP_package[i+1], 16))
        output = bytearray(UDP_package)
    detect_queue.popleft()
    detect_queue.append(UDP_package_queue_data)
    print(output)
    # return output


