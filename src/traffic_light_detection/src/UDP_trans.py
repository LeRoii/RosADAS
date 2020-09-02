import socket
from util import *
from collections import deque

class TrafficLight_UDP(object):
    def __init__(self, dest_IP, dest_port, max_objs=50):
        self.sendArr = (dest_IP, dest_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.UDP_class_id = [5, 4, 2, 2, 10, 9, 8, 6, 6, 11, 12]
        self.classes = load_classes('/home/iairiv/ros_pkgs/src/traffic_light_detection/src/tf.names')
        # self.send_empty_list = [0,0,0,0]
        # self.max_objs = max_objs
        # for i in range(self.max_objs * 5):
        #     self.send_empty_list.append(0)
        self.detect_queue = deque(
                        [[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                        [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
                        [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]])
        
    # def send_message(self, cur_time, detections):
    #     time_stamp = cur_time.split('_')
    #     time_stamp = [int(x) for x in time_stamp]
    #     send_buff = self.send_empty_list[:]
    #     send_buff[:4] = time_stamp
    #     if detections is None:
    #         self.socket.sendto(bytes(send_buff), self.sendArr)
    #     else:
    #         i = 4
    #         for x in detections:
    #             # if the new inputs expand the length of the buff, break
    #             if i + 5 > len(send_buff):
    #                 break
    #             box = [int(z) for z in x[1:5]]
    #             cls = int(x[-1])
    #             send_buff[i] = cls
    #             send_buff[i+1:i+5] = box
    #             i += 5
    #         self.socket.sendto(bytes(send_buff), self.sendArr)

    def UDP_data_transform_queue(self, detections_result, img_size, cls_thresh):
        UDP_package = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        UDP_package_queue_data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # UDP_package = ["FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF", "FF"]
        x_max = 0
        x_min = img_size
        x_max_color = 0x00
        x_min_color = 0x00
        count = 0
        print 'UDP'
        if detections_result is None:
            # print('aaa')
            # output = ''.join(UDP_package)
            # print(UDP_package_queue_data)
            # output = struct.pack('B', int(UDP_package[0], 16))
            # for i in range(13):
            #     output += struct.pack('B', int(UDP_package[i+1], 16))
            output = bytearray(UDP_package)
            print(output)
        else:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections_result:
                if cls_conf > cls_thresh and y2 < img_size / 2 and abs(y2-y1) > 15:
                    # UDP_package_queue_data[UDP_class_id[int(cls_pred)]] = 0xff
                    # if self.detect_queue[2][UDP_class_id[int(cls_pred)]] == 0xff or self.detect_queue[1][UDP_class_id[int(cls_pred)]] == 0xff:
                    #     print('\t+ Label: %s, Cls_Conf: %.5f' % (self.classes[int(cls_pred)], cls_conf.item()))
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
        self.detect_queue.popleft()
        self.detect_queue.append(UDP_package_queue_data)
        print(output)
        self.socket.sendto(output, self.sendArr)
        # return output

    def UDP_data_transform(self, detections_result, cls_thresh):
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
                    UDP_package[self.UDP_class_id[int(cls_pred)]] = 0xff
                # print(UDP_package)
                # output = struct.pack('B', int(UDP_package[0], 16))
                # for i in range(13):
                #     output += struct.pack('B',int(UDP_package[i+1], 16))
                output = bytearray(UDP_package)
        print(output)
        return output