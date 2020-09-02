import socket
import struct

class YOLO_UDP(object):
    def __init__(self, dest_IP, dest_port, send_port=7900, max_objs=50):
        self.transform_label = {0:30, 1:40, 2:20, 3:41, 5:21, 7:22, 80:50}
        self.label_name = {0:'person', 1:'bike', 2:'car', 3:'motorbike', 5:'bus', 7:'truck', 80:'rider'}
        self.sendArr = (dest_IP, dest_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("", send_port))
        self.send_empty_list = [0,0,0,0,0] #hour min, sec, usec,obj_num
        self.max_objs = max_objs
        for i in range(self.max_objs * 5):
            self.send_empty_list.append(0)
        
    def send_message(self, cur_time, detections):
        time_stamp = cur_time.split('_')
        time_stamp = [int(x) for x in time_stamp]
        send_buff = self.send_empty_list[:]
        send_buff[:4] = time_stamp
        if detections is None:
            send_buff = struct.pack('h'*len(send_buff), *send_buff)
            self.socket.sendto(bytes(send_buff), self.sendArr)
        else:
            i = 5
            count = 0
            for x in detections:
                # if the new inputs expand the length of the buff, break
                if i + 5 > len(send_buff):
                    break
                box = [int(z) for z in x[1:5]]
                cls = self.transform_label[int(x[-1])]
                send_buff[i] = cls
                send_buff[i+1:i+5] = box
                #print time_stamp, self.label_name[int(x[-1])], box
                i += 5
                count += 1
            send_buff[4] = count
            # print send_buff
            send_buff = struct.pack('h'*len(send_buff), *send_buff)
            # print(send_buff)
            self.socket.sendto(bytes(send_buff), self.sendArr)