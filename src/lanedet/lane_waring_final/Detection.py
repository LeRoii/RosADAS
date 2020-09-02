import numpy as np
import cv2

class Detection(object):
    """参数定义"""
    #car_X = 440  # 需要标定得到
    #car_Y = 26.059701492537283  # 需要标定得到

    image_x = 1920
    image_y = 1200
    formcar_X = image_x / 2
    formcar_Y = image_y

    half_car_weight = 0

    derta_time = 1  # 0.1s（需要声明）

    mid_distances = []  # 存放两次传输的车子距离车道中心的距离

    temp_dis = []

    def __init__(self):
        self.warningTimeDelay = 0

    def cal_shortest_dis(self, car_X, car_Y, dot1_x, dot1_y, dot2_x, dot2_y):
            """计算车上一点（car_X, car_Y）距离两侧车道线的距离，并返回距离两个车道线距离"""

            # 第一个边长   car--1
            a = ((dot1_x - car_X) ** 2 + (dot1_y - car_Y) ** 2) ** 0.5
            # 第二个边长   car--2
            b = ((dot2_x - car_X) ** 2 + (dot2_y - car_Y) ** 2) ** 0.5
            # 第三个边长   1--2
            c = ((dot1_x - dot2_x) ** 2 + (dot1_y - dot2_y) ** 2) ** 0.5
            # 计算半周长
            s = (a + b + c) / 2
            # 计算三角形面积
            area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
            # 计算以 1--2为底的三角形的高
            result = area * 2 / c

            return result

    def cal_time(self, distance, theta, speed):
            time = 999
            """
            计算偏离车道需要的时间：
            theta > 0:车向右偏，反之向左
            """
            if (theta == 1 and speed != 0):
                time = distance[1] / speed
            elif (theta == 0 and speed != 0):
                time = distance[0] / speed
            elif (theta == 2 or speed == 0):
                time = 99999
            return time

    def cal_theta(self, mid_distances):
        """theta == 1  向右
            theta == 0 向左
            theta == 2 启动状态"""
        sum = 0
        if(len(mid_distances) == 5):
            for i in range(len(mid_distances) - 1):
                sum += (mid_distances[-1] - mid_distances[i]) * (i+1) * 0.1
            if sum > 0:
                return 1
            if sum < 0:
                return 0
        else: return 2

    def mid_dis(self, mid_distances, now_middistance, state):
            if state == 1:
                mid_distances.clear()


            if len(mid_distances) < 5:
                mid_distances.append(now_middistance)
            else:
                mid_distances[0] = mid_distances[1]
                mid_distances[1] = mid_distances[2]
                mid_distances[2] = mid_distances[3]
                mid_distances[3] = mid_distances[4]
                mid_distances[4] = now_middistance

            return mid_distances

    def multi_to_one(self, dict):
        result = 0

        return result

    def cal_conv_lane_X(self, X, Y, ):
            """将X坐标转鸟瞰图下的坐标"""
            pts = np.float32([[606, 900], [499, 1000], [1384, 1000], [1243, 900]])
            ptsl = np.float32([[630, 700], [630, 900], [900, 900], [900, 700]])
            M = cv2.getPerspectiveTransform(pts, ptsl)
            M = np.transpose(M)

            x = (X * M[0][0] + Y * M[1][0] + M[2][0]) / (X * M[0][2] + Y * M[1][2] + M[2][2])
            y = (X * M[0][1] + Y * M[1][1] + M[2][1]) / (X * M[0][2] + Y * M[1][2] + M[2][2])
            return x

    def cal_conv_lane_Y(self, X, Y, ):
            """将纵坐标转鸟瞰图下的坐标"""
            pts = np.float32([[606, 900], [499, 1000], [1384, 1000], [1243, 900]])
            ptsl = np.float32([[630, 700], [630, 900], [900, 900], [900, 700]])
            M = cv2.getPerspectiveTransform(pts, ptsl)
            M = np.transpose(M)

            x = (X * M[0][0] + Y * M[1][0] + M[2][0]) / (X * M[0][2] + Y * M[1][2] + M[2][2])
            y = (X * M[0][1] + Y * M[1][1] + M[2][1]) / (X * M[0][2] + Y * M[1][2] + M[2][2])
            return y

    def cal_conv(self, dict):
        arr_X = []
        arr_X.append([])
        arr_X.append([])
        arr_Y = []
        arr_Y.append([])
        arr_Y.append([])
        conv_X = [[], []]
        conv_Y = [[], []]
        lanes_num = 2

        for i in range(len(dict['lanes'][0])):
            arr_X[0].append(dict['lanes'][0][i][0])
            arr_X[1].append(dict['lanes'][1][i][0])
            arr_Y[0].append(dict['lanes'][0][i][1])
            arr_Y[1].append(dict['lanes'][1][i][1])

        for i in range(len(arr_X)):
            if(all(_ == 0 for _ in arr_X[i])):
                lanes_num -= 1



        for k in range(len(arr_X)):
            for i in range(len(arr_X[k])):
                conv_X[k].append(int(self.cal_conv_lane_X(arr_X[k][i], arr_Y[k][i])))
                conv_Y[k].append(int(self.cal_conv_lane_Y(arr_X[k][i], arr_Y[k][i])))

        return conv_X, conv_Y, lanes_num

    def cal_speed(self, mid_distances, derta_time):
        speed = 0
        if len(mid_distances) == 5:
            for i in range(len(mid_distances) - 1):
                speed += (abs(mid_distances[i] - mid_distances[-1]) * 0.1 * (i+1) / ((len(mid_distances) - i - 1) * derta_time))
            #speed /= (len(mid_distances) - 1)
        else: return 1
        return speed

    def cal_avr_dis(self, temp_dis, dis, dis2):            # dis: 左侧距离   dis2：右侧距离
        left = right = 0
        temp = []
        state = 0
        temp.append(dis)
        temp.append(dis2)         #  temp[left, right], 0：距离左侧车道线的距离  1：距离右侧车道线的距离
        """若车道线检测有错误，则采取数据抛出措施"""
        if(len(temp_dis) == 0):
            temp_dis.append(temp)
        else:
            if(abs(temp_dis[-1][0] - dis) / max(temp_dis[-1][0], dis) < 0.3  and abs(temp_dis[-1][1] - dis2) / max(temp_dis[-1][1], dis2) < 0.3) or \
                    abs(temp_dis[-1][0] - dis) < 10 and abs(temp_dis[-1][1] - dis2) < 10:
                if(len(temp_dis) == 5):

                    temp_dis[0] = temp_dis[1]
                    temp_dis[1] = temp_dis[2]
                    temp_dis[2] = temp_dis[3]
                    temp_dis[3] = temp_dis[4]
                    temp_dis[4]= temp
                else:
                    temp_dis.append(temp)
            elif abs(temp_dis[-1][0] - dis) > 30 or abs(temp_dis[-1][1] - dis2) > 30:
                state = 1
                temp_dis.clear()
                temp_dis.append(temp)
        #print(temp_dis)

        """对前五帧的数据取平均值来降低误差"""
        for i in range(len(temp_dis)):
            left += temp_dis[i][0]
            right += temp_dis[i][1]
        left /= len(temp_dis)
        right /= len(temp_dis)

        temp_dis[-1][0] = left
        temp_dis[-1][1] = right
        return left, right, state

    def detect(self, dict):
            """
            dict =
            {'lanes': [
            [[897, 650], [890, 660], [884, 670], [878, 680], [872, 690], [866, 700], [860, 710], [854, 720], [848, 730],
             [843, 740], [837, 750], [832, 760], [827, 770], [822, 780], [817, 790], [812, 800], [808, 810], [803, 820],
             [799, 830], [795, 840], [790, 850], [786, 860], [782, 870], [779, 880], [775, 890], [771, 900], [768, 910],
             [765, 920], [761, 930], [758, 940], [755, 950], [752, 960], [750, 970], [747, 980], [745, 990],
             [742, 1000], [740, 1010], [738, 1020], [736, 1030], [734, 1040], [732, 1050], [731, 1060], [729, 1070],
             [728, 1080], [726, 1090], [725, 1100]],
            [[979, 650], [995, 660], [1012, 670], [1028, 680], [1044, 690], [1061, 700], [1077, 710], [1093, 720],
             [1110, 730], [1126, 740], [1142, 750], [1158, 760], [1175, 770], [1191, 780], [1207, 790], [1223, 800],
             [1239, 810], [1255, 820], [1271, 830], [1288, 840], [1304, 850], [1320, 860], [1336, 870], [1352, 880],
             [1368, 890], [1383, 900], [1399, 910], [1415, 920], [1431, 930], [1447, 940], [1463, 950], [1479, 960],
             [1494, 970], [1510, 980], [1526, 990], [1542, 1000], [1557, 1010], [1573, 1020], [1589, 1030],
             [1605, 1040], [1620, 1050], [1636, 1060], [1651, 1070], [1667, 1080], [1683, 1090], [1698, 1100]]],
            'num': 0}
            """
            conv_X, conv_Y, lanes_num = self.cal_conv(dict)
            #print("车道线数目：")
            #print(lanes_num)
            if(lanes_num < 2):
                print("车道线数目检测有问题")
                print("------------------------------------------------")
                return 4

            convcar_X = self.cal_conv_lane_X(Detection.formcar_X, Detection.formcar_Y)
            convcar_Y = self.cal_conv_lane_Y(Detection.formcar_X, Detection.formcar_Y)

            distance = []
            dis = self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[0][-10], conv_Y[0][-10], conv_X[0][-2], conv_Y[0][-2])  # 距离左侧车道线的距离
            dis += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[0][-8], conv_Y[0][-8], conv_X[0][-2], conv_Y[0][-2])  # 距离左侧车道线的距离
            dis += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[0][-6], conv_Y[0][-6], conv_X[0][-2], conv_Y[0][-2])  # 距离左侧车道线的距离
            dis += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[0][-4], conv_Y[0][-4], conv_X[0][-2], conv_Y[0][-2])  # 距离左侧车道线的距离

            dis2 = self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[1][-10], conv_Y[1][-10], conv_X[1][-2], conv_Y[1][-2])  # 距离右侧车道线的距离
            dis2 += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[1][-8], conv_Y[1][-8], conv_X[1][-2], conv_Y[1][-2])  # 距离右侧车道线的距离
            dis2 += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[1][-6], conv_Y[1][-6], conv_X[1][-2], conv_Y[1][-2])  # 距离右侧车道线的距离
            dis2 += self.cal_shortest_dis(convcar_X, convcar_Y, conv_X[1][-4], conv_Y[1][-4], conv_X[1][-2], conv_Y[1][-2])  # 距离右侧车道线的距离

            left, right, state = self.cal_avr_dis(Detection.temp_dis, dis/4, dis2/4)

            distance.append(left)
            distance.append(right)
            #print("3帧的情况：")
            #print(state)                           # state = 1：出现车道线检测不是很准的情况   state = 0：车道线检测较准确，运行正常
            #print("车距离两侧车道线的距离：")
            #print(distance)
            dis_mid = (distance[0] - distance[1]) / 2

            mid_distances = self.mid_dis(Detection.mid_distances, dis_mid, state)  # 更新mid_distances表
            #print("当前帧与上一帧车距离车道中心线的距离：")
            #print(mid_distances)

            theta = self.cal_theta(mid_distances)
            #if theta == 1:
            #    print("右")
            #if theta == 0:
            #    print("左")

            speed = self.cal_speed(mid_distances, Detection.derta_time)
            #print("速度：")
            #print(speed)

            time = self.cal_time(distance, theta, speed)
            #print("车驶离车道线的时间：")
            #print(time)

            self.warningTimeDelay = (self.warningTimeDelay - 1) if self.warningTimeDelay > 0 else 0
            if (len(Detection.mid_distances) < 5):
                print("数据未加载完成")
                print("------------------------------------------------")
                return 3

            elif time < 15 or (self.cal_theta(mid_distances) == 1 and distance[1] < 90 - Detection.half_car_weight) or (self.cal_theta(mid_distances) == 0 and distance[0] < 90):
                if self.warningTimeDelay == 0:
                    self.warningTimeDelay = 30
                    print("偏离")
                    print("------------------------------------------------")
                    return 1
                else:
                    print('self.warningTimeDelay:',self.warningTimeDelay)
                    return 0
            else:
                print("未偏")
                print("------------------------------------------------")
                return 0


