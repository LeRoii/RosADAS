import sys
#sys.path.append("/home/iairiv/code/adas0903/src/lanedet/Ultra-Fast-Lane-Detection/")
#print(sys.path)

# import configs.testconfig
from configs import testconfig
from model.model import parsingNet
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.special
from data.constant import culane_row_anchor, tusimple_row_anchor

CFG = testconfig.cfg


class LaneDetect:
    def __init__(self):
        torch.backends.cudnn.benchmark = True

        if CFG.dataset == 'CULane':
            self.cls_num_per_lane = 18
        elif CFG.dataset == 'Tusimple':
            self.cls_num_per_lane = 56
        else:
            raise NotImplementedError

        self.net = parsingNet(pretrained = False, backbone=CFG.backbone,cls_dim = (CFG.griding_num+1,self.cls_num_per_lane,4),
                    use_aux=False).cuda()

        state_dict = torch.load(CFG.test_model, map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()

        self.imgTransform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.row_anchor = culane_row_anchor

        pass

    def process(self, image):
        detectedLanes = []
        transformedImg = self.imgTransform(image)
        transformedImg = transformedImg.cuda()
        transformedImg = transformedImg.unsqueeze_(0)
        with torch.no_grad():
            out = self.net(transformedImg)

        col_sample = np.linspace(0, 800 - 1, CFG.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(CFG.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == CFG.griding_num] = 0
        out_j = loc

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                x = []
                y = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * CFG.imgWidth / 800) - 1, int(CFG.imgHeight * (self.row_anchor[self.cls_num_per_lane-1-k]/288)) - 1 )
                        # cv2.circle(img,ppp,5,(0,255,0),-1)
                        x.append(ppp[0])
                        y.append(ppp[1])

                x = np.array(x)
                y = np.array(y)
                param = np.polyfit(y, x, 2)
                # plotY = np.linspace(700,1100,12)
                # plotX = param[0] * plotY ** 2 + param[1] * plotY + param[2]

                # plotY = plotY.astype(int)
                # plotX = plotX.astype(int)

                lanePoints = np.zeros([12,2], dtype = int)
                lanePoints[:,1] = np.linspace(CFG.LANE_START_Y, CFG.LANE_END_Y, 12)
                lanePointsX = param[0] * lanePoints[:,1] ** 2 + param[1] * lanePoints[:,1] + param[2]
                lanePoints[:,0] = lanePointsX.astype(np.int)
                detectedLanes.append(lanePoints)

        return detectedLanes
        