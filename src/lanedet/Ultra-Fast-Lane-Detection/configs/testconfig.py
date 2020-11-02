from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.backbone = '18'
__C.dataset = 'CULane'
__C.griding_num = 200
# __C.test_model = '/home/iairiv/code/model/culane_18.pth'
__C.test_model = '/space/model/culane_18.pth'
__C.imgWidth = 1920
__C.imgHeight = 1200
__C.LANE_START_Y = 650
__C.LANE_END_Y = 1100
__C.DETECTED_DIFF_THRESH = 400