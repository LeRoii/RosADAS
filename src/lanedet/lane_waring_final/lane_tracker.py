import numpy as np

from lane_obj import Lane
import global_config

CFG = global_config.cfg

class LaneTracker:
    def __init__(self):
        self.leftlane = Lane('left')
        self.rightlane = Lane('right')
        self.detectedLeftLane = np.zeros([12,2], dtype = int)
        self.detectedRightLane = np.zeros([12,2], dtype = int)

        pass

    def process(self, detectedLanes):
        self.detectedLeftLane = np.zeros([12,2], dtype = int)
        self.detectedRightLane = np.zeros([12,2], dtype = int)
        
        # find candidate left right lane
        # sort in ascending order by the distance between x coordinate of 10th point of the lane and 1/2 image width
        detectedLanes.sort(key=(lambda x : abs(x[10][0] - CFG.IMAGE_WIDTH/2)))
        if len(detectedLanes) > 1:
            # sort first two lane
            sortedDetectedLane = sorted(detectedLanes[:2], key=(lambda x : x[5][0]))
            self.detectedLeftLane = sortedDetectedLane[0]
            self.detectedRightLane = sortedDetectedLane[1]

        elif len(detectedLanes) > 0:
            # (detectedLeftLane = detectedLanes[0].copy()) if detectedLanes[0][0] < CFG.IMAGE_WIDTH/2 else (detectedRightLane = detectedLanes[0].copy())
            if detectedLanes[0][5][0] < CFG.IMAGE_WIDTH/2:
                self.detectedLeftLane = detectedLanes[0].copy()
            else:
                self.detectedRightLane = detectedLanes[0].copy()
            

        self.leftlane.updateDetectedLane(self.detectedLeftLane)
        self.rightlane.updateDetectedLane(self.detectedRightLane)

        if not self.leftlane.isInit:
            self.leftlane.init(self.detectedLeftLane)
        if not self.rightlane.isInit:
            self.rightlane.init(self.detectedRightLane)

        self.leftlane.updateLanev2()
        self.rightlane.updateLanev2()

