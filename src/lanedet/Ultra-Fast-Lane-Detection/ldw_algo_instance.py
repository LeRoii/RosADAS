from lanedet import LaneDetect
from lane_tracker import LaneTracker
from lane_warning import LaneWarning

# import threading
# import playsound

# def playWarningSound():
#     playsound.playsound('/space/warn.mp3')

class laneDepartureWarning:
    def __init__(self):
        self.detection = LaneDetect()
        self.tracker = LaneTracker()
        self.warning = LaneWarning()

    def process(self, img):
        detectedLanes = self.detection.process(img)
        self.tracker.process(detectedLanes)
        llane = self.tracker.leftlane
        rlane = self.tracker.rightlane

        lanePoints = {
                'lanes':[llane.points, rlane.points]
        }

        warningret = self.warning.process(lanePoints)

        # color = (0, 0, 255) if warningret == 1 else (0, 255, 0)
        # if warningret == 1:
        #     soundplayTh = threading.Thread(target=playWarningSound)
        #     soundplayTh.start()

        # for idx in range(11):
        #     cv2.line(cv_image, (int(llane.points[idx][0]), int(llane.points[idx][1])), (int(llane.points[idx+1][0]), int(llane.points[idx+1][1])), color, 10)
        #     cv2.line(cv_image, (int(rlane.points[idx][0]), int(rlane.points[idx][1])), (int(rlane.points[idx+1][0]), int(rlane.points[idx+1][1])), color, 10)
        

        ret = {
            'leftlane':llane,
            'rightlane':rlane,
            'warningret':warningret
        }

        return ret