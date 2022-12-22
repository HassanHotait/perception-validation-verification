
import cv2
from metrics_functions_from_evaluation_script import read_groundtruth
from SMOKE.tools.my_functions import get_K
from SMOKE.tools.box import visualize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from SMOKE.tools.SMOKE_Visualizer import SMOKE_Viz

class GtInfo(object):
    def __init__(self, line):
        self.name = line[0]


        # self.truncation = float(line[1])
        # self.occlusion = int(line[2])

        # # local orientation = alpha + pi/2
        # self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

img_path='/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/image_2/000008.png'


groundtruth=read_groundtruth('/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/label_2',8)

K=get_K('000008.txt','/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/calib')
print('groundtruth: ',groundtruth)

print('groundtruth type: ',type(groundtruth))

print('groundtruth[0]: ',groundtruth[0])

print('K Matrix: \n',K)


gt_object=GtInfo(groundtruth[0])
print('GT: ',gt_object.xmax)

img=cv2.imread(img_path)


# cv2.imshow('output',img)
# cv2.waitKey(0)

# visualize((800,800),groundtruth,K[0],img)
# plt.show()

print('Object Rot_global [Deg]: ',np.degrees(float(groundtruth[0][-1])))
viz=SMOKE_Viz(lat_range_m=10,long_range_m=40,scale=10)
viz.draw_3Dbox(img,K[0],gt_list=groundtruth)
viz.draw_birdeyes(groundtruth)
viz.show()