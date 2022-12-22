from SMOKE.tools.SMOKE_Visualizer import SMOKE_Viz
import cv2

from metrics_functions_from_evaluation_script import read_groundtruth
from SMOKE.tools.my_functions import get_K

img_path='/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/image_2/000000.png'
img=cv2.imread(img_path)
groundtruth=read_groundtruth('/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/label_2',0)


K=get_K('000000.txt','/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/calib')
print('Groundtruth: \n',groundtruth)

viz=SMOKE_Viz((900,900))
viz.draw_3Dbox(img,K[0],gt_list=groundtruth)
viz.draw_birdeyes(groundtruth)
viz.show()