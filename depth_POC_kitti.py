from metrics_functions_from_evaluation_script import read_groundtruth,GtInfo
import numpy as np
import os
import csv
#from test_manuallly_obtained_matrix import get_calculated_2d_points
from EvaluatorClass3 import plot_groundtruth
import cv2

def get_K(file_name,calib_dir):

    with open(os.path.join(calib_dir, file_name), 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for line, row in enumerate(reader):
                    if row[0] == 'P2:':
                        K = row[1:]
                        K = [float(i) for i in K]
                        #print('K array: ',K)
                        #print('K elements length',len(K))
                        K = np.array(K, dtype=np.float32).reshape(3, 4)
                        P2=K
                        #print('K Shape',)
                        #print('K after reshaping (3,4)',K)
                        K = K[:3, :3]
                        #print('Input K: ',K)
                        break
                    # if 'P2' in line:
                    #     P2=line.split(' ')
                    #     P2 = np.asarray([float(i) for i in P2[1:]])
                    #     P2 = np.reshape(P2, (3, 4))

    return K,P2

def get_calculated_2d_points(K,pose):

    x_3d=pose[0]
    y_3d=pose[1]
    z_3d=pose[2]

    calculated_output=np.dot(K,np.array([[x_3d],[y_3d],[z_3d]]))
    calculated_x=int(calculated_output[0]/z_3d)
    calculated_y=int(calculated_output[1]/z_3d)

    return (calculated_x,calculated_y)
id=8
labels_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\label_2"
calib_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\calib"
groundtruth_list=read_groundtruth(gt_folder=labels_path,fileid=id,extension=".txt")

K,p2=get_K(calib_dir=calib_path,file_name=str(id).zfill(6)+".txt")
groundtruth_class=[GtInfo(gt) for gt in groundtruth_list]

img_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\image_2\\"+str(id).zfill(6)+".png"
img=cv2.imread(img_path)
print("K: ",K)
print("Groundtruth: ",groundtruth_class)
print("len(gt): ",len(groundtruth_list))

gt_img=plot_groundtruth(img,groundtruth=groundtruth_list)
camera_height=1.65

fy=K[1][1]
cy=K[1][2]

print("fy: ",fy)
print("cy: ",cy)

for obj in groundtruth_class:

    obj_center=get_calculated_2d_points(K,[obj.tx,obj.ty-(obj.h/2),obj.tz])
    gt_img=cv2.circle(gt_img,obj_center , 0, (0,255,0), 2)

    Z=((camera_height-(obj.h/2))*fy)/(obj_center[1]-cy)

    print("Calculated Depth: ",Z)
    print("Groundtruth Depth: ",obj.tz)

cv2.imshow("Gt Image",gt_img)
cv2.waitKey(0)



