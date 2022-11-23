import csv
import numpy as np
import os
import cv2
def find_indices(list,item_to_find):
    indices=[]

    for idx,value in enumerate(list):
        if value==item_to_find:
            indices.append(idx)
    
    return indices


def get_calculated_2d_points(K,pose):

    x_3d=pose[0]
    y_3d=pose[1]
    z_3d=pose[2]

    calculated_output=np.dot(K,np.array([[x_3d],[y_3d],[z_3d]]))
    calculated_x=int(calculated_output[0]/z_3d)
    calculated_y=int(calculated_output[1]/z_3d)

    return (calculated_x,calculated_y)

optimization_data_folder='StreamManualMatching_YOLO2022_11_20_17_48_04'
with open(os.path.join('/home/hasan/perception-validation-verification/results',optimization_data_folder,'logs','optimization_data.csv'),'r') as f:
    reader=csv.reader(f)
    optimization_data=list(reader) 

test_images_path=os.path.join('/home/hasan/perception-validation-verification/results',optimization_data_folder,'yolo-image-streamManualMatching_YOLO')

frames=[int(float(row[1])) for row in optimization_data[1:]]
pose=[(float(row[2]),float(row[3]),float(row[4])) for row in optimization_data[1:]]
image_pts_2d=[(int(float(row[5])),int(float(row[6]))) for row in optimization_data[1:]]

K=np.array([[1721.52,0,951],[0,469.5,469.63],[0,0,1]])

# Calculating focal length using an optimization algorithm
# Focal distance in pixels fx: 1038.38
# Focal distance in pixels fy: 277.02
# Image cx distance in pixels cx: 940.62
# Image cy distance in pixels cy: 439.34
# Calculating focal length using an optimization algorithm
# Focal distance in pixels fx: 1295.16
# Focal distance in pixels fy: 564.44
# Image cx distance in pixels cx: 1029.02
# Image cy distance in pixels cy: 436.56
# Video 1
# Focal distance in pixels fx: 1583.43
# Focal distance in pixels fy: 473.33
# Image cx distance in pixels cx: 958.08
# Image cy distance in pixels cy: 470.93
# Video 1 Filtered Bad Points from Samples
# Calculating focal length using an optimization algorithm
# Focal distance in pixels fx: 1651.52
# Focal distance in pixels fy: 426.40
# Image cx distance in pixels cx: 957.31
# Image cy distance in pixels cy: 473.68
fx=1651
fy=426
cx=957
cy=473
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

print('2D Image Points: ',image_pts_2d)
print('Pose: ',pose)

for frame in frames:

    print('FRAME: ',frame)

    img_path=os.path.join(test_images_path,str(frame).zfill(6)+'.png')
    img=cv2.imread(img_path)

    object_indices=find_indices(frames,frame)

    for i in object_indices:
        print('index: ',i)
        img = cv2.circle(img,image_pts_2d[i] , 0, (0,255,0), 3)
        img=cv2.circle(img,get_calculated_2d_points(K,pose[i]) , 0, (0,0,255), 10)


    cv2.imshow('Check Matrix',img)
    cv2.waitKey(0)





