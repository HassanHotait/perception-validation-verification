import csv 
import os
import numpy as np


n_frames_train=7481
objects_depth=[]
for i in range(n_frames_train):
    
    filename=str(i).zfill(6)+".txt"
    label_path="/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/label_2"
    with open(os.path.join(label_path,filename),'r') as f:
        reader=csv.reader(f)
        data=list(reader)

    
    for row in data:
        row=row[0].split(' ')
        
        if row[0] in ["Person","Car","Cyclist"]:
            depth=row[13]
            objects_depth.append(float(depth))


# n_frames_test=7518
# for i in range(n_frames_test):
    
#     filename=str(i).zfill(6)+".txt"
#     label_path="/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/traini/label_2"
#     with open(os.path.join(label_path,filename),'r') as f:
#         reader=csv.reader(f)
#         data=list(reader)

    
#     for row in data:
#         row=row[0].split(' ')
        
#         if row[0]!="DontCare":
#             depth=row[13]
#             objects_depth.append(float(depth))

    

print('Depth Mean',np.mean(objects_depth))
print("Depth Std: ",np.std(objects_depth))


prescan_objects_depth=[9.45,8.62]
print('Prescan Depth Mean',np.mean(prescan_objects_depth))
print("Prescan Depth Std: ",np.std(prescan_objects_depth))


