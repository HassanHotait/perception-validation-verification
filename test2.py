import csv
import numpy as np
import cv2
import glob
import os


def find_indices(list,item_to_find):
    indices=[]

    for idx,value in enumerate(list):
        if value==item_to_find:
            indices.append(idx)
    
    return indices

def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]




with open('calibration_optimisation_data/dataset test/analytics_filtered_ids.csv','r') as f:
    csv_reader = csv.reader(f)
    yolo_rows = list(csv_reader)

with open('calibration_optimisation_data/dataset test/processed_cameraTrackedData.csv','r') as f:
    csv_reader = csv.reader(f)
    mobileye_rows = list(csv_reader)

yolo_timestamps=[float(row[0]) for row in yolo_rows[1:]]
mobileeye_timestamps=[float(row[0]) for row in mobileye_rows]



for index,timestamp in enumerate(yolo_timestamps):
    t_closest=closest(mobileeye_timestamps,timestamp)

    mobileye_list_index=mobileeye_timestamps.index(t_closest)

    if yolo_rows[index+1][2]==mobileye_rows[mobileye_list_index][1]:
        writer.writerow(yolo_rows[index+1])
        writer.writerow(mobileye_rows[mobileye_list_index])

optimization_dataset_file.close()

####################################################################################################

camera_height=1.5
with open('calibration_optimisation_data/dataset test/optimization_dataset_verify.csv','r') as f:
    csv_reader = csv.reader(f)
    data_rows = list(csv_reader)


final_dataset_file=open('calibration_optimisation_data/dataset test/final_dataset.csv', 'w',newline='') 
writer = csv.writer(final_dataset_file)

header = ['image_filename', '# objects', 'X1 3D', 'Y1 3D','Z1 3D','........','Xn 3D', 'Yn 3D','Zn 3D','X1 Obj Center 2D','Y1 Obj Center 2D','........','Xn Obj Center 2D','Yn Obj Center 2D']

writer.writerow(header)

for i in range(0,len(data_rows),2):
    yolo_row=data_rows[i]
    mobileye_row=data_rows[i+1]

    n_objects=int(yolo_row[2])
    image_filename=yolo_row[1]

    pose_3d=[]
    image_2d=[]

    if yolo_row[2]==1:
        pose_3d=[-float(mobileye_row[4]),camera_height,float(mobileye_row[3])]
        xc=(yolo_row[3]+yolo_row[5])/2
        yc=(yolo_row[4]+yolo_row[6])/2
        bbox_height=abs(yolo_row[4]-yolo_row[6])
        image_2d=[xc,yc+(bbox_height/2)]

        row=[image_filename,n_objects]+pose_3d+image_2d
        writer.writerow(row)

    else:
        for i in range(n_objects):
            pose_3d.extend((-float(mobileye_row[4+(3*i)]),camera_height,float(mobileye_row[3+(3*i)])))

            xc=(float(yolo_row[3+(4*i)])+float(yolo_row[5+(4*i)]))/2
            yc=(float(yolo_row[4+(4*i)])+float(yolo_row[6+(4*i)]))/2

            bbox_height=abs(float(yolo_row[4+(4*i)])-float(yolo_row[6+(4*i)]))


            image_2d.extend((xc,yc+(bbox_height/2)))


        # sort detections in correct order

        # pose_3d=[x1,y1,z1,x2,y2,z2]
        # image_2d=[x1,y1,x2,y2]
        lateral_distances=[pose_3d[i] for i in range(0,len(pose_3d),3)]
        lateral_pixel_points=[image_2d[0] for i in range(0,len(image_2d),2)]

        sorted_lateral_distances=lateral_distances.copy()
        sorted_lateral_pixel_points=lateral_pixel_points.copy()

        sorted_lateral_distances.sort()
        sorted_lateral_pixel_points.sort()

        sorted_pose_3d=[]
        sorted_image_2d=[]

        for lat_d,lateral_pixel in zip(sorted_lateral_distances,sorted_lateral_pixel_points):
            distance_index=pose_3d.index(lat_d)
            pixel_index=image_2d.index(lateral_pixel)

            sorted_pose_3d.extend((pose_3d[distance_index],pose_3d[distance_index+1],pose_3d[distance_index+2]))
            sorted_image_2d.extend((image_2d[pixel_index],image_2d[pixel_index+1]))
            
        

        row=[image_filename,n_objects]+sorted_pose_3d+sorted_image_2d

        writer.writerow(row)

        row=[image_filename,n_objects]+pose_3d+image_2d

        writer.writerow(row)

final_dataset_file.close()

# Video Writer

frame_id=1
test_dir='calibration_optimisation_data/Mobileye/2022-02-25-12-02-31_5/yolo-image-streamMobileeyeYoloTest/'
IMAGE='frame'+str(frame_id)+'.png'
image_path=os.path.join(test_dir,IMAGE)
print(image_path)
img=cv2.imread(image_path)
print('shape',img.shape)
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
codec = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter('calibration_optimisation_data/dataset test/objects_with_distances.mp4', codec, 20, (img.shape[1],img.shape[0]))

#test_dir='/home/hasan/SMOKE/datasets/KITTI_MOD_fixed/training/images/'

with open('calibration_optimisation_data/dataset test/final_dataset.csv','r') as f:
    csv_reader = csv.reader(f)
    rows = list(csv_reader)


frames_with_relevant_objects=[int(row[0]) for row in rows[1:]]


print(frames_with_relevant_objects)
print('---------------------------------------------',len(rows))
for filepath in glob.glob(os.path.join(test_dir,'*.png')):
    image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')
    print('IMAGE PATH: ',image_path)
    if frame_id in frames_with_relevant_objects:
        img = cv2.imread(image_path)
        img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        img=cv2.circle(img, (int(float(rows[frame_id+1][5])),int(float(rows[frame_id+1][6]))), 10, (255,0,0), 3)
        img=cv2.putText(img,str((rows[frame_id+1][2],rows[frame_id+1][3],rows[frame_id+1][4])),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        indices=find_indices(frames_with_relevant_objects,frame_id)
        # print('condition true')
        for i in indices:
            img=cv2.circle(img, (int(float(rows[i+1][5])),int(float(rows[i+1][6]))), 10, (255,0,0), 3)
            img=cv2.putText(img,str((rows[i+1][2],rows[i+1][3],rows[i+1][4])),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        
    else:
        img=cv2.imread(image_path)
        img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
    # cv2.imshow('output',img)
    # cv2.waitKey(0)
    video.write(img)
    frame_id+=1

cv2.destroyAllWindows()
video.release()