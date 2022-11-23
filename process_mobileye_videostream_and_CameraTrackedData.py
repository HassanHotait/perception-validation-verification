
from distutils import log
from email.mime import image
from logging import root
from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
from SMOKE.tools.box import visualize
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import csv
import numpy as np
torch.cuda.empty_cache()
import os
import glob
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,metrics_evaluator,read_groundtruth,yolo_2_smoke_output_format

from YOLO_toolbox import yolo_visualize

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

def get_key(val):
    TYPE_ID_CONVERSION = {
    'Vehicle': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }
    for key, value in TYPE_ID_CONVERSION.items():
        if val == value:
            return key
 
    return "DontCare"

######################################################################################################################
# Define Paths for Storing Results
stream_id='MobileeyeYoloTest2'
session_datetime=str(datetime.now())  
foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')
results_path=os.path.join(root_dir,'results',foldername)
yolo_image_stream_path_tracked=os.path.join(results_path,'yolo-image-stream-tracked'+str(stream_id))
yolo_image_stream_path_filtered=os.path.join(results_path,'yolo-image-stream-filtered'+str(stream_id))
#smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))

logs_path=os.path.join(results_path,'logs')



# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
# Create Folders in which to store YOLO and SMOKE Frames
os.mkdir(yolo_image_stream_path_tracked)
os.mkdir(yolo_image_stream_path_filtered)
#os.mkdir(smoke_image_stream_path)
# Create foler in which to store text file logs
os.mkdir(logs_path)
os.mkdir(groundtruth_image_stream_path)



#################################################################################################################

# open the file in the write mode
filename='mobileeye_yolo_detections.csv'
f = open(os.path.join(logs_path,filename), 'w')
# create the csv writer
writer = csv.writer(f)
header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
writer.writerow(header)

fileid=0





from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
from YOLO_toolbox  import setup_YOLO,setup_tracker,preprocess_predict_YOLO,postprocess_YOLO,yolo_visualize

# Setup YOLOV3 | Load Weights | Initialize Tracker for Object IDS | Initialize Feature Encoder
yolo,class_names=setup_YOLO(path_to_class_names='YOLO_Detection/data/labels/coco.names',
                            path_to_weights='YOLO_Detection/weights/yolov3.tf')
model_filename = 'YOLO_Detection/model_data/mars-small128.pb'
tracker,encoder=setup_tracker(model_filename)
    

# Hide GPU 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

fileid=0

n=3000
yolo_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0
vid = cv2.VideoCapture('test_videos/'+str(2)+'.mp4')
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = vid.get(cv2.CAP_PROP_FPS)
ret,frame=vid.read()
while ret==True:
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    #frame=cv2.imread(ordered_filepath)
    print('frame: ',fileid)
    boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
    boxs,classes,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
    yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes)
    print('Yolo Predictions List: ',yolo_predictions_list)


    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        elif track.get_class()=='car' or track.get_class()=='bus' or track.get_class()=='truck':
            row=[fileid,(fileid/fps),track.get_class(),track.track_id,track.to_tlbr()[0],track.to_tlbr()[1],track.to_tlbr()[2],track.to_tlbr()[3]]
            writer.writerow(row)
        else:
            pass

    print('tracker ',tracker.tracks[0].to_tlbr())
    output_img=plot_prediction(frame,yolo_predictions_list)
    
    yolo_output_img=yolo_visualize(tracker,frame,fileid)
    #yolo_vid_writer.write(img)
    cv2.imwrite(os.path.join(yolo_image_stream_path_tracked,'frame'+str(fileid)+'.png'),yolo_output_img)
    #yolo_metrics_evaluator.evaluate_metrics(groundtruth,yolo_predictions_list)
    fileid+=1
    print('fileid: ',fileid)
    print('YOLO')
    ret,frame=vid.read()

    # if fileid==150:
    #     ret=False
f.close()

##############################################################################################################
#vid 1 relevant ids
#relevant_ids=[5,22,27,33,47,78,97,105,144,159,163]
#vid 2 relevant ids
relevant_ids=[1,4,7,15,18,31,39,43,53,54]
header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']

filtered_ids_filename='mobileeye_yolo_detections_filtered_id.csv'
outputfile=open(os.path.join(logs_path,filtered_ids_filename), 'w')

writer=csv.writer(outputfile)
writer.writerow(header)

with open(os.path.join(logs_path,filename), 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')

    for i,row in enumerate(csv_reader):
        if i!=0:
            if int(row[3]) in relevant_ids:
                writer.writerow(row)


outputfile.close()


###############################################################################################################

yolo_filtered_ids_analytics_filename='analytics_filtered_ids.csv'
filtered_id_analytics_file=open(os.path.join(logs_path,yolo_filtered_ids_analytics_filename),'w')
writer=csv.writer(filtered_id_analytics_file)
writer.writerow(['timestamp','frame','# objects','X1_min','Y1_min','X1_max','Y1_max','......','Xn_min','Yn_min','Xn_max','Yn_max'])

with open(os.path.join(logs_path,filtered_ids_filename),'r') as f:
    csv_reader = csv.reader(f)
    rows = list(csv_reader)

frames_with_relevant_objects=[int(row[0]) for row in rows[1:]]

unique_frames=list(set(frames_with_relevant_objects))

print(unique_frames)

for frame_id in unique_frames:

    print(frame_id)
    indices_of_frame=find_indices(frames_with_relevant_objects,frame_id)
    n_objects=len(indices_of_frame)


    datarow=[]

    for i in indices_of_frame:
        datarow.extend((rows[i+1][4],rows[i+1][5],rows[i+1][6],rows[i+1][7]))



    writer.writerow([rows[indices_of_frame[0]+1][1],rows[indices_of_frame[0]+1][0],n_objects]+datarow)


filtered_id_analytics_file.close()


############################################


processed_cameraTrackedData_filename='processed_cameraTrackedData.csv'
processed_cameraTrackedData_file=open(os.path.join(logs_path,processed_cameraTrackedData_filename), 'w',newline='') 
writer = csv.writer(processed_cameraTrackedData_file)

#thelist=['0', '0', '1000', '0', '0', '1000', '0', '0', '1000', '0', '0', '1000', '0\n']
video1_data_path='calibration_optimisation_data/Mobileye/2022-02-25-12-02-31_5/cameraTrackedData.csv'
video2_data_path='calibration_optimisation_data/Mobileye/2022-02-25-12-10-31_21/cameraTrackedData.csv'
with open(video2_data_path) as f_in:
    for line in f_in:
        elements=line.split(',')
        elements[-1] = elements[-1].strip()
        print(elements)
        indices=find_indices(elements,'1000')
        indices_total=[[i-1,i,i+1] for i in indices]
        indices_total=sum(indices_total,[])
        elements=np.delete(elements, indices_total).tolist()

        if len(elements)==1:
            objects=0

        elif len(elements)==4:
            objects=1

        elif len(elements)==7:
            objects=2

        elif len(elements)==10:
            objects=3   
        
        elements.insert(1,objects)
        print(elements)
        writer.writerow(elements)
        
            
processed_cameraTrackedData_file.close()


print(indices_total)

#############################################
# find closest timestamps

optimization_dataset_filename='optimization_dataset.csv'
optimization_dataset_file=open(os.path.join(logs_path,optimization_dataset_filename), 'w',newline='') 
writer = csv.writer(optimization_dataset_file)


with open(os.path.join(logs_path,yolo_filtered_ids_analytics_filename),'r') as f:
    csv_reader = csv.reader(f)
    yolo_rows = list(csv_reader)

with open(os.path.join(logs_path,processed_cameraTrackedData_filename),'r') as f:
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



############################################
# Video Writer

frame_id=0
test_dir=yolo_image_stream_path_tracked
IMAGE='frame'+str(frame_id)+'.png'
image_path=os.path.join(test_dir,IMAGE)
print(image_path)
img=cv2.imread(image_path)
print('shape',img.shape)
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
codec = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter(os.path.join(logs_path,'filtered_objects.mp4'), codec, 20, (img.shape[1],img.shape[0]))

#test_dir='/home/hasan/SMOKE/datasets/KITTI_MOD_fixed/training/images/'

with open(os.path.join(logs_path,filtered_ids_filename),'r') as f:
    csv_reader = csv.reader(f)
    rows = list(csv_reader)


frames_with_relevant_objects=[int(row[0]) for row in rows[1:]]


print(frames_with_relevant_objects)

for filepath in glob.glob(os.path.join(test_dir,'*.png')):
    image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')

    print('IMAGE PATH: ',image_path)
    img = cv2.imread(image_path)
    if frame_id in frames_with_relevant_objects:
        
        #img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        indices=find_indices(frames_with_relevant_objects,frame_id)
        print('condition true')
        for i in indices:
            img = cv2.rectangle(img, (int(float(rows[i+1][4])),int(float(rows[i+1][5]))),(int(float(rows[i+1][6])),int(float(rows[i+1][7]))) , (255,255,0), 10)
            
        
    else:
        #img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        pass
    # cv2.imshow('output',img)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(yolo_image_stream_path_filtered,'frame'+str(frame_id)+'.png'),img)
    video.write(img)
    frame_id+=1

cv2.destroyAllWindows()
video.release()

###########################################################################################################################################

# Match Data

camera_height=1.8
lateral_offset=0
longitudinal_offset=0.75
height_offset=-0.3
with open(os.path.join(logs_path,optimization_dataset_filename),'r') as f:
    csv_reader = csv.reader(f)
    data_rows = list(csv_reader)

final_dataset_filename='final_dataset.csv'
final_dataset_file=open(os.path.join(logs_path,final_dataset_filename), 'w',newline='') 
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
            pose_3d.extend((-float(mobileye_row[4+(3*i)]),camera_height+height_offset,float(mobileye_row[3+(3*i)])+longitudinal_offset))

            xc=(float(yolo_row[3+(4*i)])+float(yolo_row[5+(4*i)]))/2
            yc=(float(yolo_row[4+(4*i)])+float(yolo_row[6+(4*i)]))/2

            bbox_height=abs(float(yolo_row[4+(4*i)])-float(yolo_row[6+(4*i)]))


            image_2d.extend((xc,yc+(bbox_height/2)))

        # sort detections in correct order

        # pose_3d=[x1,y1,z1,x2,y2,z2]
        # image_2d=[x1,y1,x2,y2]
        lateral_distances=[pose_3d[i] for i in range(0,len(pose_3d),3)]
        lateral_pixel_points=[image_2d[i] for i in range(0,len(image_2d),2)]

        print('Lateral Distances: ',lateral_distances)
        print('Lateral Pixel Points: ',lateral_pixel_points)

        sorted_lateral_distances=lateral_distances.copy()
        sorted_lateral_pixel_points=lateral_pixel_points.copy()

        sorted_lateral_distances.sort(reverse=True)
        print('Sorted Lateral Distances: ',lateral_distances)
        sorted_lateral_pixel_points.sort()
        print('Sorted Lateral Pixel Points: ',sorted_lateral_pixel_points)

        sorted_pose_3d=[]
        sorted_image_2d=[]

        for lat_d,lateral_pixel in zip(sorted_lateral_distances,sorted_lateral_pixel_points):
            distance_index=pose_3d.index(lat_d)
            pixel_index=image_2d.index(lateral_pixel)

            sorted_pose_3d.extend((pose_3d[distance_index],pose_3d[distance_index+1],pose_3d[distance_index+2]))
            sorted_image_2d.extend((image_2d[pixel_index],image_2d[pixel_index+1]))

        row=[image_filename,n_objects]+sorted_pose_3d+sorted_image_2d

        writer.writerow(row)

final_dataset_file.close()

# Video Writer

frame_id=1
test_dir=yolo_image_stream_path_tracked
IMAGE='frame'+str(frame_id)+'.png'
image_path=os.path.join(test_dir,IMAGE)
print(image_path)
img=cv2.imread(image_path)
print('shape',img.shape)
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
codec = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter(os.path.join(logs_path,'objects_with_distances.mp4'), codec, 20, (img.shape[1],img.shape[0]))

#test_dir='/home/hasan/SMOKE/datasets/KITTI_MOD_fixed/training/images/'

with open(os.path.join(logs_path,final_dataset_filename),'r') as f:
    csv_reader = csv.reader(f)
    rows_final_dataset = list(csv_reader)

with open(os.path.join(logs_path,yolo_filtered_ids_analytics_filename),'r') as f:
    csv_reader = csv.reader(f)
    rows = list(csv_reader)


frames_with_relevant_objects=[int(row[1]) for row in rows[1:]]


frames_with_relevant_objects_final_dataset=[int(row[0]) for row in rows_final_dataset[1:]]


print(frames_with_relevant_objects)
print('---------------------------------------------',len(rows))
for filepath in glob.glob(os.path.join(test_dir,'*.png')):
    image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')
    print('IMAGE PATH: ',image_path)
    img = cv2.imread(image_path)



    if frame_id in frames_with_relevant_objects:
        #img = cv2.imread(image_path)
        #img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
        index=frames_with_relevant_objects.index(frame_id)
        print('condition true')
        n_objects=int(rows[index+1][2])
        for i in range(n_objects):
            img = cv2.rectangle(img, (int(float(rows[index+1][3+(4*i)])),int(float(rows[index+1][4+(4*i)]))),(int(float(rows[index+1][5+(4*i)])),int(float(rows[index+1][6+(4*i)]))) , (255,255,0), 10)


    if frame_id in frames_with_relevant_objects_final_dataset:
        #img = cv2.imread(image_path)

        img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)

        # img=cv2.circle(img, (int(float(rows[frame_id+1][5])),int(float(rows[frame_id+1][6]))), 10, (255,0,0), 3)

        # img=cv2.putText(img,str((rows[frame_id+1][2],rows[frame_id+1][3],rows[frame_id+1][4])),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)


        #indices=find_indices(frames_with_relevant_objects,frame_id)
        index=frames_with_relevant_objects_final_dataset.index(frame_id)
        index_rows=frames_with_relevant_objects.index(frame_id)
        # print('condition true')

        #for i in indices:

        n_objects=int(rows_final_dataset[index+1][1])
        print(n_objects)

        if n_objects==1:
            x_2d=int(float(rows_final_dataset[index+1][5]))
            y_2d=int(float(rows_final_dataset[index+1][6]))

            x_3D=rows_final_dataset[index+1][2]
            y_3d=rows_final_dataset[index+1][3]
            z_3d=rows_final_dataset[index+1][4]

            img=cv2.circle(img,(x_2d,y_2d), 10, (255,0,0), 3)

            xc=int((float(rows[index_rows+1][3])+float(rows[index_rows+1][5]))/2)
            yc=int((float(rows[index_rows+1][4])+float(rows[index_rows+1][6]))/2)


            img=cv2.putText(img,str((x_3D,y_3d,z_3d)),(xc-10,yc),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),2)

        else:
            print('n objects: ',n_objects)

            for i in range(n_objects):

                x_2d=int(float(rows_final_dataset[index+1][8+(2*i)]))
                y_2d=int(float(rows_final_dataset[index+1][9+(2*i)]))

                x_3D=rows_final_dataset[index+1][2+(3*i)]
                y_3d=rows_final_dataset[index+1][3+(3*i)]
                z_3d=rows_final_dataset[index+1][4+(3*i)]

                img=cv2.circle(img,(x_2d,y_2d), 10, (255,0,0), 3)

                xc=int((float(rows[index_rows+1][3+(4*i)])+float(rows[index_rows+1][5+(4*i)]))/2)
                yc=int((float(rows[index_rows+1][4+(4*i)])+float(rows[index_rows+1][6+(4*i)]))/2)



                img=cv2.putText(img,str((x_3D,y_3d,z_3d)),(xc-10,yc),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),2)
        
    else:
        img=cv2.imread(image_path)
        img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
    # cv2.imshow('output',img)
    # cv2.waitKey(0)
    video.write(img)
    frame_id+=1

cv2.destroyAllWindows()
video.release()




    





