
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
import cv2
torch.cuda.empty_cache()
import os
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
#from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,metrics_evaluator,read_groundtruth,yolo_2_smoke_output_format
from EvaluatorClass2 import metrics_evaluator,plot_groundtruth
from metrics_functions_from_evaluation_script import yolo_2_smoke_output_format,read_groundtruth,plot_prediction,write_prediction,read_prediction,get_pred_classes_boxes
import subprocess
import keyboard

class ManualMatchingMouseHandler():
    def __init__(self,clean_image,distances,boxes):
        print('Class Initialized')
        self.selected_LB_points=[]
        self.frame_with_stats=clean_image
        self.clean_img=clean_image.copy()
        self.original_distances=distances.copy()
        self.original_boxes=boxes.copy()

        self.matched_points_boxes=[]
        self.matched_distances=[]

        self.current_row_pose_list=self.original_distances

    def self_match_detections(self):

        
        for j,point in enumerate(self.selected_LB_points):
            print('Selected Points: ',len(self.selected_LB_points))
            for i,box in enumerate(self.original_boxes):

                if box[1][0]-box[0][0]>700:
                    continue

                check=point_in_box(point,box)

                if check==True:
                    print()
                    self.matched_points_boxes.append((self.matched_distances[j],box))
                    self.original_boxes.pop(i)


    def mouse_click(self,event, x, y, 
                    flags, param):

        
        
        # to check if left mouse 
        # button was clicked

        # if key == ord("r"):
        #     print('Pressed r')
        
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Selected Point')
            
            # current_row_pose_list=self.original_distances
            self.frame_with_stats=self.clean_img.copy()
            self.selected_LB_points.append((x,y))
            self.matched_distances.append(self.current_row_pose_list[0])
            self.self_match_detections()
            
            # font for left click event
            font = cv2.FONT_HERSHEY_TRIPLEX
            #LB = str(current_row_pose_list[0])
            
            # display that left button 
            # was clicked.
            print('Matched Detections: ',self.matched_points_boxes)
            for (pose,box) in self.matched_points_boxes:
                print('Box: ',box)
                print('Pose: ',pose)
                box_midpoint=(int((box[1][0]+box[0][0])/2),int((box[1][1]+box[0][1])/2))
                cv2.putText(self.frame_with_stats, str(pose), box_midpoint, 
                            font, 1, 
                            (255, 255, 0), 
                            2) 

            if len(self.current_row_pose_list)!=0:
                self.current_row_pose_list.pop(0)
            cv2.putText(self.frame_with_stats,str(self.current_row_pose_list),(250,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,00,00),4)
            
            cv2.imshow('image', self.frame_with_stats)
            
            
            
        # to check if right mouse 
        # button was clicked
        if event == cv2.EVENT_RBUTTONDOWN:
            print('Ignored Point')
            self.frame_with_stats=self.clean_img.copy()
            # font for right click event
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            if len(self.current_row_pose_list)!=0:
                self.current_row_pose_list.pop(0)
            for (pose,box) in self.matched_points_boxes:
                print('Box: ',box)
                print('Pose: ',pose)
                box_midpoint=(int((box[1][0]+box[0][0])/2),int((box[1][1]+box[0][1])/2))
                cv2.putText(self.frame_with_stats, str(pose), box_midpoint, 
                            font, 1, 
                            (255, 255, 0), 
                            2) 
            cv2.putText(self.frame_with_stats,str(self.current_row_pose_list),(250,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,00,00),4)
            cv2.imshow('image', self.frame_with_stats)

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

def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]

def point_in_box(point,box):

    TLx=box[0][0]
    TLy=box[0][1]
    BRx=box[1][0]
    BRy=box[1][1]

    ptx=point[0]
    pty=point[1]

    if TLx<ptx<BRx and TLy<pty<BRy:
        return True
    else:
        return False






predict_then_evaluate=True
only_predict=False
only_evaluate=False

folder_path_for_evaluation='Streamkitti_training_set_smoke_metricstoday'


stream_id='ManualMatching_YOLO'
session_datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
# boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
# test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')



# if predict_then_evaluate==True or only_predict==True:
results_path=os.path.join(root_dir,'results',foldername)

# elif only_evaluate==True:
#     results_path=os.path.join(root_dir,'results',folder_path_for_evaluation)
# else:
#     pass


data_path=os.path.join(results_path,'data')
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
#smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
#groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))

logs_path=os.path.join(results_path,'logs')




# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
os.mkdir(data_path)
# Create Folders in which to store YOLO and SMOKE Frames
os.mkdir(yolo_image_stream_path)
#os.mkdir(smoke_image_stream_path)
# Create foler in which to store text file logs
os.mkdir(logs_path)
#os.mkdir(groundtruth_image_stream_path)

optimization_data_filepath=open(os.path.join(logs_path,'optimization_data.csv'),'w')
optimization_data_file=csv.writer(optimization_data_filepath)
header = ['timetsamp','image_filename', 'X1 3D', 'Y1 3D','Z1 3D','X1 Obj Center 2D','Y1 Obj Center 2D']
optimization_data_file.writerow(header)

video_filename='1.mp4'
dictionary={
                            "1.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-02-31_5/cameraTrackedData.csv",
                            "2.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-10-31_21/cameraTrackedData.csv",
                            "3.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-13-01_26/cameraTrackedData.csv",
                            "4.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-16-01_32/cameraTrackedData.csv",
                            "5.mp4": "calibration_optimisation_data/Mobileye/2022-06-08-13-22-38_5/cameraTrackedData.csv",
                            "6.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-19-01_38/cameraTrackedData.csv",
                            "7.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-19-31_39/cameraTrackedData.csv",
                            }

with open(os.path.join(dictionary[video_filename]),'r') as f_in:
    reader=csv.reader(f_in)
    mobileye_datarows=list(reader)

mobileye_float_list=[[] for i in range(len(mobileye_datarows))]
timestamps=[float(row[0]) for row in mobileye_datarows]
distances=[]
for i,row in enumerate(mobileye_datarows):

    pose=[(-float(row[3]),float(row[2])),(-float(row[6]),float(row[5])),(-float(row[9]),float(row[8])),(-float(row[12]),float(row[11]))]
    distances.append(pose)
    for item in row:
        mobileye_float_list[i].append(float(item))

    

print(mobileye_float_list)

print('length of rows: ',len(mobileye_datarows))
print('length of items in float list: ',len(mobileye_float_list))


# # open the file in the write mode
# f = open(os.path.join(logs_path,'mobileeye_yolo_detections.csv'), 'w')
# # create the csv writer
# writer = csv.writer(f)
# header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
# writer.writerow(header)

# fileid=0





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



n=4
# yolo_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0

vid_in = cv2.VideoCapture(os.path.join('test_videos',video_filename))
vid_fps =int(vid_in.get(cv2.CAP_PROP_FPS))
# vid = cv2.VideoCapture('test_videos/'+str(1)+'.mp4')
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# fps    = vid.get(cv2.CAP_PROP_FPS)
# ret,frame=vid.read()
ret,frame=vid_in.read()

#mouse_handler=ManualMatchingMouseHandler()
condition=True
retry=False
processed_boxes_of_interest=[]
optimization_data_sample_points=0
while condition==True:
    print('frame: ',fileid)
    timestamp=(fileid/vid_fps)

    closest_mobileye_timestamp=closest(timestamps,timestamp)

    current_row_pose_list=distances[timestamps.index(closest_mobileye_timestamp)]

    print('frame distances: ',current_row_pose_list)

    check_detected_objects=[(0.0,1000.0)==pose for pose in current_row_pose_list]

    print('Check Detected Objects: ',check_detected_objects)

    if False in check_detected_objects:
        mobileye_detected_objects=True

    else:
        mobileye_detected_objects=False

    # ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    # frame=cv2.imread(ordered_filepath)
    

    if mobileye_detected_objects==True:
        boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)

        print('Boxes before postproceesing: ',boxes)

        boxs,classes,scores,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)

        print('Boxes after postproceesing: ',boxs)

        yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes,scores=scores)

        pred_classes,pred_boxes=get_pred_classes_boxes(yolo_predictions_list)

        boxes_of_interest=[box for j,box in enumerate(boxs) if pred_classes[j]=='Car']

        image=frame
        
        if retry==False:
            for box in boxes_of_interest:
                print('box---------------------',box)
                TL= (int(box[0]),int(box[1]))
                BR=(int(box[0]+box[2]),int(box[1]+box[3]))
                image = cv2.circle(image,TL, 2, (0,255,0), 15)
                image = cv2.circle(image,BR , 2, (255,0,0), 15)
                processed_boxes_of_interest.append((TL,BR))
            print('yolo predictions list: ',yolo_predictions_list)
            write_prediction(data_path,fileid,yolo_predictions_list)

            

            output_img=plot_prediction(image,yolo_predictions_list)

            output_img=cv2.putText(output_img,'Frame: '+str(fileid),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
            output_img=cv2.putText(output_img,'Frame Timestamp: '+ '{:.4f}'.format(timestamp),(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
            output_img=cv2.putText(output_img,'Mobileye Timestamp: '+ '{:.4f}'.format(closest_mobileye_timestamp),(0,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
            output_img=cv2.putText(output_img,'Mobileye datarow : '+  str(timestamps.index(closest_mobileye_timestamp))+'/'+str(len(mobileye_datarows)),(0,110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
            output_img=cv2.putText(output_img,'Optimization Data Samples: '+  str(optimization_data_sample_points),(0,140),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
            output_img_distances=cv2.putText(output_img.copy(),str(current_row_pose_list),(250,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,00,00),4)
        

        
        print('Current Pose List: ',current_row_pose_list)
        
        print('Processed Boxes Of Interest: ',processed_boxes_of_interest)
        
        mouse_handler=ManualMatchingMouseHandler(output_img,current_row_pose_list,processed_boxes_of_interest)
        #mouse_handler.frame_with_stats=output_img


        

        cv2.imshow('image',output_img_distances)
        cv2.setMouseCallback('image',mouse_handler.mouse_click)
        #cv2.waitKey(0)

        print('Matched Detections: ',mouse_handler.matched_points_boxes)
        

        key = cv2.waitKey(0) & 0xFF

                # if the 'q' key is pressed, exit from loop
        if key == ord("q"):
            condition=False
            

        #if the 'n' key is pressed, go to next frame
        if key == ord("n"):
            retry=False
            ret,frame=vid_in.read()
            
            processed_boxes_of_interest=[]


            if ret==False:
                condition=False
            else:
                condition=True

            print('Matched Detections To Be Written: ',mouse_handler.matched_points_boxes)
            camera_height=1.25
            lat_offset=0
            long_offset=0.9
            for (pose,box) in mouse_handler.matched_points_boxes:
                X_3D=pose[0]
                Y_3D=camera_height
                Z_3D=pose[1]+long_offset

                X_2D=(box[0][0]+box[1][0])/2
                Y_2D=box[1][1]

                optimization_data_row=[timestamp,fileid,X_3D,Y_3D,Z_3D,X_2D,Y_2D]
                optimization_data_file.writerow(optimization_data_row)
                optimization_data_sample_points+=1

            if len(mouse_handler.matched_points_boxes)>0:
                cv2.imwrite(os.path.join(yolo_image_stream_path,str(fileid).zfill(6)+'.png'),mouse_handler.frame_with_stats)


            fileid+=1


            continue

        if key == ord("r"):
            condition=True
            retry=True


            

    else:
        fileid+=1
        ret,frame=vid_in.read()

    # 




    






torch.cuda.empty_cache()




