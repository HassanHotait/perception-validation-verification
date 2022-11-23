
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
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
#from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,metrics_evaluator,read_groundtruth,yolo_2_smoke_output_format
from EvaluatorClass2 import metrics_evaluator,plot_groundtruth,get_gt_classes_boxes,get_gt_difficulty,get_gt_truncation_occlusion,get_pred_classes_boxes
from metrics_functions_from_evaluation_script import yolo_2_smoke_output_format,read_groundtruth,plot_prediction,get_IoU


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


stream_id='yolo_find_cyclists'
session_datetime=str(datetime.now())  

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')
results_path=os.path.join(root_dir,'results',foldername)
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
#smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
person_bike_intersection_image_stream_path=os.path.join(results_path,'person-bike-intersection-image-stream'+str(stream_id))

logs_path=os.path.join(results_path,'logs')



# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
# Create Folders in which to store YOLO and SMOKE Frames
os.mkdir(yolo_image_stream_path)
#os.mkdir(smoke_image_stream_path)
# Create foler in which to store text file logs
os.mkdir(logs_path)
os.mkdir(groundtruth_image_stream_path)
os.mkdir(person_bike_intersection_image_stream_path)




# # open the file in the write mode
# f = open(os.path.join(logs_path,'mobileeye_yolo_detections.csv'), 'w')
# # create the csv writer
# writer = csv.writer(f)
# header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
# writer.writerow(header)

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

n=1500
#yolo_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0
# vid = cv2.VideoCapture('test_videos/'+str(1)+'.mp4')
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# fps    = vid.get(cv2.CAP_PROP_FPS)
# ret,frame=vid.read()
count_gt_cyclists=0
count_pred_bike_or_person_as_cyclist=0
count_pred_bike_and_person=0
frames_with_potential_cyclist=[]
for i in range(n):
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)




    # yolo_log.write('\n')
    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    

    # Find Cyclists
    gt_classes,gt_boxs=get_gt_classes_boxes(groundtruth)
    gt_truncations,gt_occlusions=get_gt_truncation_occlusion(groundtruth)

    if 'Cyclist' not in gt_classes:
        dontcheck=True
    else:
        dontcheck=False

    if dontcheck==False:


        print('frame: ',fileid)
        boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
        print('Boxes Before: ',boxes)
        print('Classes Before: ',classes)
        boxs,classes,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
        print('Boxes after NMS: ',boxs)
        print('Classes after NMS: ',classes)
        yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes)
        print('Yolo Predictions List: ',yolo_predictions_list)




        for i,gt_class in enumerate(gt_classes):

            computed_difficulty=get_gt_difficulty(gt_box=gt_boxs[i],gt_truncation=gt_truncations[i],gt_occlusion=gt_occlusions[i])
            if computed_difficulty!='Ignored':

                output_img=plot_prediction(frame,yolo_predictions_list)


                # Count Gt Cyclist
                if gt_class=='Cyclist':
                    count_gt_cyclists+=1
                    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid)+'.png'),groundtruth_image)


                
                pred_classes,pred_boxes=get_pred_classes_boxes(yolo_predictions_list)
                

                # Count Cyclist Detected as Bike or Person
                for j,pred_class in enumerate(pred_classes):

                    iou=get_IoU(gt_boxs[i],pred_boxes[j])


                    if gt_class=='Cyclist'  and pred_class in ['Pedestrian','Bicycle'] and iou>0.3:
                        count_pred_bike_or_person_as_cyclist+=1
                        


                        
                        #yolo_vid_writer.write(img)
                        cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),output_img)


                    
                pred_car_boxes=[ pred_boxes[i] for i in range(len(pred_boxes)) if pred_classes[i]=='Bicycle']

                pred_person_boxes=[ pred_boxes[i] for i in range(len(pred_boxes)) if pred_classes[i]=='Pedestrian']


                for car_box in pred_car_boxes:
                    for person_box in pred_person_boxes:
                        person_bike_intersection=get_IoU(car_box,person_box)

                        if person_bike_intersection>0.4:
                            count_pred_bike_and_person+=1
                            cv2.imwrite(os.path.join(person_bike_intersection_image_stream_path,'frame'+str(fileid)+'.png'),output_img)
                            frames_with_potential_cyclist.append(i)


            
            









    #yolo_metrics_evaluator.evaluate_metrics(groundtruth,yolo_predictions_list)




    fileid+=1
    print('fileid: ',fileid)
    print('YOLO')
    # ret,frame=vid.read()

#print('Method Robustness; ',count_pred_potential_cyclist/count_gt_cyclists)


unique_frames_with_cyclists=list(set(frames_with_potential_cyclist))

print('Gt Frames with Cyclists: ',count_gt_cyclists)
print('Frames with potential cylists: ',count_pred_bike_or_person_as_cyclist)
print('Predictions with Intersecting Person and Bike (Representing Cyclist): ',count_pred_bike_and_person)


print('Number of Frames with Potential Cyclists: ',frames_with_potential_cyclist)








