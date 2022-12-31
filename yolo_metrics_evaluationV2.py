# Hide GPU 
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from logging import root
# from SMOKE.smoke.config import cfg
# from SMOKE.smoke.engine import default_argument_parser
# from SMOKE.tools.my_functions import setup_network,preprocess
# from SMOKE.smoke.utils.check_point import DetectronCheckpointer
# from SMOKE.tools.box import visualize
import matplotlib.pyplot as plt
import cv2
from PIL import Image
#import torch
import csv
import numpy as np
#torch.cuda.empty_cache()
import os
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
#from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,metrics_evaluator,read_groundtruth,yolo_2_smoke_output_format
from EvaluatorClass3 import metrics_evaluator,plot_groundtruth
from metrics_functions_from_evaluation_script import yolo_2_smoke_output_format,yolo_2_json_output_format,read_groundtruth,plot_prediction,write_prediction,write_json,read_prediction,construct_dataframe_v2,get_class_AP
import subprocess
import dataframe_image as dfi 
import argparse

parser = argparse.ArgumentParser(description="Program Arg Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-root_dir", "--root_directory", default="C:/Users/hashot51/Desktop/perception-validation-verification", help="repository root directory")
args = parser.parse_args()


def get_coco_val_set_imgs(path):
    with open(path,"r") as f:
        reader=csv.reader(f)
        data=list(reader)


    imgs_list=[]
    for row in data:
        img_filename=row[0].split('/')[-1].split(' ')[0]
        imgs_list.append(img_filename)

    return imgs_list

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



predict_then_evaluate=False
only_predict=True
only_evaluate=False

folder_path_for_evaluation='StreamOfficial_YOLOV3_eval_threshold_0.5_2022_12_22_17_53_48'


stream_id="YOLO_benchmark_COCO_@0.05_"
session_datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir=args.root_directory#'/home/hashot51/Projects/perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')
coco_dataset_path="C:\\Users\\hashot51\\Desktop\\TensorFlow-2.x-YOLOv3\\model_data\\coco\\val2017"



if predict_then_evaluate==True or only_predict==True:
    results_path=os.path.join(root_dir,'results',foldername)

elif only_evaluate==True:
    results_path=os.path.join(root_dir,'results',folder_path_for_evaluation)
else:
    pass


data_path=os.path.join(results_path,'data')
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
#smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))

logs_path=os.path.join(results_path,'logs')

# # # Get Number of Images in Test Dir
# # lst = os.listdir(test_images_path) # your directory path
# # number_files = len(lst)

# Get Number of Images in COCO VAL Dataset
lst = get_coco_val_set_imgs("C:\\Users\\hashot51\\Desktop\\TensorFlow-2.x-YOLOv3\\model_data\\coco\\val2017.txt") # your directory path
number_files = len(lst)
# print("lst",lst[:100])

# print('number of files: ',number_fies)



if predict_then_evaluate==True or only_predict==True:
    # Create Folder with Datetime to store results of stream
    os.mkdir(results_path)
    os.mkdir(data_path)
    # Create Folders in which to store YOLO and SMOKE Frames
    os.mkdir(yolo_image_stream_path)
    #os.mkdir(smoke_image_stream_path)
    # Create foler in which to store text file logs
    os.mkdir(logs_path)
    os.mkdir(groundtruth_image_stream_path)




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
    



n=number_files
yolo_metrics_evaluator=metrics_evaluator("YOLOv3",n,logs_path,results_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)

for fileid,filename in enumerate(lst):

    #filepath=
    #ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+".png") # SMOKE
    ordered_filepath=os.path.join(coco_dataset_path,filename) # KITTI
    frame=cv2.imread(ordered_filepath)
    print('frame: ',fileid)



    # groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    # groundtruth_image=plot_groundtruth(frame,groundtruth)
    # cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid)+'.png'),groundtruth_image)

    if predict_then_evaluate==True or only_predict==True:
        boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)

        boxs,classes,scores,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)

        #yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes,scores=scores)
        yolo_predictions_list=yolo_2_json_output_format(boxs=boxs,classes=classes,scores=scores)

        print('yolo predictions list: ',yolo_predictions_list)
        #write_prediction(data_path,fileid,yolo_predictions_list)
        write_json(data_path,fileid,yolo_predictions_list)

        output_img=plot_prediction(frame,yolo_predictions_list)
        cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),output_img)


    if only_evaluate==True or predict_then_evaluate==True:
        yolo_predictions_read_from_file=read_prediction(data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=yolo_metrics_evaluator.eval_metrics(groundtruth,yolo_predictions_read_from_file)





precision_evaluation_path='.\SMOKE\smoke\data\datasets\evaluation\kitti\kitti_eval_40\eval12.exe'


if only_evaluate==True or predict_then_evaluate==True:
    yolo_metrics_evaluator.run_kitti_AP_evaluation_executable(root_dir,precision_evaluation_path,"data")
    yolo_metrics_evaluator.construct_dataframe()
    yolo_metrics_evaluator.show_results()




