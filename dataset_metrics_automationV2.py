
from logging import root
from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
from SMOKE.tools.box import visualize
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import numpy as np
torch.cuda.empty_cache()
import os
from datetime import datetime
#from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
# from evaluation_toolbox import smoke_get_n_classes,yolo_get_n_classes,Point,get_IoU,get_evaluation_metrics,plot_groundtruth
# from metrics_functions_from_evaluation_script import metrics_evaluator
#from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,smoke_get_n_classes,yolo_get_n_classes,read_groundtruth
#from metrics_functions_from_evaluation_script import metrics_evaluator
from EvaluatorClass2 import metrics_evaluator,plot_groundtruth
from metrics_functions_from_evaluation_script import smoke_get_n_classes,plot_prediction,read_groundtruth,get_key,read_prediction,write_prediction,convert_prediction_text_format,get_class_AP,construct_dataframe
import glob
import subprocess
import csv
import pandas as pd
import dataframe_image as dfi

from IPython.display import display
predict_then_evaluate=False
only_predict=False
only_evaluate=True

folder_path_for_evaluation='Streamkitti_training_set_smoke_metrics2022_11_22_10_50_51'







stream_id='kitti_training_set_smoke_metrics'
session_datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')

if predict_then_evaluate==True or only_predict==True:
    results_path=os.path.join(root_dir,'results',foldername)

elif only_evaluate==True:
    results_path=os.path.join(root_dir,'results',folder_path_for_evaluation)
else:
    pass
    
#yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
data_path=os.path.join(results_path,'data')
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))

groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
logs_path=os.path.join(results_path,'logs')

lst = os.listdir(test_images_path) # your directory path
number_files = len(lst)


if predict_then_evaluate==True or only_predict==True:
# Create Folder with Datetime to store results of stream
    os.mkdir(results_path)
    os.mkdir(data_path)
    # Create Folders in which to store YOLO and SMOKE Frames
    #os.mkdir(yolo_image_stream_path)
    os.mkdir(smoke_image_stream_path)
    # Create foler in which to store text file logs
    os.mkdir(logs_path)
    os.mkdir(groundtruth_image_stream_path)


# Setup SMOKE

args = default_argument_parser().parse_args()
model,network_configuration,gpu_device,cpu_device=setup_network(args)

# Load Weights
checkpointer = DetectronCheckpointer(
    cfg, model, save_dir=cfg.OUTPUT_DIR
)
ckpt=cfg.MODEL.WEIGHT
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)





# from absl import flags
# import sys
# FLAGS = flags.FLAGS
# FLAGS(sys.argv)
# from YOLO_toolbox  import setup_YOLO,setup_tracker,preprocess_predict_YOLO,postprocess_YOLO,yolo_visualize

# Setup YOLOV3 | Load Weights | Initialize Tracker for Object IDS | Initialize Feature Encoder
# yolo,class_names=setup_YOLO(path_to_class_names='YOLO_Detection/data/labels/coco.names',
#                             path_to_weights='YOLO_Detection/weights/yolov3.tf')
# model_filename = 'YOLO_Detection/model_data/mars-small128.pb'
# tracker,encoder=setup_tracker(model_filename)

n=7478
smoke_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0
for i in range(n):#filepath in glob.glob(os.path.join(test_images_path,'*.png')):

    #fileid=i
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),groundtruth_image)

    if predict_then_evaluate==True or only_predict==True:

        pilimage=Image.fromarray(frame)
        smoke_predictions_list=preprocess_then_predict(model,network_configuration,fileid,pilimage,gpu_device,cpu_device)
        #print('Length ------------------------: ',len(smoke_predictions_list[0]))
        write_prediction(data_path,fileid,smoke_predictions_list)




        output_img=plot_prediction(groundtruth_image,smoke_predictions_list)
        cv2.imwrite(os.path.join(smoke_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),output_img)
    #print('SMOKE predictions list: ',smoke_predictions_list)

    
    
    #print('SMOKE Predictions List: ',smoke_predictions_list)
    #smoke_detections_n=len(smoke_predictions_list)
    #smoke_n_classes=smoke_get_n_classes(smoke_predictions_list)


    if only_evaluate==True or predict_then_evaluate==True:

        smoke_predictions_read_from_file=read_prediction(data_path,fileid)


        #print('SMOKE Predictions read from file: ',smoke_predictions_read_from_file)

        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator.evaluate_metrics(groundtruth,smoke_predictions_read_from_file)

    #print('Ground Truth: ',groundtruth)

            
    fileid+=1


# Hide GPU 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 



# while ret==True:
#     #YOLO Output
#     print('frame: ',fileid)
#     boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
#     print('YOLO Classes: ',classes)
#     YOLO_output=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
#     print('YOLO Output: ',YOLO_output)
#     yolo_n_classes=yolo_get_n_classes(YOLO_output)
#     yolo_log.write(str(fileid)+' '+str(yolo_n_classes[0])+' '+str(yolo_n_classes[1])+' '+str(yolo_n_classes[2])+' '+str(yolo_n_classes[3]))
#     yolo_log.write('\n')
#     img=yolo_visualize(tracker,frame,fileid)
#     yolo_vid_writer.write(img)
#     cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),img)


#     fileid+=1
#     print('fileid: ',fileid)
#     print('YOLO')
#     ret,frame=vid.read()

# yolo_log.close()


#####################################################################################################################



#print('Current Working Directory: ',os. getcwd())

torch.cuda.empty_cache()

precision_evaluation_path='SMOKE/smoke/data/datasets/evaluation/kitti/kitti_eval_40/evaluate_object_3d_offline'
boxs_groundtruth_path='SMOKE/datasets/kitti/training/label_2'





command = "./{} {} {} ".format(precision_evaluation_path,boxs_groundtruth_path, results_path)
#print(command)

if only_evaluate==True or predict_then_evaluate==True:
    average_precision_command=subprocess.check_output(command, shell=True, universal_newlines=True).strip()

    print(average_precision_command)



    cars_easy_AP,cars_moderate_AP,cars_hard_AP=get_class_AP(results_path,'Car')
    pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=get_class_AP(results_path,'Pedestrian')


    cars_AP=[cars_easy_AP,cars_moderate_AP,cars_hard_AP]
    pedestrians_AP=[pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP]

    df=construct_dataframe(cars_AP,pedestrians_AP,car_metrics,pedestrian_metrics,difficulty_metrics,n_object_classes,n_object_difficulties)

    dfi.export(df,os.path.join(results_path,'MetricsTable.png'))


    metrics_img=cv2.imread(os.path.join(results_path,'MetricsTable.png'))
    cv2.imshow('Metrics',metrics_img)
    cv2.waitKey(0)