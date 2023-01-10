from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
import matplotlib.pyplot as plt
import cv2
import torch
import os
from datetime import datetime
from EvaluatorClass3 import metrics_evaluator,plot_groundtruth,MetricsHolder
from metrics_functions_from_evaluation_script import smoke_get_n_classes,plot_prediction,read_groundtruth,get_key,read_prediction,write_prediction,get_class_AP,construct_dataframe_v2,construct_dataframe,yolo_2_smoke_output_format,plot_SMOKE_vs_YOLO
import subprocess
import argparse
import pandas as pd

import dataframe_image as dfi

from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
from YOLO_toolbox  import setup_YOLO,setup_tracker,preprocess_predict_YOLO,postprocess_YOLO,yolo_visualize

parser = argparse.ArgumentParser(description="Program Arg Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-root_dir", "--root_directory", default="C:/Users/hashot51/Desktop/perception-validation-verification", help="repository root directory")
args = parser.parse_args()


# Define Settings

visualize_3D=False
predict_then_evaluate=True
only_predict=False
only_evaluate=False


# Folder Used when only_evaluate=True
folder_path_for_evaluation='StreamSMOKEvsYOLO2022_12_26_00_02_16'



# Define Test Name
stream_id='SMOKEvsYOLO_Offical_'
session_datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
foldername='Stream_'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

# Define Directories

root_dir=args.root_directory#'/home/hashot51/Projects/perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')


# Define new results directory or use older one for evaluation

if predict_then_evaluate==True or only_predict==True:
    results_path=os.path.join(root_dir,'results',foldername)

elif only_evaluate==True:
    results_path=os.path.join(root_dir,'results',folder_path_for_evaluation)
else:
    pass
    
# Define More Directories

smoke_data_path=os.path.join(results_path,'smoke_data')
yolo_data_path=os.path.join(results_path,'yolo_data')
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
yolo_logs_path=os.path.join(results_path,'yolo_logs')
smoke_logs_path=os.path.join(results_path,'smoke_logs')
plot_path_for_evaluation=os.path.join(results_path,"plot")

# Get Number of Images in Test Dir
lst = os.listdir(test_images_path) # your directory path
number_files = len(lst)
print('number of files: ',number_files)


if predict_then_evaluate==True or only_predict==True:
    # Create Folder with Datetime to store results of stream
    os.mkdir(results_path)
    os.mkdir(smoke_data_path)
    os.mkdir(yolo_data_path)
    #os.mkdir(plot_path_for_evaluation)
    # Create Folders in which to store YOLO and SMOKE Frames
    os.mkdir(smoke_image_stream_path)
    os.mkdir(yolo_image_stream_path)
    # Create foler in which to store text file logs
    os.mkdir(yolo_logs_path)
    os.mkdir(smoke_logs_path)
    os.mkdir(groundtruth_image_stream_path)




if only_predict==True or predict_then_evaluate==True:
    # Setup SMOKE
    args = default_argument_parser().parse_args()
    model,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


    # Setup YOLOV3 | Load Weights | Initialize Tracker for Object IDS | Initialize Feature Encoder
    yolo,class_names=setup_YOLO(path_to_class_names='YOLO_Detection/data/labels/coco.names',
                                path_to_weights='YOLO_Detection/weights/yolov3.tf')
    model_filename = 'YOLO_Detection/model_data/mars-small128.pb'
    tracker,encoder=setup_tracker(model_filename)


# Unit Test Frames of Interest 
frames_of_interest=[1,2,3,4,5,8,10,15,16,18,19,21,23,25,26,27,37,40,42,43,48,51,60,62,68,73,76]



n=100#number_files#len(frames_of_interest[:8])
smoke_metrics_evaluator=metrics_evaluator("SMOKE",n,smoke_logs_path,results_path)
yolo_metrics_evaluator=metrics_evaluator("YOLOv3",n,yolo_logs_path,results_path)



for fileid in range(n):

    # Read Frame and Visualize Groundtruth 2D
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid,extension=".txt")
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),groundtruth_image)

    if predict_then_evaluate==True or only_predict==True:
        #  SMOKE Predict,Visualize & Save predictions in logs
        smoke_predictions_list,K=preprocess_then_predict(model,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset="Kitti")
        write_prediction(smoke_data_path,fileid,smoke_predictions_list)
        output_img=plot_prediction(frame.copy(),smoke_predictions_list)
        #print('P2: ',P2)
        #b,g,r=cv2.split(frame)
        #rgb_frame=cv2.merge([r,g,b])
        #fig,img,birdimage=visualize((900,900),smoke_predictions_list,K,rgb_frame)
        #  plt.show()
        cv2.imwrite(os.path.join(smoke_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),output_img)

        #  YOLO Predict,Visualize & Save predictions in logs
        boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame.copy())

        boxs,classes,scores,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)

        yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes,scores=scores)
        #yolo_predictions_list=yolo_2_json_output_format(boxs=boxs,classes=classes,scores=scores)

        print('yolo predictions list: ',yolo_predictions_list)
        write_prediction(yolo_data_path,fileid,yolo_predictions_list)
        #write_json(data_path,fileid,yolo_predictions_list)

        output_img=plot_prediction(frame.copy(),yolo_predictions_list)
        cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),output_img)
    if only_evaluate==True or predict_then_evaluate==True:
        # SMOKE Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file=read_prediction(smoke_data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator.eval_metrics(groundtruth,smoke_predictions_read_from_file)

        # cv2.imshow('Visualize 2D Image',output_img)
        # cv2.waitKey(0)

        # YOLO Read predictions from file then feed to evaluator
        yolo_predictions_read_from_file=read_prediction(yolo_data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=yolo_metrics_evaluator.eval_metrics(groundtruth,yolo_predictions_read_from_file)



precision_evaluation_path='.\SMOKE\smoke\data\datasets\evaluation\kitti\kitti_eval_40\eval12.exe'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\label_2"



command = "{} {} {} ".format(precision_evaluation_path,boxs_groundtruth_path, results_path.replace("/","\\"))#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\results\\Streamtest_AP_eval2022_12_23_15_05_34"
print(command)

if only_evaluate==True or predict_then_evaluate==True:
    smoke_metrics_evaluator.run_kitti_AP_evaluation_executable(root_dir,precision_evaluation_path,"smoke_data")
    smoke_metrics_table_df=smoke_metrics_evaluator.construct_dataframe()
    smoke_metrics_evaluator.show_results()

    ####################################################################
    yolo_metrics_evaluator.run_kitti_AP_evaluation_executable(root_dir,precision_evaluation_path,"yolo_data")
    yolo_metrics_table_df=yolo_metrics_evaluator.construct_dataframe()
    yolo_metrics_evaluator.show_results()


    plot_SMOKE_vs_YOLO(smoke_metrics_table_df,yolo_metrics_table_df,results_path)


    