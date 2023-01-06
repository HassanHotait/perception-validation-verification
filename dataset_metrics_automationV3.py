from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
import matplotlib.pyplot as plt
import cv2
import torch
import os
from datetime import datetime
from EvaluatorClass4 import metrics_evaluator,plot_groundtruth,MetricsHolder,new_plot_groundtruth
from metrics_functions_from_evaluation_script import smoke_get_n_classes,plot_prediction,read_groundtruth,get_key,read_prediction,write_prediction,new_read_groundtruth,video_writer
import subprocess
import argparse
import pandas as pd
from Dataset2Kitti.SMOKE_Visualizer import detectionInfo,SMOKE_Viz
from DepthEvaluatorClass import depth_evaluator

import dataframe_image as dfi

parser = argparse.ArgumentParser(description="Program Arg Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-root_dir", "--root_directory", default="C:/Users/hashot51/Desktop/perception-validation-verification", help="repository root directory")
args = parser.parse_args()


# Define Settings
SMOKE_threshold_setting=cfg.TEST.DETECTIONS_THRESHOLD
print("SMOKE Running @ Score Threshold = ",SMOKE_threshold_setting)
print("SMOKE threshold config: ",SMOKE_threshold_setting)
# Data is "Kitti" or "Prescan"
dataset="Prescan"
# Select Prescan Scenario (Used Only if dataset="Prescan")
prescan_scenario_id=16
prescan_datalogger_id=1
# Select Kitti Datasplit  (train.txt-val.txt-trainval.txt-custom) (Used Only if dataset="Kitti")
kitti_datasplit="val.txt"
## Run Configuration
# If Visualize is True predictions are visualized per frame and saved | else data is saved in results folder
visualize=False
predict_then_evaluate=True
only_predict=False
only_evaluate=False

# Evaluate AP wrt 11 or 40 recall points
kitt_eval_recall_pts=11
# Folder Used when only_evaluate=True
folder_path_for_evaluation='StreamSMOKE_get_FP2022_12_27_21_45_15'
# Define Test Name
short_test_description="fix_eval_exe"
if dataset=="Kitti":
    stream_id='SMOKE_{}_dataset_@_{}_thresh_split_{}_{}'.format(dataset,SMOKE_threshold_setting,kitti_datasplit,short_test_description)
elif dataset=="Prescan":
    dataset_id="Prescan_Scenario{}_logger{}".format(prescan_scenario_id,prescan_datalogger_id)
    stream_id='SMOKE_{}_dataset_@_{}_thresh_{}'.format(dataset,SMOKE_threshold_setting,short_test_description)

session_datetime=datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

# Define Directories
root_dir=args.root_directory#'/home/hashot51/Projects/perception-validation-verification'

if dataset=="Kitti":
    # Kitti Dataset
    boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
    test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')

    path_to_datasplit_csv_file=os.path.join(root_dir,"SMOKE\\datasets\\kitti\\training\\ImageSets\\{}".format(kitti_datasplit))
    #"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\ImageSets\\val.txt"
    datasplit_filename_list=pd.read_csv(path_to_datasplit_csv_file).values.tolist()
    # print("val split: \n",val_split_list)
    labels_extension='.txt'
elif dataset=="Prescan":
    # Prescan Scenario 15 Dataset
    boxs_groundtruth_path=os.path.join(root_dir,'Dataset2Kitti\\PrescanRawData_Scenario{}\\DataLogger{}\\label_2'.format(prescan_scenario_id,prescan_datalogger_id))
    test_images_path=os.path.join(root_dir,'Dataset2Kitti\\PrescanRawData_Scenario{}\\DataLogger{}\\images_2'.format(prescan_scenario_id,prescan_datalogger_id))
    labels_extension='.txt'
else:
    raise "Input Dataset Unknown"


# Define new results directory or use older one for evaluation

if predict_then_evaluate==True or only_predict==True:
    results_path=os.path.join(root_dir,'results',foldername)

elif only_evaluate==True:
    results_path=os.path.join(root_dir,'results',folder_path_for_evaluation).replace("/","\\")
else:
    pass
    
# Define More Directories
data_path=os.path.join(results_path,'smoke_predictions')
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
logs_path=os.path.join(results_path,'logs')
plot_path_for_evaluation=os.path.join(results_path,"plot")

smoke_3D_image_stream_path=os.path.join(results_path,'smoke_3D_image_stream')
smoke_3D_birdview_stream_path=os.path.join(results_path,'smoke_3D_birdview_stream')
smoke_3D_overview_stream_path=os.path.join(results_path,'smoke_3D_overview_stream')

smoke_depth_evaluation_path=os.path.join(results_path,'smoke_depth_evaluation')






if predict_then_evaluate==True or only_predict==True:
    # Create Folder with Datetime to store results of stream
    os.mkdir(results_path)
    os.mkdir(data_path)
    #os.mkdir(plot_path_for_evaluation)
    # Create Folders in which to store YOLO and SMOKE Frames
    os.mkdir(smoke_image_stream_path)
    # Create foler in which to store text file logs
    os.mkdir(logs_path)
    os.mkdir(groundtruth_image_stream_path)
    os.mkdir(smoke_3D_image_stream_path)
    os.mkdir(smoke_3D_birdview_stream_path)
    os.mkdir(smoke_3D_overview_stream_path)
    os.mkdir(smoke_depth_evaluation_path)

# Val Split

# path_to_val_csv_file="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\ImageSets\\val.txt"
# val_split_list=pd.read_csv(path_to_val_csv_file).values.tolist()
# print("val split: \n",val_split_list)

# Get Number of Images in Test Dir
lst = os.listdir(test_images_path) # your directory path
number_files =20#len(lst)
print('number of files: ',number_files)

# Setup SMOKE
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


# Unit Test Frames of Interest 
frames_of_interest=[1,2,3,4,5,8,10,15,16,18,19,21,23,25,26,27,37,40,42,43,48,51,60,62,68,73,76]



#n=5#number_files#len(frames_of_interest[:8])

smoke_metrics_evaluator=metrics_evaluator("SMOKE",number_files,logs_path,results_path)
smoke_depth_evaluator=depth_evaluator(number_files,smoke_depth_evaluation_path)

for fileid in  range(number_files):
    
    if dataset=="Kitti":
        if kitti_datasplit!="custom":
            fileid=datasplit_filename_list[fileid][0]

    #fileid=fileid[0]
    # Read Frame and Visualize Groundtruth 2D
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    frame_2D_detections=frame.copy()
    groundtruth=new_read_groundtruth(boxs_groundtruth_path,fileid,extension=labels_extension,dataset=dataset)
    print("Groundtruth: ",groundtruth)
    groundtruth_image=new_plot_groundtruth(frame_2D_detections,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,str(fileid).zfill(6)+'.png'),groundtruth_image)
    #Viz 3D
    viz=SMOKE_Viz(frame,lat_range_m=40,long_range_m=70,scale=20,dataset=dataset)
    if predict_then_evaluate==True or only_predict==True:
        # Predict,Visualize & Save predictions in logs
        smoke_predictions_list,K=preprocess_then_predict(model,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset)
        write_prediction(data_path,fileid,smoke_predictions_list)
        output_img=plot_prediction(frame_2D_detections,smoke_predictions_list)
        frame_detection_info_list=[detectionInfo(pred) for pred in smoke_predictions_list]
        print("Groundtruth input for GtInfo: ",groundtruth)
        gt_info_list=[gt for gt in groundtruth]




        # Predictions
        viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list)
        viz.draw_birdeyes(obj_list=frame_detection_info_list)
        # GroundTruth
        viz.draw_3Dbox(K,frame.copy(),gt_list=gt_info_list)
        viz.draw_birdeyes(obj_list=gt_info_list)
        smoke_image_results_filepath=os.path.join(smoke_3D_image_stream_path,str(fileid).zfill(6)+'.png')
        smoke_birdview_results_filepath=os.path.join(smoke_3D_birdview_stream_path,str(fileid).zfill(6)+'.png')
        smoke_overview_results_filepath=os.path.join(smoke_3D_overview_stream_path,str(fileid).zfill(6)+'.png')
        viz.save_fig(image_path=smoke_image_results_filepath,
            birdview_path=smoke_birdview_results_filepath,
            overview_path=smoke_overview_results_filepath
            )
        if visualize==True:
            viz.show()

        plt.close()

        smoke_depth_evaluator.match_pred_with_gt_by_object_center(K,groundtruth=gt_info_list,predictions=frame_detection_info_list)
        #smoke_depth_evaluator.viz_object_centers(frame.copy())
        smoke_depth_evaluator.plot_eror()

        cv2.imwrite(os.path.join(smoke_image_stream_path,str(fileid).zfill(6)+'.png'),output_img)
    if only_evaluate==True or predict_then_evaluate==True:
        # Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file=read_prediction(data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator.eval_metrics(groundtruth,smoke_predictions_read_from_file)

        # cv2.imshow('Visualize 2D Image',output_img)
        # cv2.waitKey(0)



precision_evaluation_path='.\SMOKE\smoke\data\datasets\evaluation\kitti\kitti_eval_{}\kitti_eval_{}.exe'.format(kitt_eval_recall_pts,kitt_eval_recall_pts)
#boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\label_2"



command = "{} {} {} ".format(precision_evaluation_path,boxs_groundtruth_path, results_path.replace("/","\\"))#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\results\\Streamtest_AP_eval2022_12_23_15_05_34"
print(command)

if only_evaluate==True or predict_then_evaluate==True:
    smoke_metrics_evaluator.run_kitti_AP_evaluation_executable(root_dir,precision_evaluation_path,"smoke_predictions",label_dir=boxs_groundtruth_path)
    smoke_metrics_evaluator.construct_dataframe()
    #smoke_metrics_evaluator.show_results()


img_results_folders=[smoke_image_stream_path]
video_writer(images_folder_path=smoke_image_stream_path,video_filename="2D Detections Results Video.mp4",fps=10,results_path=results_path)
video_writer(images_folder_path=smoke_3D_image_stream_path,video_filename="3D Detections Results Video.mp4",fps=10,results_path=results_path)
video_writer(images_folder_path=smoke_3D_birdview_stream_path,video_filename="3D Birdview Results Video.mp4",fps=10,results_path=results_path)
video_writer(images_folder_path=smoke_3D_overview_stream_path,video_filename="3D Overview Results Video.mp4",fps=10,results_path=results_path)


# Video Writer   