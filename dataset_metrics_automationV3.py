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
from metrics_functions_from_evaluation_script import smoke_get_n_classes,new_plot_prediction,read_groundtruth,get_key,read_prediction,new_write_prediction,new_read_groundtruth,video_writer,get_dataset_depth_stats
import subprocess
import argparse
import pandas as pd
from Dataset2Kitti.SMOKE_Visualizer import detectionInfo,SMOKE_Viz
from DepthEvaluatorClass import depth_evaluator
import copy
import math

import dataframe_image as dfi
from depth_POC_Prescan import get_optimized_depth_flat_road,get_optimized_depth_by_stats

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
kitt_eval_recall_pts=40
# Folder Used when only_evaluate=True
folder_path_for_evaluation='StreamSMOKE_Prescan_dataset_@_0.25_thresh_compare_different_depths_methods_2023_01_08_00_26_12'
# Define Test Name
short_test_description="compare_different_depths_methods"
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
data_path_v1=os.path.join(results_path,'smoke_predictions_v1')
data_path_v2=os.path.join(results_path,'smoke_predictions_v2')
data_path_v3=os.path.join(results_path,'smoke_predictions_v3')
data_path_v4=os.path.join(results_path,'smoke_predictions_v4')
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
logs_path=os.path.join(results_path,'logs')
logs_path_v1=os.path.join(results_path,'logs_v1')
logs_path_v2=os.path.join(results_path,'logs_v2')
logs_path_v3=os.path.join(results_path,'logs_v3')
logs_path_v4=os.path.join(results_path,'logs_v4')
kitti_AP_eval_path_smoke=os.path.join(results_path,'kitti_AP_evaluation')
kitti_AP_eval_path_smoke_v1=os.path.join(results_path,'kitti_AP_evaluation_v1')
kitti_AP_eval_path_smoke_v2=os.path.join(results_path,'kitti_AP_evaluation_v2')
kitti_AP_eval_path_smoke_v3=os.path.join(results_path,'kitti_AP_evaluation_v3')
kitti_AP_eval_path_smoke_v4=os.path.join(results_path,'kitti_AP_evaluation_v4')
plot_path_for_evaluation=os.path.join(results_path,"plot")

smoke_3D_image_stream_path=os.path.join(results_path,'smoke_3D_image_stream')
smoke_3D_birdview_stream_path=os.path.join(results_path,'smoke_3D_birdview_stream')
smoke_3D_overview_stream_path=os.path.join(results_path,'smoke_3D_overview_stream')
smoke_depth_evaluation_path=os.path.join(results_path,'smoke_depth_evaluation')






if predict_then_evaluate==True or only_predict==True:
    # Create Folder with Datetime to store results of stream
    os.mkdir(results_path)
    os.mkdir(data_path)
    os.mkdir(data_path_v1)
    os.mkdir(data_path_v2)
    os.mkdir(data_path_v3)
    os.mkdir(data_path_v4)
    #os.mkdir(plot_path_for_evaluation)
    # Create Folders in which to store YOLO and SMOKE Frames
    os.mkdir(smoke_image_stream_path)
    # Create foler in which to store text file logs
    os.mkdir(logs_path)
    os.mkdir(logs_path_v1)
    os.mkdir(logs_path_v2)
    os.mkdir(logs_path_v3)
    os.mkdir(logs_path_v4)
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
number_files =20#00#len(lst)
print('number of files: ',number_files)

# Setup SMOKE Default
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model_default,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model_default, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

# Setup SMOKE V1
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model_v1,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model_v1, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

# Setup SMOKE V2
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model_v2,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model_v2, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

#Setup SMOKE V3
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model_v3,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model_v3, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

#Setup SMOKE V4
if only_predict==True or predict_then_evaluate==True:
    args = default_argument_parser().parse_args()
    model_v4,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model_v4, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


# Unit Test Frames of Interest 
frames_of_interest=[1,2,3,4,5,8,10,15,16,18,19,21,23,25,26,27,37,40,42,43,48,51,60,62,68,73,76]



#n=5#number_files#len(frames_of_interest[:8])

# SMOKE Normal Output (Not Modified) # depths_ref ALL Frames
smoke_metrics_evaluator=metrics_evaluator("SMOKE default",number_files,logs_path,results_path)
smoke_depth_evaluator=depth_evaluator(number_files,smoke_depth_evaluation_path)

# SMOKE depths_ref per frames
smoke_metrics_evaluator_ref_per_frame=metrics_evaluator("SMOKE ref per frame",number_files,logs_path_v1,results_path)
smoke_depth_evaluator_ref_per_frame=depth_evaluator(number_files,smoke_depth_evaluation_path)

# # SMOKE depths_ref per Scenario
smoke_metrics_evaluator_ref_per_scenario=metrics_evaluator("SMOKE ref per scenario",number_files,logs_path_v2,results_path)
smoke_depth_evaluator_ref_per_scenario=depth_evaluator(number_files,smoke_depth_evaluation_path)
# # SMOKE My method
# smoke_metrics_evaluator_my_depth_method=metrics_evaluator("SMOKE Flat Road Depth",number_files,logs_path_v3,results_path)
# smoke_depth_evaluator_my_depth_method=depth_evaluator(number_files,smoke_depth_evaluation_path)

# SMOKE MY Output
smoke_metrics_evaluator_my_depth_method=metrics_evaluator("SMOKE my depth",number_files,logs_path_v3,results_path)
smoke_depth_evaluator_my_depth_method=depth_evaluator(number_files,smoke_depth_evaluation_path)

# SMOKE Ideal Depth
smoke_metrics_evaluator_ideal_depth=metrics_evaluator("SMOKE Ideal Depth",number_files,logs_path_v4,results_path)
smoke_depth_evaluator_ideal_depth=depth_evaluator(number_files,smoke_depth_evaluation_path)

cfg_stats=cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE
for fileid in  range(number_files):

    mean,std,_=get_dataset_depth_stats(labels_path=boxs_groundtruth_path,frame_id=fileid,depth_condition=45,extension=".txt")
    
    if math.isnan(mean) or math.isnan(std):
        smoke_metrics_evaluator.frame_id+=1
        smoke_metrics_evaluator_ref_per_frame.frame_id+=1
        smoke_metrics_evaluator_ref_per_scenario.frame_id+=1
        smoke_depth_evaluator.internal_counter+=1
        smoke_depth_evaluator_ref_per_scenario.internal_counter+=1
        smoke_depth_evaluator_ref_per_frame.internal_counter+=1
        #smoke_depth_evaluator_.n_frames-=1
        continue
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
        ## Predict,Visualize & Save predictions in logs
        # SMOKE Default
        smoke_predictions_list,K=preprocess_then_predict(model_default,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset,default=True)
        frame_detection_info_list=[detectionInfo(pred) for pred in smoke_predictions_list]
        new_write_prediction(data_path,fileid,frame_detection_info_list)

        # SMOKE V1
        print("file id: ",fileid)
        smoke_predictions_list_v1,K=preprocess_then_predict(model_v1,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset,default=False,method="frame_depth_refs",frame_id=fileid)
        frame_detection_info_list_v1=[detectionInfo(pred) for pred in smoke_predictions_list_v1]
        new_write_prediction(data_path_v1,fileid,frame_detection_info_list_v1)
        #frame_detection_info_list_optimized_per_frame=frame_detection_info_list.copy()#[detectionInfo(copy.deepcopy(pred)) for pred in smoke_predictions_list]#copy.deepcopy(frame_detection_info_list)#[detectionInfo(pred) for pred in smoke_predictions_list]
        #frame_detection_info_list_my_depth_method=[detectionInfo(copy.deepcopy(pred)) for pred in smoke_predictions_list]#copy.deepcopy(frame_detection_info_list)#[detectionInfo(pred) for pred in smoke_predictions_list]
        # SMOKE V2
        smoke_predictions_list_v2,K=preprocess_then_predict(model_v2,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset,default=False,method="dataset_depth_refs")
        frame_detection_info_list_v2=[detectionInfo(pred) for pred in smoke_predictions_list_v2]
        new_write_prediction(data_path_v2,fileid,frame_detection_info_list_v2)
        # SMOKE V3
        smoke_predictions_list_v3,K=preprocess_then_predict(model_v3,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset,default=False,method="flat_road")
        frame_detection_info_list_v3=[detectionInfo(pred) for pred in smoke_predictions_list_v3]
        new_write_prediction(data_path_v3,fileid,frame_detection_info_list_v3)

        # SMOKE V4
        smoke_predictions_list_v4,K=preprocess_then_predict(model_v4,cfg,fileid,ordered_filepath,gpu_device,cpu_device,dataset=dataset,default=False,method="ideal_depth",frame_id=fileid)
        frame_detection_info_list_v4=[detectionInfo(pred) for pred in smoke_predictions_list_v4]
        new_write_prediction(data_path_v4,fileid,frame_detection_info_list_v4)
        # new_write_prediction(data_path,fileid,frame_detection_info_list)
        # output_img=new_plot_prediction(frame_2D_detections,frame_detection_info_list,color=(0,0,255))
        # # # SMOKE Ref Per Frame
        # frame_detection_info_list_optimized_per_frame=[get_optimized_depth_by_stats(cfg_stats=cfg_stats,pred_obj=copy.copy(obj),path=boxs_groundtruth_path,fileid=fileid) for obj in frame_detection_info_list]
        # new_write_prediction(data_path_v1,fileid,frame_detection_info_list_optimized_per_frame)
        # # output_img=new_plot_prediction(output_img,frame_detection_info_list_optimized_per_frame,color=(0,0,255))

        # frame_detection_info_list_optimized_per_scenario=[get_optimized_depth_by_stats(cfg_stats=cfg_stats,pred_obj=copy.copy(obj),path=boxs_groundtruth_path,fileid="All") for obj in frame_detection_info_list]
        # new_write_prediction(data_path_v2,fileid,frame_detection_info_list_optimized_per_scenario)
        # #  # SMOKE My Depth Computation
        # frame_detection_info_list_my_depth_method=[get_optimized_depth_flat_road(K=K,pred_obj=copy.copy(obj),camera_height=1.32) for obj in frame_detection_info_list]
        # #get_optimized_depth_flat_road(K,detection_info_list_optimized_flat_ground[i],camera_height=camera_height)
        # new_write_prediction(data_path_v3,fileid,frame_detection_info_list_my_depth_method)
        # output_img=new_plot_prediction(output_img,frame_detection_info_list_my_depth_method,color=(0,0,255))


        # for j,pred_obj in enumerate(frame_detection_info_list):
        #     optimized_pred_by_frame_depth_stats=get_optimized_depth_by_stats(cfg_stats=cfg_stats,pred_obj=pred_obj,path=boxs_groundtruth_path,fileid=fileid)
        #     optimized_pred_by_frame_depth_stats.get_preds_string()
        #     frame_detection_info_list_optimized_per_frame[j]=optimized_pred_by_frame_depth_stats
        #     print("Pred Obtimized by frame stats method: ",optimized_pred_by_frame_depth_stats.tz)
        #new_write_prediction(data_path_v1,fileid,frame_detection_info_list_optimized_per_frame)
            #optimized_pred_flat_road= get_optimized_depth_flat_road(K,frame_detection_info_list_my_depth_method[j],camera_height=1.32)
            #optimized_pred_flat_road.get_preds_string()
            #print("Frame optimized by my depth method: ",optimized_pred_flat_road.tz)
            
            #frame_detection_info_list_my_depth_method[j]=optimized_pred_flat_road

        
            # new_write_prediction(data_path_v2,fileid,frame_detection_info_list_my_depth_method)

        #output_img=new_plot_prediction(output_img,frame_detection_info_list_optimized_per_frame,color=(0,0,255))
        #output_img=new_plot_prediction(output_img,frame_detection_info_list_my_depth_method,color=(0,255,255))





        # cv2.imshow("Output Img",output_img)
        # cv2.waitKey(0)






        # Predictions Default
        viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list)
        viz.draw_birdeyes(obj_list=frame_detection_info_list)
        # # Predictions V1
        # viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list_v1,optimized=True)
        # viz.draw_birdeyes(obj_list=frame_detection_info_list_v1)
        # # # Predictions V2
        # viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list_v2,optimized=True)
        # viz.draw_birdeyes(obj_list=frame_detection_info_list_v2)
        # # Predictions V3
        # viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list_v3,optimized=True)
        # viz.draw_birdeyes(obj_list=frame_detection_info_list_v3)
        # # Predictions V4
        viz.draw_3Dbox(K,frame,predictions_list=frame_detection_info_list_v4,optimized=True)
        viz.draw_birdeyes(obj_list=frame_detection_info_list_v4)
        # GroundTruth
        viz.draw_3Dbox(K,frame.copy(),gt_list=groundtruth)
        viz.draw_birdeyes(obj_list=groundtruth)
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

        smoke_depth_evaluator.match_pred_with_gt_by_object_center(K,groundtruth=groundtruth,predictions=frame_detection_info_list)#smoke_depth_evaluator.viz_object_centers(frame.copy())
        x_range_default,y_average=smoke_depth_evaluator.plot_eror("Default SMOKE")

        smoke_depth_evaluator_ref_per_frame.match_pred_with_gt_by_object_center(K,groundtruth=groundtruth,predictions=frame_detection_info_list_v1)#smoke_depth_evaluator.viz_object_centers(frame.copy())
        x_range_v1,y_average_v1=smoke_depth_evaluator_ref_per_frame.plot_eror("SMOKE Depth Ref Per Frame")

        smoke_depth_evaluator_ref_per_scenario.match_pred_with_gt_by_object_center(K,groundtruth=groundtruth,predictions=frame_detection_info_list_v2)#smoke_depth_evaluator.viz_object_centers(frame.copy())
        x_range_v2,y_average_v2=smoke_depth_evaluator_ref_per_scenario.plot_eror("SMOKE Depth Ref Per Scenario")

        smoke_depth_evaluator_my_depth_method.match_pred_with_gt_by_object_center(K,groundtruth=groundtruth,predictions=frame_detection_info_list_v3)#smoke_depth_evaluator.viz_object_centers(frame.copy())
        x_range_v3,y_average_v3=smoke_depth_evaluator_my_depth_method.plot_eror("My Depth")

        smoke_depth_evaluator_ideal_depth.match_pred_with_gt_by_object_center(K,groundtruth=groundtruth,predictions=frame_detection_info_list_v4)#smoke_depth_evaluator.viz_object_centers(frame.copy())
        x_range_v4,y_average_v4=smoke_depth_evaluator_ideal_depth.plot_eror("Ideal Depth")



        #cv2.imwrite(os.path.join(smoke_image_stream_path,str(fileid).zfill(6)+'.png'),output_img)
    if only_evaluate==True or predict_then_evaluate==True:
        ## SMOKE Default
        # Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file=read_prediction(data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator.eval_metrics(groundtruth,smoke_predictions_read_from_file)

        ## SMOKE Ref Per Frame
        # Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file_v1=read_prediction(data_path_v1,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator_ref_per_frame.eval_metrics(groundtruth,smoke_predictions_read_from_file_v1)

        ## SMOKE Ref Per Frame
        # Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file_v2=read_prediction(data_path_v2,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator_ref_per_scenario.eval_metrics(groundtruth,smoke_predictions_read_from_file_v2)
        
        
        # SMOKE My Depth Computation
        #Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file_v3=read_prediction(data_path_v3,fileid)
        car_metrics,pedestrian_metrics,cycist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator_my_depth_method.eval_metrics(groundtruth,smoke_predictions_read_from_file_v3)

        # SMOKE Ideal Depth
        #Read predictions from file then feed to evaluator
        smoke_predictions_read_from_file_v4=read_prediction(data_path_v4,fileid)
        car_metrics,pedestrian_metrics,cycist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator_ideal_depth.eval_metrics(groundtruth,smoke_predictions_read_from_file_v4)
        # cv2.imshow('Visualize 2D Image',output_img)
        # cv2.waitKey(0)



precision_evaluation_path='.\SMOKE\smoke\data\datasets\evaluation\kitti\kitti_eval_{}\kitti_eval_{}.exe'.format(kitt_eval_recall_pts,kitt_eval_recall_pts)
#boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\SMOKE\\datasets\\kitti\\training\\label_2"



#command = "{} {} {} ".format(precision_evaluation_path,boxs_groundtruth_path, results_path.replace("/","\\"))#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\results\\Streamtest_AP_eval2022_12_23_15_05_34"
#print(command)

if only_evaluate==True or predict_then_evaluate==True:
    smoke_metrics_evaluator.run_kitti_AP_evaluation_executable(precision_evaluation_path,predictions_foldername=data_path,label_dir=boxs_groundtruth_path,kitti_eval_path=kitti_AP_eval_path_smoke)
    smoke_metrics_evaluator.construct_dataframe(kitti_AP_eval_path_smoke)
    #smoke_metrics_evaluator.show_results()

    # SMOKE V1
    smoke_metrics_evaluator_ref_per_frame.run_kitti_AP_evaluation_executable(precision_evaluation_path,predictions_foldername=data_path_v1,label_dir=boxs_groundtruth_path,kitti_eval_path=kitti_AP_eval_path_smoke_v1)
    smoke_metrics_evaluator_ref_per_frame.construct_dataframe(kitti_AP_eval_path_smoke_v1)

    # SMOKE V2
    smoke_metrics_evaluator_ref_per_scenario.run_kitti_AP_evaluation_executable(precision_evaluation_path,predictions_foldername=data_path_v2,label_dir=boxs_groundtruth_path,kitti_eval_path=kitti_AP_eval_path_smoke_v2)
    smoke_metrics_evaluator_ref_per_scenario.construct_dataframe(kitti_AP_eval_path_smoke_v2)

    # SMOKE V3
    smoke_metrics_evaluator_my_depth_method.run_kitti_AP_evaluation_executable(precision_evaluation_path,predictions_foldername=data_path_v3,label_dir=boxs_groundtruth_path,kitti_eval_path=kitti_AP_eval_path_smoke_v3)
    smoke_metrics_evaluator_my_depth_method.construct_dataframe(kitti_AP_eval_path_smoke_v3)
    # SMOKE V4
    smoke_metrics_evaluator_ideal_depth.run_kitti_AP_evaluation_executable(precision_evaluation_path,predictions_foldername=data_path_v4,label_dir=boxs_groundtruth_path,kitti_eval_path=kitti_AP_eval_path_smoke_v4)
    smoke_metrics_evaluator_ideal_depth.construct_dataframe(kitti_AP_eval_path_smoke_v4)

# Visualize Comparison
average_plot_labels=["[{}-{}]".format(i*10,(i+1)*10) for i in range(0,7)]

fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(x_range_default,y_average,label="SMOKE Default | AP = {}".format(smoke_metrics_evaluator.cars_AP[0]))#,list(range(1,8)),y_average_v1,label="SMOKE Depth Ref per Frame")#,list(range(1,8)),)
print("X Default List: ",x_range_default)
print("Y Average Default List: ",y_average)
ax.plot(x_range_v1,y_average_v1,label="SMOKE Depth Ref per Frame | AP = {}".format(smoke_metrics_evaluator_ref_per_frame.cars_AP[0]))
print("X Ref per Frame List: ",x_range_v1)
print("Y Average Depth Ref per Frame List: ",y_average_v1)
ax.plot(x_range_v2,y_average_v2,label="SMOKE Depth Ref per Scenario | AP = {}".format(smoke_metrics_evaluator_ref_per_scenario.cars_AP[0]))
print("X Ref per Scenario List: ",x_range_v2)
print("Y Average Depth Ref per Scenario List: ",y_average_v2)
ax.plot(x_range_v3,y_average_v3,label="SMOKE Depth assuming Flat Road | AP = {}".format(smoke_metrics_evaluator_my_depth_method.cars_AP[0]))
print("Y Flat Road List : ",x_range_v3)
print("Y Average Flat Road List : ",y_average_v3)
ax.plot(x_range_v4,y_average_v4,label="SMOKE Ideal Depth | AP = {}".format(smoke_metrics_evaluator_ideal_depth.cars_AP[0]))
print("Y Flat Road List : ",x_range_v4)
print("Y Average Flat Road List : ",y_average_v4)
ax.legend()
#plt.plot(list(ange(1,8)),average_y)
plt.xticks(ticks=list(range(1,8)),labels=average_plot_labels)
plt.ylabel("Average Error [m]")
plt.xlabel("GroundTruth Range [m]")
plt.xlim([0,8])
#plt.ylim([0,40])
plt.grid(True)
plt.show()
fig.savefig(os.path.join(results_path,"Depth Evaluation via Range Average.png"))
# img_results_folders=[smoke_image_stream_path]
# video_writer(images_folder_path=smoke_image_stream_path,video_filename="2D Detections Results Video.mp4",fps=10,results_path=results_path)
# video_writer(images_folder_path=smoke_3D_image_stream_path,video_filename="3D Detections Results Video.mp4",fps=10,results_path=results_path)
# video_writer(images_folder_path=smoke_3D_birdview_stream_path,video_filename="3D Birdview Results Video.mp4",fps=10,results_path=results_path)
# video_writer(images_folder_path=smoke_3D_overview_stream_path,video_filename="3D Overview Results Video.mp4",fps=10,results_path=results_path)


# Video Writer   