from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
import matplotlib.pyplot as plt
import cv2
import torch
torch.cuda.empty_cache()
import os
from datetime import datetime
from EvaluatorClass2 import metrics_evaluator,plot_groundtruth
from metrics_functions_from_evaluation_script import smoke_get_n_classes,plot_prediction,read_groundtruth,get_key,read_prediction,write_prediction,convert_prediction_text_format,get_class_AP,construct_dataframe_v2
import subprocess

import dataframe_image as dfi

visualize_3D=False
predict_then_evaluate=True
only_predict=False
only_evaluate=False

folder_path_for_evaluation='StreamSMOKE_metrics_eval2022_12_20_14_07_07'







stream_id='SMOKE_metrics_eval'
session_datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hashot51/Projects/perception-validation-verification'
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
print('number of files: ',number_files)


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

print('Setup Network Args: ',args)
model,cfg,gpu_device,cpu_device=setup_network(args)



checkpointer = DetectronCheckpointer(
    cfg, model, save_dir=cfg.OUTPUT_DIR
)
ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)





n=10
smoke_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
#fileid=0
#"/home/hashot51/Projects/perception-validation-verification/camera_calibration/ComputerVision/SMOKE_test.mp41.png"
#"/home/hashot51/Projects/perception-validation-verification/camera_calibration/ComputerVision/test_images/output.mp41.png"
for fileid in range(n):#filepath in glob.glob(os.path.join(test_images_path,'*.png')):

    #fileid=i
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    # cv2.imshow('Input Frame: ',frame)
    # cv2.waitKey(0)
    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),groundtruth_image)

    if predict_then_evaluate==True or only_predict==True:


        #pilimage=Image.fromarray(frame)#.convert('BGR')
        smoke_predictions_list,P2,K=preprocess_then_predict(model,cfg,fileid,ordered_filepath,gpu_device,cpu_device)
        #print('Length ------------------------: ',len(smoke_predictions_list[0]))
        write_prediction(data_path,fileid,smoke_predictions_list)




        output_img=plot_prediction(frame,smoke_predictions_list)
        print('P2: ',P2)
        b,g,r=cv2.split(frame)
        rgb_frame=cv2.merge([r,g,b])
        #fig,img,birdimage=visualize((900,900),smoke_predictions_list,K,rgb_frame)
        #plt.show()
        # cv2.imshow('3D Box Img',img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(smoke_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),output_img)


    
    


    if only_evaluate==True or predict_then_evaluate==True:

        smoke_predictions_read_from_file=read_prediction(data_path,fileid)
        car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_object_classes,n_object_difficulties=smoke_metrics_evaluator.evaluate_metrics(groundtruth,smoke_predictions_read_from_file)



            


torch.cuda.empty_cache()

precision_evaluation_path='SMOKE/smoke/data/datasets/evaluation/kitti/kitti_eval_40/evaluate_object_3d_offline'
boxs_groundtruth_path='/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/label_2'





command = "./{} {} {} ".format(precision_evaluation_path,boxs_groundtruth_path, results_path)
#print(command)

if only_evaluate==True or predict_then_evaluate==True:
    average_precision_command=subprocess.check_output(command, shell=True, universal_newlines=True).strip()

    print(average_precision_command)



    cars_easy_AP,cars_moderate_AP,cars_hard_AP=get_class_AP(results_path,'Car')
    pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=get_class_AP(results_path,'Pedestrian')


    cars_AP=[cars_easy_AP,cars_moderate_AP,cars_hard_AP]
    pedestrians_AP=[pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP]

    df,bar_metrics=construct_dataframe_v2(cars_AP,pedestrians_AP,car_metrics,pedestrian_metrics,difficulty_metrics,n_object_classes,n_object_difficulties)

    dfi.export(df,os.path.join(results_path,'MetricsTable.png'))


    metrics_img=cv2.imread(os.path.join(results_path,'MetricsTable.png'))
    cv2.imshow('Metrics',metrics_img)
    cv2.waitKey(0)

    bar_metrics.iloc[:,0:3].plot(kind='bar',title="SMOKE AP Evaluation ",figsize=(20, 8))
    plt.legend(loc=(-0.16,0.7))
    plt.xlabel("Metrics")
    plt.ylabel("Percentage %")
    plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
    plt.yticks(range(0,105,5))
    plt.savefig(os.path.join(results_path,"bar_metrics_AP.png"),dpi=600,bbox_inches="tight")
    plt.grid(True)
    plt.show()

    bar_metrics.iloc[:,3:6].plot(kind='bar',title="SMOKE Precision Evaluation",figsize=(20, 8))
    plt.legend(loc=(-0.16,0.7))
    plt.xlabel("Metrics")
    plt.ylabel("Percentage %")
    plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
    plt.yticks(range(0,105,5))
    plt.savefig(os.path.join(results_path,"bar_metrics_precision.png"),dpi=600,bbox_inches="tight")
    plt.grid(True)
    plt.show()

    bar_metrics.iloc[:,6:9].plot(kind='bar',title="SMOKE Recall Evaluation",figsize=(20, 8))
    plt.legend(loc=(-0.16,0.7))
    plt.xlabel("Metrics")
    plt.ylabel("Percentage %")
    plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
    plt.yticks(range(0,105,5))
    plt.savefig(os.path.join(results_path,"bar_metrics_recall.png"),dpi=600,bbox_inches="tight")
    plt.grid(True)
    plt.show()

    