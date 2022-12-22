


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

import SMOKE.smoke.engine.inference.compute_on_dataset2






root_dir='/home/hashot51/Projects/perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')


    





# Setup SMOKE

args = default_argument_parser().parse_args()
model,network_configuration,gpu_device,cpu_device=setup_network(args)

# Load Weights
checkpointer = DetectronCheckpointer(
    cfg, model, save_dir=cfg.OUTPUT_DIR
)
# ckpt=cfg.MODEL.WEIGHT
# _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


print('Model Weights: ',ckpt)







n=1
for i in [8]:#range(n):#filepath in glob.glob(os.path.join(test_images_path,'*.png')):

    ordered_filepath=os.path.join(test_images_path,str(i).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    copy_frame=cv2.imread(ordered_filepath)
    pilimage=Image.fromarray(copy_frame)
    groundtruth=read_groundtruth(boxs_groundtruth_path,i)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    # cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),groundtruth_image)



    smoke_predictions_list=preprocess_then_predict(model,network_configuration,i,pilimage,gpu_device,cpu_device)

    predictions=compute_on_dataset2(model,copy_frame,network_configuration)
    print('Predictions V2 #: ',len(predictions))
    print('Predictions #: ',len(smoke_predictions_list))

    # img_tensor,target,P2=preprocess(network_configuration,fileid,pilimage)
    # tuple_target=(target,)
    # with torch.no_grad():
    #     model.eval()
    #     output=model.forward(img_tensor[0].to(gpu_device),targets=tuple_target)
    #     output = output.to(cpu_device)

    # smoke_predictions_list=output.tolist()
    #print('Length ------------------------: ',len(smoke_predictions_list[0]))





    output_img=plot_prediction(groundtruth_image,smoke_predictions_list)

    cv2.imshow('Prediction: ',output_img)
    cv2.waitKey(0)





