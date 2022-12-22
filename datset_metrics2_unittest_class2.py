from doctest import OutputChecker
from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
from SMOKE.tools.box import visualize
import os
import glob
# from evaluation_toolbox import smoke_get_n_classes,yolo_get_n_classes,Point,get_IoU,get_evaluation_metrics,plot_groundtruth,get_key,plot_prediction
# from evaluation_metrics_unittest import get_gt_classes, get_gt_classes_boxes, get_metrics_label_specific, get_pred_classes_boxes,tabularize,read_groundtruth
from PIL import Image
import torch
torch.cuda.empty_cache()
import cv2
from metrics_functions_from_evaluation_script import (read_groundtruth,plot_prediction)

from EvaluatorClass2 import metrics_evaluator,plot_groundtruth



# Setup SMOKE

args = default_argument_parser().parse_args()
model,network_configuration,gpu_device,cpu_device=setup_network(args)

# Load Weights
checkpointer = DetectronCheckpointer(
    cfg, model, save_dir=cfg.OUTPUT_DIR
)
ckpt=cfg.MODEL.WEIGHT
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

root_dir='/home/hasan/perception-validation-verification'
test_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples/frames_of_interest_2')
gt_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples/GT images')
pred_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples/Pred Images')
logs_folder=os.path.join(root_dir,'unit_test_figs/test_samples/logs')

#results_path=os.path.join(root_dir,'results',foldername)

#fileid=6

frames_of_interest=[2,3,5,10,16,19,21,26,27,37,40,42,43,48,51,60,62,68,73,76]
#frames_of_interest=[6,8]
# Edge Cases Examples
frames_of_interest=[1,4,5,8,15,16,18,19,21,23,25,26]
MetricEvaluator=metrics_evaluator(len(frames_of_interest),logs_folder)
#MetricEvaluator.n_frames=4
# TP_easy=[]
# FP_easy=[]
# FN_easy=[]

# TP_moderate=[]
# FP_moderate=[]
# FN_moderate=[]



for i in frames_of_interest:#filepath in glob.glob(os.path.join(test_samples_folder,'*.png')):
    ordered_filepath=os.path.join(test_samples_folder,str(i).zfill(6)+'.png')
    #print('Ordered Filepath: ',ordered_filepath)
    frame=cv2.imread(ordered_filepath)

    # pilimage=Image.fromarray(frame)
    smoke_predictions_list=preprocess_then_predict(model,network_configuration,i,ordered_filepath,gpu_device,cpu_device)

    #print('Predictions: ',smoke_predictions_list)

    groundtruth=read_groundtruth(test_samples_folder,i)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(gt_samples_folder,'frame'+str(i)+'.png'),groundtruth_image)
    output_img=plot_prediction(groundtruth_image,smoke_predictions_list)
    cv2.imwrite(os.path.join(pred_samples_folder,'frame'+str(i)+'.png'),output_img)

    MetricEvaluator.evaluate_metrics(groundtruth,smoke_predictions_list)


torch.cuda.empty_cache()

