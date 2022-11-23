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
from metrics_functions_from_evaluation_script import (read_groundtruth,plot_groundtruth,plot_prediction,
                                                        get_gt_classes_boxes,get_pred_classes_boxes,
                                                        get_metrics_label_specific,tabularize)

from metrics_functions_from_evaluation_script import metrics_evaluator

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
test_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples')
gt_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples/GT images')
pred_samples_folder=os.path.join(root_dir,'unit_test_figs/test_samples/Pred Images')
logs_folder=os.path.join(root_dir,'unit_test_figs/test_samples/logs')

#results_path=os.path.join(root_dir,'results',foldername)

#fileid=6

frames_of_interest=[2,3,5,10,16,19,21,26,27,37,40,42,43,48,51,60,62,68,73,76]
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
    print('Ordered Filepath: ',ordered_filepath)
    frame=cv2.imread(ordered_filepath)

    pilimage=Image.fromarray(frame)
    smoke_predictions_list=preprocess_then_predict(model,network_configuration,i,pilimage,gpu_device,cpu_device)

    print('Predictions: ',smoke_predictions_list)

    groundtruth=read_groundtruth(test_samples_folder,i)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(gt_samples_folder,'frame'+str(i)+'.png'),groundtruth_image)
    output_img=plot_prediction(groundtruth_image,smoke_predictions_list)
    cv2.imwrite(os.path.join(pred_samples_folder,'frame'+str(i)+'.png'),output_img)

    MetricEvaluator.evaluate_metrics(groundtruth,smoke_predictions_list)

    #print('Ground Truth: ',groundtruth)

    # gt_classes,gt_boxs=get_gt_classes_boxes(groundtruth)
    # pred_classes,pred_boxs=get_pred_classes_boxes(smoke_predictions_list)
    

    # print('Gt Classes: ',len(gt_classes))
    # print('Gt Boxs: ',len(gt_boxs))
    # print('Pred Classes: ',len(pred_classes))
    # print('Pred Boxs: ',len(pred_boxs))

    # # TP=get_evaluation_metrics(groundtruth,smoke_predictions_list)
    # labels=['Car','Cyclist','Pedestrian']
    # classA1_gt_instances,classA1_pred_instances,TP_A1,FP_A1,FN_A1=get_metrics_label_specific('Car','Easy',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)
    # classA2_gt_instances,classA2_pred_instances,TP_A2,FP_A2,FN_A2=get_metrics_label_specific('Car','Moderate',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)
    # TP_easy.append(TP_A1)
    # FP_easy.append(FP_A1)
    # FN_easy.append(FN_A1)
    # TP_moderate.append(TP_A2)
    # FP_moderate.append(FP_A2)
    # FN_moderate.append(FN_A2)
    #fileid=fileid+1

# tabularize(TP_easy,FP_easy,FN_easy,'Cars Easy')
# tabularize(TP_moderate,FP_moderate,FN_moderate,'Cars Moderate')
torch.cuda.empty_cache()