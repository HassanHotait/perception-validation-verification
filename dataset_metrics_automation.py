
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
from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,smoke_get_n_classes,yolo_get_n_classes,read_groundtruth
from metrics_functions_from_evaluation_script import metrics_evaluator
import glob







stream_id='kitti_training_set_smoke_metrics'
session_datetime=str(datetime.now())  

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')
results_path=os.path.join(root_dir,'results',foldername)
#yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
logs_path=os.path.join(results_path,'logs')

lst = os.listdir(test_images_path) # your directory path
number_files = len(lst)

# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
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

n=10
smoke_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0
for i in range(n):#filepath in glob.glob(os.path.join(test_images_path,'*.png')):
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    pilimage=Image.fromarray(frame)
    smoke_predictions_list=preprocess_then_predict(model,network_configuration,fileid,pilimage,gpu_device,cpu_device)
    print('SMOKE Predictions List: ',smoke_predictions_list)
    smoke_detections_n=len(smoke_predictions_list)
    smoke_n_classes=smoke_get_n_classes(smoke_predictions_list)

    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),groundtruth_image)
    output_img=plot_prediction(groundtruth_image,smoke_predictions_list)
    cv2.imwrite(os.path.join(smoke_image_stream_path,'frame'+str(fileid).zfill(6)+'.png'),output_img)

    
    

    smoke_metrics_evaluator.evaluate_metrics(groundtruth,smoke_predictions_list)

    print('Ground Truth: ',groundtruth)

            
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



torch.cuda.empty_cache()

