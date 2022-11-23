
from logging import root
from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess
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
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
from evaluation_toolbox import smoke_get_n_classes,yolo_get_n_classes,Point,get_IoU,get_evaluation_metrics,plot_groundtruth
from metrics_functions_from_evaluation_script import metrics_evaluator







stream_id=1
session_datetime=str(datetime.now())  

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/KITTI_MOD_fixed/training/boxes')
results_path=os.path.join(root_dir,'results',foldername)
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))
logs_path=os.path.join(results_path,'logs')

# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
# Create Folders in which to store YOLO and SMOKE Frames
os.mkdir(yolo_image_stream_path)
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


# Video Reader | Test Videos as Input

vid = cv2.VideoCapture(str(stream_id)+'.mp4')

# YOLO Video Writer for Demos and Logs
codec = cv2.VideoWriter_fourcc(*'XVID')

# If you want to playback video at normal speed, uncomment line below
#vid_fps =int(vid.get(cv2.CAP_PROP_FPS))

# Get Vid width,height from Video Reader
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# YOLO and SMOKE Video Writers
yolo_vid_writer = cv2.VideoWriter(os.path.join(results_path,'stream'+str(stream_id)+'_yolo.mp4'), codec, 10, (vid_width, vid_height))
#smoke_vid_writer= cv2.VideoWriter('processed_videos/stream'+str(stream_id)+'_smoke.mp4', codec, 10, (vid_width, vid_height))

# Log Files
yolo_log_filename=os.path.join(logs_path,'record_detections_stream'+str(stream_id)+'YOLO.txt')
smoke_log_filename=os.path.join(logs_path,'record_detections_stream'+str(stream_id)+'SMOKE.txt')

yolo_log=open(yolo_log_filename, 'w') 
smoke_log=open(smoke_log_filename,'w') 


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
    

# ret,frame=vid.read()
# while ret==True:
#     pilimage=Image.fromarray(frame)
#     img_tensor,target,P2=preprocess(network_configuration,fileid,pilimage)
#     tuple_target=(target,)
#     with torch.no_grad():
#         model.eval()
#         output=model.forward(img_tensor[0].to(gpu_device),targets=tuple_target)
#         output = output.to(cpu_device)

#     # SMOKE Output
#     smoke_predictions_list=output.tolist()
#     print('SMOKE Predictions List: ',smoke_predictions_list)
#     smoke_detections_n=len(smoke_predictions_list)
#     smoke_n_classes=smoke_get_n_classes(smoke_predictions_list)

#     boxs_groundtruth_file='2011_09_26_drive_'+str(stream_id).zfill(4)+'_sync_'+str(fileid).zfill(10)+'.txt'
#     with open(os.path.join(boxs_groundtruth_path,boxs_groundtruth_file),'r') as file:
#         boxs_groundtruth_string=file.read()
    
    

#     groundtruth_list=boxs_groundtruth_string.split()
#     groundtruth = [groundtruth_list[x:x+6] for x in range(0,int(len(groundtruth_list)),6)]

#     groundtruth_image=plot_groundtruth(frame,groundtruth)
#     cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid)+'.png'),groundtruth_image)

#     print('Ground Truth: ',groundtruth)

#     TP=get_evaluation_metrics(groundtruth,smoke_predictions_list)
#     # for i in range(len(groundtruth)):
#     #     true_pt1=Point(float(groundtruth[i][3]),float(groundtruth[i][4]))
#     #     true_class=groundtruth[i][1]

#     #     for j in range(smoke_detections_n):

#     #         pred_pt1=Point(smoke_predictions_list[j][2],smoke_predictions_list[j][3])
#     #         pred_pt2=Point(smoke_predictions_list[j][4],smoke_predictions_list[j][5])
#     #         pred_class_ID=smoke_predictions_list[j][0]
#     #         pred_class_string=get_key(pred_class_ID,TYPE_ID_CONVERSION)

#     #         print('Pred',(pred_pt1,pred_pt2))
#     #         print('Truth', (true_pt1,true_pt2))
#     #         IoU=get_IoU((pred_pt1,pred_pt2),(true_pt1,true_pt2))

#     #         if true_class=='Car':

#     #             if IoU>=0.7 and pred_class_string==true_class:
#     #                 print('True Positive Found')
#     #                 TP=TP+1

#     #             else:
#     #                 pass
            
#     #         elif true_class=='Cyclist':
#     #             if IoU>=0.5 and pred_class_string==true_class:
#     #                 print('True Positive Found')
#     #                 TP=TP+1

#     #             else:
#     #                 pass

#     #         elif true_class=='Pedestrian':
#     #             if IoU>=0.5 and pred_class_string==true_class:
#     #                 print('True Positive Found')
#     #                 TP=TP+1

#     #             else:
#     #                 pass

#     #         else:
#     #             pass
            



#     smoke_log.write(str(fileid)+' '+str(smoke_detections_n)+' '+str(smoke_n_classes[0])+' '+str(smoke_n_classes[1])+' '+str(smoke_n_classes[2])+' '+str(TP))
#     smoke_log.write('\n')

#     #YOLO Output
#     # boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
#     # YOLO_output=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
#     # print('Classes: ',YOLO_output)

#     birdeye_view_shape=(100,100)
#     fig,boundingbox_image,birdview_image=visualize(birdeye_view_shape,smoke_predictions_list,P2,pilimage,tracker)
    
#     #plt.savefig('saved_image.png')

#     fig.savefig(os.path.join(smoke_image_stream_path,'frame'+str(fileid)+'.png'))



    
#     fileid+=1
#     print('fileid: ',fileid)
#     print('SMOKE')
#     ret,frame=vid.read()

# smoke_log.close()

# Hide GPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

fileid=0

vid = cv2.VideoCapture('test_videos/'+str(stream_id)+'.mp4')
ret,frame=vid.read()
while ret==True:
    #YOLO Output
    print('frame: ',fileid)
    boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
    print('YOLO Classes: ',classes)
    YOLO_output=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
    print('YOLO Output: ',YOLO_output)
    yolo_n_classes=yolo_get_n_classes(YOLO_output)
    yolo_log.write(str(fileid)+' '+str(yolo_n_classes[0])+' '+str(yolo_n_classes[1])+' '+str(yolo_n_classes[2])+' '+str(yolo_n_classes[3]))
    yolo_log.write('\n')
    img=yolo_visualize(tracker,frame,fileid)
    yolo_vid_writer.write(img)
    cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),img)


    fileid+=1
    print('fileid: ',fileid)
    print('YOLO')
    ret,frame=vid.read()

yolo_log.close()


#####################################################################################################################

with open(smoke_log_filename,'r') as file:
    smoke_file=file.read()


SMOKE_list=smoke_file.split()
SMOKE = [SMOKE_list[x:x+5] for x in range(0,int(len(SMOKE_list)),5)]


with open(yolo_log_filename,'r') as file:
    yolo_file=file.read()


YOLO_list=yolo_file.split()
YOLO = [YOLO_list[x:x+5] for x in range(0,int(len(YOLO_list)),5)]

smoke_n_detections_yaxis=[float(list_elements[1]) for list_elements in SMOKE ]
yolo_n_detections_yaxis=[float(list_elements[1]) for list_elements in YOLO ]
print('n elements: ',len(SMOKE))



evaluation_fig=plt.figure(1)
fig, axs = plt.subplots(3,figsize=(20.0, 10.0))
fig.suptitle('Vertically stacked subplots')

axs[0].bar(range(len(smoke_n_detections_yaxis)),smoke_n_detections_yaxis,label='SMOKE')
axs[0].set_xticks(range(0,len(smoke_n_detections_yaxis),4))
axs[0].set_ylabel('# SMOKE Detections')
axs[0].grid(True)
axs[0].set_ylim([0,10])

# axs[0].set_xticks
# axs[0].set_xticklabels(rotation = (45), fontsize = 10)
axs[1].bar(range(len(smoke_n_detections_yaxis)),yolo_n_detections_yaxis,label='YOLOV3')
axs[1].set_xticks(range(0,len(smoke_n_detections_yaxis),4))
axs[1].set_ylabel('# YOLOV3 Detections')
axs[1].set_ylim([0,10])
axs[1].grid(True)

error=np.array(smoke_n_detections_yaxis)-np.array(yolo_n_detections_yaxis)

axs[2].bar(range(len(smoke_n_detections_yaxis)),np.abs(error),label='ERROR')
axs[2].set_xticks(range(0,len(smoke_n_detections_yaxis),4))
axs[2].set_ylabel('# Detections Error')
axs[2].grid(True)


#plt.legend(loc='lower rightq')
# plt.ylim((0 ,10))
# plt.xlim((0,110))
# figManager = fig.get_current_fig_manager()
# figManager.window.showMaximized()
fig.savefig(os.path.join(results_path,'comparison.png'))



