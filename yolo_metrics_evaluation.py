
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
import csv
import numpy as np
torch.cuda.empty_cache()
import os
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,metrics_evaluator,read_groundtruth,yolo_2_smoke_output_format




def get_key(val):
    TYPE_ID_CONVERSION = {
    'Vehicle': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }
    for key, value in TYPE_ID_CONVERSION.items():
        if val == value:
            return key
 
    return "DontCare"


stream_id='kitti_trainingset_Yolo_metrics'
session_datetime=str(datetime.now())  

foldername='Stream'+str(stream_id)+session_datetime
print('Foldername: ',foldername)

root_dir='/home/hasan//perception-validation-verification'
boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
test_images_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/image_2')
results_path=os.path.join(root_dir,'results',foldername)
yolo_image_stream_path=os.path.join(results_path,'yolo-image-stream'+str(stream_id))
#smoke_image_stream_path=os.path.join(results_path,'smoke-image-stream'+str(stream_id))
groundtruth_image_stream_path=os.path.join(results_path,'groundtruth-image-stream'+str(stream_id))

logs_path=os.path.join(results_path,'logs')



# Create Folder with Datetime to store results of stream
os.mkdir(results_path)
# Create Folders in which to store YOLO and SMOKE Frames
os.mkdir(yolo_image_stream_path)
#os.mkdir(smoke_image_stream_path)
# Create foler in which to store text file logs
os.mkdir(logs_path)
os.mkdir(groundtruth_image_stream_path)




# # open the file in the write mode
# f = open(os.path.join(logs_path,'mobileeye_yolo_detections.csv'), 'w')
# # create the csv writer
# writer = csv.writer(f)
# header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
# writer.writerow(header)

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
    

# Hide GPU 
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

fileid=0

n=7479
yolo_metrics_evaluator=metrics_evaluator(n,logs_path)
#smoke_metrics_evaluator.results_path=logs_path
print('LOGS Path: ',logs_path)
fileid=0
# vid = cv2.VideoCapture('test_videos/'+str(1)+'.mp4')
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# fps    = vid.get(cv2.CAP_PROP_FPS)
# ret,frame=vid.read()
for i in range(n):
    ordered_filepath=os.path.join(test_images_path,str(fileid).zfill(6)+'.png')
    frame=cv2.imread(ordered_filepath)
    print('frame: ',fileid)
    boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,frame)
    print('Boxes Before: ',boxes)
    print('Classes Before: ',classes)
    boxs,classes,tracker=postprocess_YOLO(encoder,tracker,class_names,frame,boxes,scores,classes)
    print('Boxes after NMS: ',boxs)
    print('Classes after NMS: ',classes)
    yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes)
    print('Yolo Predictions List: ',yolo_predictions_list)
    # for i,prediction in enumerate(yolo_predictions_list):
    #     if prediction[0]==0:
    #         row=[fileid,(fileid/fps),tracker.tracks[i].track_id,prediction[2],prediction[3],prediction[4],prediction[5]]
    #         writer.writerow(row)

    # for track in tracker.tracks:
    #     if track.get_class()=='car' or track.get_class()=='bus' or track.get_class()=='truck':
    #         row=[fileid,(fileid/fps),track.get_class(),track.track_id,track.to_tlbr()[0],track.to_tlbr()[1],track.to_tlbr()[2],track.to_tlbr()[3]]
    #         writer.writerow(row)

    #print('tracker ',tracker.tracks[0].to_tlbr())
    # tracks_list=[track.track_id for track in tracker.tracks if track.is_confirmed() and track.time_since_update <1]
    # print(tracks_list)
    # print('# of objects predicted: ',len(yolo_predictions_list))
    # print('# of objects tracker',len(tracks_list))
    # print('# of objects in active targets',len(tracker.active_targets))
    # print('Check: ',len(yolo_predictions_list)==len(tracker.active_targets))
    #yolo_n_classes=yolo_get_n_classes(YOLO_output)
    # yolo_log.write(str(fileid)+' '+str(yolo_n_classes[0])+' '+str(yolo_n_classes[1])+' '+str(yolo_n_classes[2])+' '+str(yolo_n_classes[3]))
    # yolo_log.write('\n')
    groundtruth=read_groundtruth(boxs_groundtruth_path,fileid)
    groundtruth_image=plot_groundtruth(frame,groundtruth)
    cv2.imwrite(os.path.join(groundtruth_image_stream_path,'frame'+str(fileid)+'.png'),groundtruth_image)


    output_img=plot_prediction(frame,yolo_predictions_list)
    #yolo_vid_writer.write(img)
    cv2.imwrite(os.path.join(yolo_image_stream_path,'frame'+str(fileid)+'.png'),output_img)



    yolo_metrics_evaluator.evaluate_metrics(groundtruth,yolo_predictions_list)




    fileid+=1
    print('fileid: ',fileid)
    print('YOLO')
    # ret,frame=vid.read()










