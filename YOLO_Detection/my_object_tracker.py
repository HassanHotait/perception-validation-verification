from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

#from numba import jit
import os
import glob
# USES CPU 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3,YoloV3Tiny
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
from YOLO_toolbox import setup_YOLO,setup_tracker,preprocess_predict_YOLO,yolo_visualize




yolo,class_names=setup_YOLO(path_to_class_names='./data/labels/coco.names',path_to_weights='./weights/yolov3.tf')


model_filename = 'model_data/mars-small128.pb'
tracker,encoder=setup_tracker(model_filename)



stream_id=2

vid = cv2.VideoCapture('/home/hasan/perception-validation-verification/test_videos/2.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
#vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./data/video/stream'+str(stream_id)+'_yolo.mp4', codec, 10, (vid_width, vid_height))


counter = []

fps_history=[]
timestamps=[]
current_T=0
frame_id=0
f=open('./data/logs/record_detections_stream'+str(stream_id)+'YOLO.txt', 'w') 

test_dir='/home/hasan/SMOKE/datasets/KITTI_MOD_fixed/training/images/'

while True:
    #print('file-----------------',filepath)
        
    IMAGE=str(frame_id).zfill(10)+'.png'
    t1 = time.time()

    results_dict = {}
    
    # # Read a PIL image
    # filename=test_dir+'2011_09_26_drive_0001_sync_'+IMAGE
    # # image_filename=str(fileid).zfill(6)+".png"
    # # path_to_image=image_dir+image_filename
    # img = Image.open(filename).convert('RGB')
    # cv2_img=cv2.imread(filename)
    # filename=filename.split('/')[-1]
    _, cv2_img = vid.read()
    if cv2_img is None:
        print('Completed')
        break

    boxes, scores, classes, nums=preprocess_predict_YOLO(yolo,cv2_img)
    # img_in = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # img_in = tf.expand_dims(img_in, 0)
    # img_in = transform_images(img_in, 416)

    t1 = time.time()
    print('--------------------------------------------------')

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(cv2_img, boxes[0])
    features = encoder(cv2_img, converted_boxes)
    # Current Obstacle
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]



    boxs = np.array([d.tlwh for d in detections])

    scores = np.array([d.confidence for d in detections])

    print('scores: ',scores)
    classes = np.array([d.class_name for d in detections])
    print('Classes: ',classes)
    IMAGE=str(frame_id).zfill(6)
    n_cars=0
    n_bikes=0
    n_pedestrians=0

    for i in classes:
        if i=='car' or i=='truck':
            n_cars+=1

        elif i=='bicycle' or i=='motorbike':
            n_bikes+=1

        elif i=='person':
            n_pedestrians+=1
        else:
            pass


    f.write('2011_09_26_drive_0001_sync_'+IMAGE+' '+str(len(classes))+' '+str(n_cars)+' '+str(n_pedestrians)+' '+str(n_bikes))
    f.write('\n')

    nms_max_overlap=0.8
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    print('indices: ',indices)
    detections = [detections[i] for i in indices]
    print(' length Input Detections: ',len(detections))
    print()
    print('Type of detections input: ',type(detections))

    tracker.predict()
    tracker.update(detections)
    print('# of tracks: ',len(tracker.tracks))
    print('Check: ',len(tracker.tracks)==len(detections))

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]



    current_count = int(0)

    #visualize(tracker,colors,cv2_img,counter,t1,frame_id,current_count)
    img=yolo_visualize(tracker,cv2_img,frame_id)
    cv2.imshow('output', img)

    out.write(cv2_img)

    if cv2.waitKey(1) == ord('q'):
        break

    frame_id+=1
vid.release()
out.release()
f.close()
cv2.destroyAllWindows()

print('Timestamps: ',timestamps)
print('FPS: ',fps_history)



plt.plot(timestamps,fps_history)
plt.xlabel('Timestamps [s]')
plt.ylabel('FPS')
plt.show()