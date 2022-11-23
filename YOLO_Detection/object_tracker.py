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


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
#print(class_names)
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

stream_id=59
vid = cv2.VideoCapture('stream'+str(stream_id)+'.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('./data/video/stream'+str(stream_id)+'_yolo.mp4', codec, 10, (vid_width, vid_height))


from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

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

    img_in = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()
    print('--------------------------------------------------')
    boxes, scores, classes, nums = yolo.predict(img_in,steps=1)
    #print('scores[0]: ',scores[0])
    print('# of items in scores[0]: ',len(scores[0]))
    #print('scores[0][0]: ',scores[0][0])
    #print('boxes[0]: ',boxes[0])
    print('# of items in boxes[0]: ',len(boxes[0]))
    #print('boxes: ',boxes)
    

    # print('Scores: ',scores)
    # print('Classes: ',classes)
    # print('Numbers: ',nums)
    #

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(cv2_img, boxes[0])

    print('Converted boxes: ',converted_boxes)
    print('# of items in converted_boxes: ',len(converted_boxes))
    #print('# of items in converted_boxes[0]: ',len(converted_boxes[0]))
    features = encoder(cv2_img, converted_boxes)
    print('type of scores[0]',type(scores[0]))
    print('type of names ',type(names))
    print('type of converted_boxes',type(converted_boxes))


    # Current Obstacle
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]


    print('zip(converted_boxes,scores[0],names,features): ',zip(converted_boxes, scores[0], names, features))

    for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features):
        print('bbox: ',bbox)
        print('score: ',score)
        print('class_name: ',class_name)
        print('feature: ',feature)


    

    print('Type of detections: ',type(detections))
    print('detections: ',detections)
    print('n detections initial: ',len(detections))

    # for d in detections:
    #     print('d in detections: ',d)
    #     print('d.twlh in detections: ',d.tlwh)
    #     print('d.confidence in detections: ',d.confidence)


    boxs = np.array([d.tlwh for d in detections])
    # print('boxs: ',boxs)
    # print('boxs[0]: ',boxs[0])

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


    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    print('indices: ',indices)
    detections = [detections[i] for i in indices]
    print(' length Input Detections: ',len(detections))
    print('Type of detections input: ',type(detections))

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]



    current_count = int(0)
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        #print('Class',class_name)
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(cv2_img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(cv2_img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(cv2_img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(cv2_img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = cv2_img.shape
        cv2.line(cv2_img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
        cv2.line(cv2_img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)

        center_y = int(((bbox[1])+(bbox[3]))/2)

        if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
            if class_name == 'car' or class_name == 'truck':
                counter.append(int(track.track_id))
                current_count += 1

    total_count = len(set(counter))
    cv2.putText(cv2_img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(cv2_img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

    delta_T=time.time()-t1
    fps = 1./(time.time()-t1)
    fps_history.append(fps)
    current_T=current_T+delta_T
    timestamps.append(current_T)
    cv2.putText(cv2_img, "Frame ID:"+str(frame_id), (0,30), 0, 1, (0,0,255), 2)
    #cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', cv2_img)
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