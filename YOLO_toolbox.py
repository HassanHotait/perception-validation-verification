from YOLO_Detection.yolov3_tf2.models import YoloV3
from YOLO_Detection.tools import generate_detections as gdet
from YOLO_Detection.deep_sort import nn_matching
from YOLO_Detection.deep_sort.tracker import Tracker
import cv2
import tensorflow as tf
from YOLO_Detection.yolov3_tf2.dataset import transform_images
import numpy as np
from YOLO_Detection.deep_sort.detection import Detection
from YOLO_Detection.yolov3_tf2.utils import convert_boxes
from _collections import deque
import time
from YOLO_Detection.deep_sort import preprocessing
import matplotlib.pyplot as plt

def setup_YOLO(path_to_class_names,path_to_weights):
    class_names = [c.strip() for c in open(path_to_class_names).readlines()]
    #print(class_names)
    yolo = YoloV3(classes=len(class_names))
    #yolo.load_weights('./weights/yolov3.tf')
    yolo.load_weights(path_to_weights)

    return yolo,class_names

def setup_tracker(model_filename,max_cosine_distance=0.5,nn_budget=None,nms_max_overlap=0.8):
    # max_cosine_distance = 0.5
    # nn_budget = None
    # nms_max_overlap = 0.8

    #model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    return tracker,encoder

def preprocess_predict_YOLO(yolo,input_img):
    img_in = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)
    boxes, scores, classes, nums = yolo.predict(img_in,steps=1)
    return boxes,scores,classes,nums

def postprocess_YOLO(encoder,tracker,class_names,input_img,boxes,scores,classes):
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(input_img, boxes[0])


    features = encoder(input_img, converted_boxes)



    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]



    boxs = np.array([d.tlwh for d in detections])


    scores = np.array([d.confidence for d in detections])

    print('scores: ',scores)
    classes = np.array([d.class_name for d in detections])

    nms_max_overlap=0.8
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)

    detections = [detections[i] for i in indices]

    boxs=[boxs[i] for i in indices]
    classes=[classes[i] for i in indices]
    scores=[scores[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    # #output=[track.get_class() for track in tracker.tracks if track.is_confirmed()]
    # output=[]

    # for track in tracker.tracks:
    #     if not track.is_confirmed() or track.time_since_update >1:
    #         continue
    #     else:
    #         output.append(track.get_class())


    return boxs,classes,scores,tracker

    

def yolo_visualize(tracker,cv2_img,frame_id,timestamp):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    pts = [deque(maxlen=30) for _ in range(1000)]
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
                pass
                #counter.append(int(track.track_id))
                #current_count += 1

    #total_count = len(set(counter))
    # cv2.putText(cv2_img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    # cv2.putText(cv2_img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

    # delta_T=time.time()-t1
    # fps = 1./(time.time()-t1)
    #fps_history.append(fps)
    #current_T=current_T+delta_T
    #timestamps.append(current_T)
    cv2.putText(cv2_img, "Frame ID:"+str(frame_id), (0,30), 0, 1, (0,0,255), 2)
    cv2.putText(cv2_img, "Timestamp: "+str('{0:.2f}'.format(timestamp)), (0,60), 0, 1, (0,0,255), 2)
    #cv2.resizeWindow('output', 1024, 768)
    return cv2_img