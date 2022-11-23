from tools.deep_sort import preprocessing
from tools.deep_sort import nn_matching
from tools.deep_sort.detection import Detection
from tools.deep_sort.tracker import Tracker
import numpy as np
import tools.generate_detections as gdet


def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int)
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2]-box[0])
        box[3] = int(box[3]-box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0,0,0,0]:
            returned_boxes.append(box)
    return returned_boxes

#classes = classes[0]

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}

# function to return key for any value
def get_key(val,my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def get_deepsort_input(SMOKE_output,img,max_cosine_distance=0.5,nn_budget=None,nms_max_overlap=0.8):
    # max_cosine_distance = 0.5
    # nn_budget = None
    # nms_max_overlap = 0.8
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2 }

    model_filename = 'tools/deep_sort/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    names = []
    # for i in range(len(classes)):
    #     names.append(class_names[int(classes[i])])
    names = np.array(names)
    names=np.array(['Car','Cyclist','Pedestrian'])

    boxes=[]
    scores=[]
    classes=[]

    for i in range(len(SMOKE_output)):
        boxes.append(SMOKE_output[i][2:6])
        scores.append([SMOKE_output[i][13]])
        classes.append(get_key([SMOKE_output[i][0]],TYPE_ID_CONVERSION))

        converted_boxes=boxes
        converted_boxes[i][2]=boxes[i][2]-boxes[i][0]
        converted_boxes[i][3]=boxes[i][3]-boxes[i][1]
    
    scores=np.array(scores)

    features = encoder(img, converted_boxes)

    
    




    #converted_boxes = convert_boxes(img, boxes[0])



    # detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
    #                 zip(converted_boxes, scores[0], names, features)]

    # boxs = np.array([d.tlwh for d in detections])
    # scores = np.array([d.confidence for d in detections])
    # classes = np.array([d.class_name for d in detections])
    # indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    # detections = [detections[i] for i in indices]
    print('scores',scores)
    for score in scores:
        print('score: ',score)
    print('Length of Detections: ',(len(scores)))
    detections=[Detection(bbox,score,class_name,feature)  for bbox, score, class_name, feature in zip(converted_boxes, scores, classes, features)]
    print('Length of detections: ',len(detections))

    tracker.predict()
    tracker.update(detections)

    return tracker