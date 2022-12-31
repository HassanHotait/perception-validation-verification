#================================================================
#
#   File name   : evaluate_mAP.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to evaluate model mAP and FPS
#
#================================================================
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import read_class_names
from yolov3.configs import *
import shutil
import json
import time
from convert_predictions_to_json import convert_predictions

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                                mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def get_gt(dataset,ground_truth_dir_path,n_frames):
    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
    NUM_CLASS = read_class_names(TRAIN_CLASSES)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

    # if not os.path.exists('mAP'): os.mkdir('mAP')
    os.mkdir(ground_truth_dir_path)

    #print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')

    gt_counter_per_class = {}
    #print("dataset num samples: ",dataset.num_samples)
    for index in range(n_frames):
        ann_dataset = dataset.annotations[index]

        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
        # print("original image: \n",original_image)

        # cv2.imshow("original image",original_image)
        # cv2.waitKey(0)

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        for i in range(num_bbox_gt):
            class_name = NUM_CLASS[classes_gt[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

            # if class_name=="donut":
            #     print("donut in groundtruth: ",index)

            # count that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)


def get_mAP(ground_truth_dir_path,predictions_dir_path):


    # Ground Truth Stats

    #ground_truth_dir_path="C:\\Users\\hashot51\\Desktop\\yolo_benchmark_checkpoint\\mAP\\ground-truth"
    #predictions_dir_path="C:\\Users\\hashot51\\Desktop\\TensorFlow-2.x-YOLOv3\\converted_predictions"

    gt_counter_per_class = {}

    # count that object

    groundtruth_filenames=os.listdir(ground_truth_dir_path)

    print("groundtruth filenames: ",groundtruth_filenames)

    for filename in groundtruth_filenames:
        with open(os.path.join(ground_truth_dir_path,filename),"r") as f:
            json_file_contents=json.load(f)

        for gt in json_file_contents:

            class_name=gt["class_name"]

            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1


    gt_classes = list(gt_counter_per_class.keys())
    # sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    # print("gt_classes: ",gt_classes)
    # print("n_classes: ",n_classes)


    #################################################################################

    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)
    #NUM_CLASS = read_class_names(TRAIN_CLASSES)

    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    
    with open(os.path.join("C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\COCO_Evaluation","results.txt"), 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_filename = f'{class_name}_predictions.json'
            predictions_filepath=os.path.join(predictions_dir_path,predictions_filename)

            if os.path.exists(predictions_filepath):
                predictions_data = json.load(open(predictions_filepath))
            else:
                with open(predictions_filepath,"w") as predictions_file:
                    json.dump([],predictions_file)

                with open(predictions_filepath,"r") as predictions_file:
                    predictions_data = json.load(predictions_file)

            
            # Assign predictions to ground truth objects
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["fileid"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [ float(x) for x in prediction["bbox"].split() ] # bounding box of prediction
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        #print("class match")
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:# if ovmax > minimum overlap
                    #print("gt match: ",gt_match["used"])
                    if not bool(gt_match["used"]):
                        
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                        #print("False Positive 1")
                else:
                    # false positive
                    #print("False Positive 2")
                    fp[idx] = 1

            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print("recall: ",rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print("precision: ",prec)

            ap, mrec, mprec = voc_ap(rec, prec)

            print("ap: ",ap)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, 0)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100

if __name__ == '__main__':       
    # if YOLO_FRAMEWORK == "tf": # TensorFlow detection
    #     if YOLO_TYPE == "yolov4":
    #         Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
    #     if YOLO_TYPE == "yolov3":
    #         Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

    #     if YOLO_CUSTOM_WEIGHTS == False:
    #         yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    #         load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
    #     else:
    #         yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    #         yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use custom weights
        
    # elif YOLO_FRAMEWORK == "trt": # TensorRT detection
    #     saved_model_loaded = tf.saved_model.load(f"./checkpoints/{TRAIN_MODEL_NAME}", tags=[tag_constants.SERVING])
    #     signature_keys = list(saved_model_loaded.signatures.keys())
    #     yolo = saved_model_loaded.signatures['serving_default']

    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    convert_predictions(predictions_kitti_format_dir="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\results\\StreamYOLO_benchmark_COCO_@0.05_2022_12_28_01_21_13\\data",
                        predictions_coco_format_dir="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\COCO_Evaluation\\yolo_predictions_json_format",
                        n_frames=4952)
    gt_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\COCO_Evaluation\\coco\\coco_gt"
    get_gt(testset,gt_path,4952)
    get_mAP(ground_truth_dir_path=gt_path,
            predictions_dir_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\COCO_Evaluation\\yolo_predictions_json_format")
