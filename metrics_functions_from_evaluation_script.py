import cv2
from matplotlib.pyplot import table
import numpy as np
import random
import matplotlib.pyplot as plt

#from torch import mode
import csv
import pandas as pd
from evaluation_toolbox import rectanges_overlap,Point,get_IoU,get_key,yolo_get_key
import dataframe_image as dfi
import os
from xlsxwriter.utility import xl_rowcol_to_cell
import  xlsxwriter
field_name_cars=['Easy Car TP',
'Easy Car FP',
'Easy Car FN',
'Easy Car Frame Precision',
'Easy Car Frame Recall',
'Easy Car ΣTP',
'Easy Car ΣFP',
'Easy Car ΣFN',
'Easy Car ΣPrecision',
'Easy Car ΣRecall',
'Easy Car Mean Precision',
'Easy Car Mean Recall',
'Moderate Car TP',
'Moderate Car FP',
'Moderate Car FN',
'Moderate Car Frame Precision',
'Moderate Car Frame Recall',
'Moderate Car ΣTP',
'Moderate Car ΣFP',
'Moderate Car ΣFN',
'Moderate Car ΣPrecision',
'Moderate Car ΣRecall',
'Moderate Car Mean Precision',
'Moderate Car Mean Recall',
'Hard Car TP',
'Hard Car FP',
'Hard Car FN',
'Hard Car Frame Precision',
'Hard Car Frame Recall',
'Hard Car ΣTP',
'Hard Car ΣFP',
'Hard Car ΣFN',
'Hard Car ΣPrecision',
'Hard Car ΣRecall',
'Hard Car Mean Precision',
'Hard Car Mean Recall',
'Overall Car TP',
'Overall Car FP',
'Overall Car FN',
'Overall Car Frame Precision',
'Overall Car Frame Recall',
'Overall Car ΣTP',
'Overall Car ΣFP',
'Overall Car ΣFN',
'Overall Car ΣPrecision',
'Overall Car ΣRecall',
'Overall Car Mean Precision',
'Overall Car Mean Recall',
'Gt Cars',
'Filtered Gt Cars',
'Gt Easy Cars',
'Gt Moderate Cars',
'Gt Hard Cars',
'Gt Ignored Cars',
'Pred Cars',
'Filtered Pred Cars',
'Pred Easy Cars',
'Pred Moderate Cars',
'Pred Hard Cars',
'Pred Ignored Cars',
'ΣGt Cars',
'Σ filtered Gt Cars',
'ΣGt Easy Cars',
'ΣGt Moderate Cars',
'ΣGt Hard Cars',
'ΣPred Cars',
'ΣPred Easy Cars',
'ΣPred Moderate Cars',
'ΣPred Hard Cars',]


field_name_cyclists=['Easy Cyclist TP',
'Easy Cyclist FP',
'Easy Cyclist FN',
'Easy Cyclist Frame Precision',
'Easy Cyclist Frame Recall',
'Easy Cyclist ΣTP',
'Easy Cyclist ΣFP',
'Easy Cyclist ΣFN',
'Easy Cyclist ΣPrecision',
'Easy Cyclist ΣRecall',
'Easy Cyclist Mean Precision',
'Easy Cyclist Mean Recall',
'Moderate Cyclist TP',
'Moderate Cyclist FP',
'Moderate Cyclist FN',
'Moderate Cyclist Frame Precision',
'Moderate Cyclist Frame Recall',
'Moderate Cyclist ΣTP',
'Moderate Cyclist ΣFP',
'Moderate Cyclist ΣFN',
'Moderate Cyclist ΣPrecision',
'Moderate Cyclist ΣRecall',
'Moderate Cyclist Mean Precision',
'Moderate Cyclist Mean Recall',
'Hard Cyclist TP',
'Hard Cyclist FP',
'Hard Cyclist FN',
'Hard Cyclist Frame Precision',
'Hard Cyclist Frame Recall',
'Hard Cyclist ΣTP',
'Hard Cyclist ΣFP',
'Hard Cyclist ΣFN',
'Hard Cyclist ΣPrecision',
'Hard Cyclist ΣRecall',
'Hard Cyclist Mean Precision',
'Hard Cyclist Mean Recall',
'Overall Cyclist TP',
'Overall Cyclist FP',
'Overall Cyclist FN',
'Overall Cyclist Frame Precision',
'Overall Cyclist Frame Recall',
'Overall Cyclist ΣTP',
'Overall Cyclist ΣFP',
'Overall Cyclist ΣFN',
'Overall Cyclist ΣPrecision',
'Overall Cyclist ΣRecall',
'Overall Cyclist Mean Precision',
'Overall Cyclist Mean Recall',
'Gt Cyclists',
'Filtered Gt Cyclists',
'Gt Easy Cyclists',
'Gt Moderate Cyclists',
'Gt Hard Cyclists',
'Gt Ignored Cyclists',
'Pred Cyclists',
'Filtered Pred Cyclists',
'Pred Easy Cyclists',
'Pred Moderate Cyclists',
'Pred Hard Cyclists',
'Pred Ignored Cyclists',
'ΣGt Cyclists',
'Σ filtered Gt Cyclists',
'ΣGt Easy Cyclists',
'ΣGt Moderate Cyclists',
'ΣGt Hard Cyclists',
'ΣPred Cyclists',
'ΣPred Easy Cyclists',
'ΣPred Moderate Cyclists',
'ΣPred Hard Cyclists',]

field_name_pedestrians=['Easy Pedestrian TP',
'Easy Pedestrian FP',
'Easy Pedestrian FN',
'Easy Pedestrian Frame Precision',
'Easy Pedestrian Frame Recall',
'Easy Pedestrian ΣTP',
'Easy Pedestrian ΣFP',
'Easy Pedestrian ΣFN',
'Easy Pedestrian ΣPrecision',
'Easy Pedestrian ΣRecall',
'Easy Pedestrian Mean Precision',
'Easy Pedestrian Mean Recall',
'Moderate Pedestrian TP',
'Moderate Pedestrian FP',
'Moderate Pedestrian FN',
'Moderate Pedestrian Frame Precision',
'Moderate Pedestrian Frame Recall',
'Moderate Pedestrian ΣTP',
'Moderate Pedestrian ΣFP',
'Moderate Pedestrian ΣFN',
'Moderate Pedestrian ΣPrecision',
'Moderate Pedestrian ΣRecall',
'Moderate Pedestrian Mean Precision',
'Moderate Pedestrian Mean Recall',
'Hard Pedestrian TP',
'Hard Pedestrian FP',
'Hard Pedestrian FN',
'Hard Pedestrian Frame Precision',
'Hard Pedestrian Frame Recall',
'Hard Pedestrian ΣTP',
'Hard Pedestrian ΣFP',
'Hard Pedestrian ΣFN',
'Hard Pedestrian ΣPrecision',
'Hard Pedestrian ΣRecall',
'Hard Pedestrian Mean Precision',
'Hard Pedestrian Mean Recall',
'Overall Pedestrian TP',
'Overall Pedestrian FP',
'Overall Pedestrian FN',
'Overall Pedestrian Frame Precision',
'Overall Pedestrian Frame Recall',
'Overall Pedestrian ΣTP',
'Overall Pedestrian ΣFP',
'Overall Pedestrian ΣFN',
'Overall Pedestrian ΣPrecision',
'Overall Pedestrian ΣRecall',
'Overall Pedestrian Mean Precision',
'Overall Pedestrian Mean Recall',
'Gt Pedestrians',
'Filtered Gt Pedestrians',
'Gt Easy Pedestrians',
'Gt Moderate Pedestrians',
'Gt Hard Pedestrians',
'Gt Ignored Pedestrians',
'Pred Pedestrians',
'Filtered Pred Pedestrians',
'Pred Easy Pedestrians',
'Pred Moderate Pedestrians',
'Pred Hard Pedestrians',
'Pred Ignored Pedestrians',
'ΣGt Pedestrians',
'Σ filtered Gt Pedestrians',
'ΣGt Easy Pedestrians',
'ΣGt Moderate Pedestrians',
'ΣGt Hard Pedestrians',
'ΣPred Pedestrians',
'ΣPred Easy Pedestrians',
'ΣPred Moderate Pedestrians',
'ΣPred Hard Pedestrians',]

def read_groundtruth(gt_folder,fileid):
    boxs_groundtruth_file=os.path.join(gt_folder,str(fileid).zfill(6)+'.txt')
    with open(boxs_groundtruth_file,'r') as file:
        boxs_groundtruth_string=file.read()

    with open(boxs_groundtruth_file,'r') as file:
        firstLine_elements = len(file.readline().split())


    groundtruth_list=boxs_groundtruth_string.split()
    groundtruth = [groundtruth_list[x:x+firstLine_elements] for x in range(0,int(len(groundtruth_list)),firstLine_elements)]

    return groundtruth

def box_does_not_intersect_with_all_boxes_in_list(random_box,boxes_in_list):
    flag=False

    for box in boxes_in_list:
        #print((random_box[0],random_box[1],box[0],box[1]))
        inter_flag,overlap_condition=rectanges_overlap(random_box[0],random_box[1],box[0],box[1])
        
        if inter_flag==True:
            flag=True

            return True,overlap_condition

        else:
            flag=False

    return flag,overlap_condition



def get_plot_groundtruth(img,gt_classes): 
    n_truth=len(gt_classes)
    height=img.shape[0]
    width=img.shape[1]

    print('h: ',height)
    print('w: ',width)

    start_point=(width/2,height/2)
    
                            # easy,moderate,hard
    possible_box_dimensions=[50,30]

    
    img_with_groundtruth=img

    boxs=[]

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    #org = (int(pt1.x+(box_width/2)),int(pt1.y+(box_height/2)))

    # fontScale
    fontScale = 1

    # Blue color in BGR
    green_color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 2

    gt_class_dim_list=[]

    for i in range(n_truth):
        # Shift
        #flag_intersection=False
        random_box_dimensions_selector=random.randint(0,1)
        box_width=possible_box_dimensions[random_box_dimensions_selector]
        box_height=box_width
        gt_class_dim_list.append((box_width,box_height))
        x=random.randint(box_width,width-box_width)
        y=random.randint(box_height,height-box_height)
        pt1=Point(x,y)
        pt2=Point(x+box_width,y+box_height)
        random_box=(pt1,pt2)
        # print('x: ',x)
        # print('y :',y)

        if i!=0:
            flag_intersection,overlap_condition=box_does_not_intersect_with_all_boxes_in_list(random_box,boxs)


            # for i in range(len(boxs)):
            #     box_in_list_pt1=Point(boxs[i][0],boxs[i][1])
            #     box_in_list_pt2=Point(boxs[i][2],boxs[i][3])

            while flag_intersection==True:
                #print('While Loop Counter: ',while_loop_counter)
                x=random.randint(box_width,width-box_width)
                y=random.randint(box_height,height-box_height)
                pt1=Point(x,y)
                pt2=Point(x+box_width,y+box_height)
                random_box=(pt1,pt2)
                flag_intersection,overlap_condition=box_does_not_intersect_with_all_boxes_in_list(random_box,boxs)
                #while_loop_counter+=1

                #print('plotting_point:',while_loop_counter)
            #print('rect: ',((pt1.x,pt1.y),(pt2.x,pt2.y)))

            print('Overlap Condition: ',overlap_condition)
            print('Rectangle: ',i)
            print('Just plotted a rectangle flag intersection was: ',flag_intersection)
            img_with_groundtruth=cv2.rectangle(img_with_groundtruth,(pt1.x,pt1.y),(pt2.x,pt2.y),green_color,2)
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
  
            # org
            org = (int(pt1.x+(box_width/2)),int(pt1.y+(box_height/2)))

   
            # Using cv2.putText() method
            #img_with_groundtruth = cv2.putText(img_with_groundtruth, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
            img_with_groundtruth = cv2.putText(img_with_groundtruth, str(gt_classes[i]), org, font, fontScale, green_color, thickness, cv2.LINE_AA)
            boxs.append((Point(x,y),Point(x+box_width,y+box_height)))

                

            #boxs.append((x,y,x+box_width,y+box_height))

        else:
            flag_intersection=False
            print('Rectangle: ',i)
            print('Just plotted a rectangle flag intersection was: ',flag_intersection)
            img_with_groundtruth=cv2.rectangle(img_with_groundtruth,(pt1.x,pt1.y),(pt2.x,pt2.y),green_color,2)

# org
            org = (int(pt1.x+(box_width/2)),int(pt1.y+(box_height/2)))
   
            # Using cv2.putText() method
            img_with_groundtruth = cv2.putText(img_with_groundtruth, str(gt_classes[i]), org, font, fontScale, green_color, thickness, cv2.LINE_AA)
            boxs.append((Point(x,y),Point(x+box_width,y+box_height)))
            #print('boxs: ',boxs)

        
    return boxs,img_with_groundtruth,gt_class_dim_list

def get_plot_predictions(gt_boxs,box_dimensions,img_dimensions,gt_img,predicted_classes,gt_classes,total_predictions_n):

    img_width=img_dimensions[0]
    img_height=img_dimensions[1]
    pred_boxs=[]
    output_img=gt_img

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    #org = (int(pt1.x+(box_width/2)),int(pt1.y+(box_height/2)))

    # fontScale
    fontScale = 1

    # Blue color in BGR
    red_color = (0,0,255)

    # Line thickness of 2 px
    thickness = 2
    i=0
    for gt_box in gt_boxs[0:total_predictions_n]:
        box_width=box_dimensions[i][0]#box_dimensions[0]
        box_height=box_dimensions[i][1]#x_dimensions[1]
        x_shift=random.randint(-int(box_width/4),int(box_width/4))
        y_shift=random.randint(-int(box_height/4),int(box_height/4))
        pred_pt1=Point(gt_box[0].x+x_shift,gt_box[0].y+y_shift)
        pred_pt2=Point(pred_pt1.x+box_width,pred_pt1.y+box_height)

        pred_boxs.append((pred_pt1,pred_pt2))

        output_img=cv2.rectangle(output_img,(pred_pt1.x,pred_pt1.y),(pred_pt2.x,pred_pt2.y),red_color,thickness)
        org = (int(pred_pt1.x+(box_width/2)),int(pred_pt1.y+(box_height/2)))
        output_img = cv2.putText(output_img, predicted_classes[i], org, font, fontScale, red_color, thickness, cv2.LINE_AA)
        i+=1
        

    if len(predicted_classes)>len(gt_boxs):
        false_positive_index_range=range(len(gt_boxs),len(predicted_classes))
        print('False positive index range: ',list(false_positive_index_range))
        for i in list(false_positive_index_range):
            x=random.randint(box_width,img_width-box_width)
            y=random.randint(box_height,img_height-box_height)
            pred_pt1=Point(x,y)
            pred_pt2=Point(x+box_width,y+box_height)
            random_box=(pred_pt1,pred_pt2)

            flag_intersection,overlap_condition=box_does_not_intersect_with_all_boxes_in_list(random_box,gt_boxs+pred_boxs)


            # for i in range(len(boxs)):q
            #     box_in_list_pt1=Point(boxs[i][0],boxs[i][1])
            #     box_in_list_pt2=Point(boxs[i][2],boxs[i][3])

            while flag_intersection==True:
                #print('While Loop Counter: ',while_loop_counter)
                x=random.randint(box_width,img_width-box_width)
                y=random.randint(box_height,img_height-box_height)
                pred_pt1=Point(x,y)
                pred_pt2=Point(x+box_width,y+box_height)
                random_box=(pred_pt1,pred_pt2)
                flag_intersection,overlap_condition=box_does_not_intersect_with_all_boxes_in_list(random_box,gt_boxs+pred_boxs)


            output_img=cv2.rectangle(output_img,(pred_pt1.x,pred_pt1.y),(pred_pt2.x,pred_pt2.y),red_color,thickness)
            print('i in FPs: ',i)
            print('FP Predicted Class Index:',predicted_classes[i])
            org = (int(pred_pt1.x+(box_width/2)),int(pred_pt1.y+(box_height/2)))
            output_img = cv2.putText(output_img, predicted_classes[i], org, font, fontScale, red_color, thickness, cv2.LINE_AA)
            
            false_positive_box=[random_box]

            pred_boxs=pred_boxs+false_positive_box
    else:
        pass

    return pred_boxs,output_img

def get_gt_classes(labels,n_gt_detections):

    gt_classes=[labels[random.randint(0,len(labels)-1)] for i in range(n_gt_detections)]

    return gt_classes

def get_predicted_classes(labels,total_predictions_n):


    predicted_classes=[labels[random.randint(0,len(labels)-1)] for i in range(total_predictions_n)]

    # for i in range(false_positive_n):
    #     x_shift=random.randint(-box_width/4,box_width/4)
    #     y_shift=random.randint(-box_height/4,box_height/4)
    #     pred_pt1=Point(gt_box[0].x+x_shift,gt_box[0].y+y_shift)
    #     pred_pt2=Point(pred_pt1.x+box_width,pred_pt1.y+box_height)
        
    #     false_positive_box=[(pred_pt1,pred_pt2)]

    #     pred_boxs=pred_boxs+false_positive_box

    return predicted_classes

# def get_metrics(gt_classes,predicted_classes,labels):
#     gt_label_count=[]
#     pred_label_count=[]
#     for label in labels:
#         gt_label_count.append(gt_classes.count(label))
#         pred_label_count.append(predicted_classes.count(label))

#     if len(predicted_classes)>len(gt_classes):
#         FP=len(predicted_classes)-len(gt_classes)
#     else:
#         FP=0
    
#     TP=0

#     if len(gt_classes)>len(predicted_classes):
#         indix_range=len(predicted_classes)

#     else:
#         indix_range=len(gt_classes)

#     for i in range(indix_range):
#         if gt_classes[i]==predicted_classes[i]:
#             TP=TP+1

#         else:
#             FP=FP+1


#     gt_ROIs=len(gt_classes)
#     predicted_ROIs=len(pred_boxs)
#     FN=len(gt_boxs)-TP

#     return gt_ROIs,predicted_ROIs,TP,FP,FN



def get_metrics_label_specific(label,difficulty,labels,gt_classes,predicted_classes,gt_boxs,pred_boxs):
    gt_label_count=[]
    pred_label_count=[]
    for l in labels:
        gt_label_count.append(gt_classes.count(l))
        pred_label_count.append(predicted_classes.count(l))

    i=labels.index(label)
    print('i: ',i)
    # gt_instances_label=gt_label_count[i]
    # predicted_instances_label=pred_label_count[i]

    #print()

    print('Gt Label Count: ',gt_label_count)
    print('Predictions Label Count: ',pred_label_count)

    print('Gt Class A Instances: ',gt_label_count[0])
    print('Pred Class A Instances: ',pred_label_count[0])

    print('Gt Class B Instances: ',gt_label_count[1])
    print('Pred Class B Instances: ',pred_label_count[1])

    print('Gt Class C Instances: ',gt_label_count[2])
    print('Pred Class C Instances: ',pred_label_count[2])

    

    gt_index_list=get_box_difficulty_index(gt_boxs,gt_classes,label,difficulty)
    pred_index_list=get_box_difficulty_index(pred_boxs,predicted_classes,label,difficulty)

    print('Predicted Class Instances: ',pred_label_count[0])
    print('Predicted Easy instances: ',len(pred_index_list))


    print('Gt Class Instances: ',gt_label_count[0])
    print('Gt Easy instances: ', len(gt_index_list))



    gt_classes_label=[gt_classes[i] for i in gt_index_list] 

    predicted_classes_label=[predicted_classes[i] for i in pred_index_list]

    # gt_instances_label=len(gt_classes)
    # predicted_instances_label=len(predicted_classes)


    print('GT Classes index range',gt_index_list)
    print('Pred Classes index range',pred_index_list)

    print('---------------------------------------------')

    print('gt classes: ',gt_classes)
    print('pred classes: ',predicted_classes)


    if len(gt_classes)>len(predicted_classes):
        indix_range=pred_index_list

    else:
        indix_range=gt_index_list

    TP=0

    # for box in gt_boxs:
    #     print('Box: ',(box[0.]))

    print('n gt boxs: ',len(gt_boxs))
    print('n pred boxs: ',len(pred_boxs))

    if len(gt_classes)>len(predicted_classes):
        indix_range= gt_index_list
    else:
        indix_range=pred_index_list


    iou_list=[]

    for x in range(len(gt_boxs)):
        for j in range(len(pred_boxs)):
            iou=get_IoU(gt_boxs[x],pred_boxs[j])
            iou_list.append(iou)
            print('IoU: ',iou)

            # for i in indix_range:

            if gt_classes[x]=='Car':
                iou_threshold=0.7
            else:
                iou_threshold=0.5

            if iou>iou_threshold:
        
                if gt_classes[x]==predicted_classes[j] and gt_classes[x]==label and x in gt_index_list and j in pred_index_list:
                    TP=TP+1
                    #print('IoU: ',get_IoU(gt_boxs[counter],pred_boxs[counter]))
                else:
                    pass
            else:
                pass

    FP=len(predicted_classes_label)-TP
    FN=len(gt_classes_label)-TP

    print('IoU List: ',iou_list)

        

    return len(gt_classes_label),len(predicted_classes_label),TP,FP,FN



def get_metrics_label_specific_2(label,difficulty,labels,gt_classes,predicted_classes,gt_boxs,pred_boxs,gt_truncation,gt_occlusion):

    # Get Labels Count For Groundtruth and Predictions List
    gt_label_count=[]
    pred_label_count=[]
    for l in labels:
        gt_label_count.append(gt_classes.count(l))
        pred_label_count.append(predicted_classes.count(l))

    i=labels.index(label)
    print('i: ',i)
    # gt_instances_label=gt_label_count[i]
    # predicted_instances_label=pred_label_count[i]

    #print()

    print('Gt Label Count: ',gt_label_count)
    print('Predictions Label Count: ',pred_label_count)

    print('Gt Class A Instances: ',gt_label_count[0])
    print('Pred Class A Instances: ',pred_label_count[0])

    print('Gt Class B Instances: ',gt_label_count[1])
    print('Pred Class B Instances: ',pred_label_count[1])

    print('Gt Class C Instances: ',gt_label_count[2])
    print('Pred Class C Instances: ',pred_label_count[2])

    

    gt_index_list=get_box_difficulty_index_2(gt_boxs,gt_classes,gt_truncation,gt_occlusion,label,difficulty)
    pred_index_list=[]#get_box_difficulty_index(pred_boxs,predicted_classes,label,difficulty)

    print('Predicted Class Instances: ',pred_label_count[0])
    #print('Predicted Easy instances: ',len(pred_index_list))


    print('Gt Class Instances: ',gt_label_count[0])
    print('Gt Easy instances: ', len(gt_index_list))



    gt_classes_label=[gt_classes[i] for i in gt_index_list] 

    

    # gt_instances_label=len(gt_classes)
    # predicted_instances_label=len(predicted_classes)


    print('GT Classes index range',gt_index_list)
    #print('Pred Classes index range',pred_index_list)

    print('---------------------------------------------')

    print('gt classes: ',gt_classes)
    print('pred classes: ',predicted_classes)




    TP=0
    FP=0

    # for box in gt_boxs:
    #     print('Box: ',(box[0.]))

    print('n gt boxs: ',len(gt_boxs))
    print('n pred boxs: ',len(pred_boxs))



    iou_list=[]

    for x in range(len(gt_boxs)):
        for j in range(len(pred_boxs)):
            iou=get_IoU(gt_boxs[x],pred_boxs[j])
            iou_list.append(iou)
            print('IoU: ',iou)

            # for i in indix_range:

            if gt_classes[x]=='Car':
                iou_threshold=0.7
            else:
                iou_threshold=0.5


            if gt_classes[x]==predicted_classes[j] and gt_classes[x]==label and x in gt_index_list:

                if iou>iou_threshold:
                    TP=TP+1
                    pred_index_list.append(j)
                
                else:
                    pass
                    #print('IoU: ',get_IoU(gt_boxs[counter],pred_boxs[counter]))
            else:
                pass


    

    difficulty_unidentified_predictions_boxs=[pred_boxs[i] for i in range(len(pred_boxs)) if i not in pred_index_list]
    difficulty_unidentified_predictions_classes=[predicted_classes[i] for i in range(len(pred_boxs)) if i not in pred_index_list]

    identified_predictions_indices_difficulty_by_height=get_box_difficulty_index(difficulty_unidentified_predictions_boxs,difficulty_unidentified_predictions_classes,label,difficulty)

    pred_index_list=pred_index_list+identified_predictions_indices_difficulty_by_height
    predicted_classes_label=[predicted_classes[i] for i in pred_index_list]
    print('Total Predictions: ',len(pred_boxs))
    print('Difficulty Specific Predictions: ',len(pred_index_list))
    print('Difficulty Unidentified Predictions: ',len(difficulty_unidentified_predictions_boxs))

    FP=len(predicted_classes_label)-TP
    FN=len(gt_classes_label)-TP

    print('IoU List: ',iou_list)

        

    return len(gt_classes_label),len(predicted_classes_label),TP,FP,FN


    


# def visualize_metrics_calculations(gt_ROIs_n,predicted_ROIs_n,TP,FP,FN):
#     # font
#     font = cv2.FONT_HERSHEY_SIMPLEX

#     # org
#     #org = (int(pt1.x+(box_width/2)),int(pt1.y+(box_height/2)))

#     # fontScale
#     fontScale = 1

#     # Blue color in BGR
#     red_color = (0,0,255)

#     # Line thickness of 2 px
#     thickness = 2

#     if predicted_ROIs_n==TP+FP:
#         check_precision=True
#     else:
#         check_precision=False

#     if gt_ROIs_n==TP+FN:
#         check_recall=True
#     else:
#         check_recall=False

    


#     evaluation_metrics_window=np.zeros((img_height,img_width,3))
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '# Ground Truth Objects: '+str(gt_ROIs_n), (0,30), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '# Predicted Objects: '+str(predicted_ROIs_n), (0,60), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'TP: '+str(TP), (0,90), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'FP: '+str(FP), (0,120), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'FN: '+str(FN), (0,150), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     #print("The band for {} is {} out of 10".format(name, band))
#     precision_string_expression="Precision = TP/(TP+FP) = {} / {}  Check = {}".format(TP,TP+FP,check_precision)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, precision_string_expression, (0,180), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     recall_string_expression="Recall = TP/(TP+FN) = {} / {}  Check = {}".format(TP,TP+FN,check_recall)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, recall_string_expression, (0,210), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '(TP+FP) = # Predicted Objects', (0,240), font, fontScale, red_color, thickness, cv2.LINE_AA)
#     evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '(TP+FN) = # Ground Truth Objects', (0,270), font, fontScale, red_color, thickness, cv2.LINE_AA)


#     return evaluation_metrics_window


    

def get_box_difficulty_index(gt_boxs,gt_classes,label,difficulty):

    easy_difficulty_index_list=[]
    moderate_difficulty_index_list=[]
    hard_difficulty_index_list=[]

    print('n gt boxs: ',len(gt_boxs))
    print('n gt classes: ',len(gt_classes))

    #print(gt_boxs)
    for i in range(len(gt_boxs)):
        box_height=gt_boxs[i][1].y-gt_boxs[i][0].y
        if box_height>40:
            #easy_difficulty_index_list.append(i)

            if gt_classes[i]==label:
                easy_difficulty_index_list.append(i)
            else:
                pass
        elif 25<box_height<40:
           # moderate_difficulty_index_list.append(i)

            if gt_classes[i]==label:
                moderate_difficulty_index_list.append(i)
            else:
                pass
        else:
            pass

    if difficulty=='Easy':
        return easy_difficulty_index_list

    elif difficulty=='Moderate':
        return moderate_difficulty_index_list
    else:
        return hard_difficulty_index_list


def get_box_difficulty_index_2(gt_boxs,gt_classes,gt_truncation,gt_occlusion,label,difficulty):

    easy_difficulty_index_list=[]
    moderate_difficulty_index_list=[]
    hard_difficulty_index_list=[]
    dont_care_index_list=[]


    print('n gt boxs: ',len(gt_boxs))
    print('n gt classes: ',len(gt_classes))

    #print(gt_boxs)
    for i in range(len(gt_boxs)):
        box_height=gt_boxs[i][1].y-gt_boxs[i][0].y
        truncation=gt_truncation[i]
        occlusion=gt_occlusion[i]
        if box_height>=40 and occlusion==0 and truncation<=0.15:
            #easy_difficulty_index_list.append(i)

            if gt_classes[i]==label:
                easy_difficulty_index_list.append(i)
            else:
                pass
        elif box_height>=25 and occlusion in [0,1] and truncation<=0.3:
           # moderate_difficulty_index_list.append(i)

            if gt_classes[i]==label:
                moderate_difficulty_index_list.append(i)
            else:
                pass

        elif box_height>=25 and occlusion in [0,1,2] and truncation<=0.5:

            if gt_classes[i]==label:
                hard_difficulty_index_list.append(i)
            else:
                pass

        else:
            dont_care_index_list.append(i)

    if difficulty=='Easy':
        return easy_difficulty_index_list

    elif difficulty=='Moderate':
        return moderate_difficulty_index_list
    else:
        return hard_difficulty_index_list


def tabularize_metrics(data_array,gt_total_class_count,filtered_gt_class_count,gt_difficulty_class_count,ignored_gt_count,pred_total_class_count,pred_filtered_class_count,pred_difficulty_class_count,ignored_pred_count,n_frames,path):
    TP_A1_list=data_array[:,0]
    FP_A1_list=data_array[:,1]
    FN_A1_list=data_array[:,2]

    TP_A2_list=data_array[:,3]
    FP_A2_list=data_array[:,4]
    FN_A2_list=data_array[:,5]

    TP_A3_list=data_array[:,6]
    FP_A3_list=data_array[:,7]
    FN_A3_list=data_array[:,8]


    TP_A_list=TP_A1_list+TP_A2_list+TP_A3_list
    FP_A_list=FP_A1_list+FP_A2_list+FP_A3_list
    FN_A_list=FN_A1_list+FN_A2_list+FN_A3_list

    


    TP_B1_list=data_array[:,9]
    FP_B1_list=data_array[:,10]
    FN_B1_list=data_array[:,11]

    TP_B2_list=data_array[:,12]
    FP_B2_list=data_array[:,13]
    FN_B2_list=data_array[:,14]

    TP_B3_list=data_array[:,15]
    FP_B3_list=data_array[:,16]
    FN_B3_list=data_array[:,17]

    TP_B_list=TP_B1_list+TP_B2_list+TP_B3_list
    FP_B_list=FP_B1_list+FP_B2_list+FP_B3_list
    FN_B_list=FN_B1_list+FN_B2_list+FN_B3_list

    TP_C1_list=data_array[:,18]
    FP_C1_list=data_array[:,19]
    FN_C1_list=data_array[:,20]

    TP_C2_list=data_array[:,21]
    FP_C2_list=data_array[:,22]
    FN_C2_list=data_array[:,23]

    TP_C3_list=data_array[:,24]
    FP_C3_list=data_array[:,25]
    FN_C3_list=data_array[:,26]

    TP_C_list=TP_C1_list+TP_C2_list+TP_C3_list
    FP_C_list=FP_C1_list+FP_C2_list+FP_C3_list
    FN_C_list=FN_C1_list+FN_C2_list+FN_C3_list

    TP_easy_list=TP_A1_list+TP_C1_list
    FP_easy_list=FP_A1_list+FP_C1_list
    FN_easy_list=FN_A1_list+FN_C1_list


    TP_moderate_list=TP_A2_list+TP_C2_list
    FP_moderate_list=FP_A2_list+FP_C2_list
    FN_moderate_list=FN_A2_list+FN_C2_list

    TP_hard_list=TP_A3_list+TP_C3_list
    FP_hard_list=FP_A3_list+FP_C3_list
    FN_hard_list=FN_A3_list+FN_C3_list

    ΣTP_A1=[sum(TP_A1_list[0:i+1]) for i in range(len(TP_A1_list))]
    ΣFP_A1=[sum(FP_A1_list[0:i+1]) for i in range(len(FP_A1_list))]
    ΣFN_A1=[sum(FN_A1_list[0:i+1]) for i in range(len(FN_A1_list))]

    ΣTP_A2=[sum(TP_A2_list[0:i+1]) for i in range(len(TP_A2_list))]
    ΣFP_A2=[sum(FP_A2_list[0:i+1]) for i in range(len(FP_A2_list))]
    ΣFN_A2=[sum(FN_A2_list[0:i+1]) for i in range(len(FN_A2_list))]

    ΣTP_A3=[sum(TP_A3_list[0:i+1]) for i in range(len(TP_A3_list))]
    ΣFP_A3=[sum(FP_A3_list[0:i+1]) for i in range(len(FP_A3_list))]
    ΣFN_A3=[sum(FN_A3_list[0:i+1]) for i in range(len(FN_A3_list))]

    ΣTP_A=[sum(TP_A_list[0:i+1]) for i in range(len(TP_A_list))]
    ΣFP_A=[sum(FP_A_list[0:i+1]) for i in range(len(FP_A_list))]
    ΣFN_A=[sum(FN_A_list[0:i+1]) for i in range(len(FN_A_list))]

    ΣTP_B1=[sum(TP_B1_list[0:i+1]) for i in range(len(TP_B1_list))]
    ΣFP_B1=[sum(FP_B1_list[0:i+1]) for i in range(len(FP_B1_list))]
    ΣFN_B1=[sum(FN_B1_list[0:i+1]) for i in range(len(FN_B1_list))]

    ΣTP_B2=[sum(TP_B2_list[0:i+1]) for i in range(len(TP_B2_list))]
    ΣFP_B2=[sum(FP_B2_list[0:i+1]) for i in range(len(FP_B2_list))]
    ΣFN_B2=[sum(FN_B2_list[0:i+1]) for i in range(len(FN_B2_list))]

    ΣTP_B3=[sum(TP_B3_list[0:i+1]) for i in range(len(TP_B3_list))]
    ΣFP_B3=[sum(FP_B3_list[0:i+1]) for i in range(len(FP_B3_list))]
    ΣFN_B3=[sum(FN_B3_list[0:i+1]) for i in range(len(FN_B3_list))]

    ΣTP_B=[sum(TP_B_list[0:i+1]) for i in range(len(TP_B_list))]
    ΣFP_B=[sum(FP_B_list[0:i+1]) for i in range(len(FP_B_list))]
    ΣFN_B=[sum(FN_B_list[0:i+1]) for i in range(len(FN_B_list))]

    ΣTP_C1=[sum(TP_C1_list[0:i+1]) for i in range(len(TP_C1_list))]
    ΣFP_C1=[sum(FP_C1_list[0:i+1]) for i in range(len(FP_C1_list))]
    ΣFN_C1=[sum(FN_C1_list[0:i+1]) for i in range(len(FN_C1_list))]

    ΣTP_C2=[sum(TP_C2_list[0:i+1]) for i in range(len(TP_C2_list))]
    ΣFP_C2=[sum(FP_C2_list[0:i+1]) for i in range(len(FP_C2_list))]
    ΣFN_C2=[sum(FN_C2_list[0:i+1]) for i in range(len(FN_C2_list))]

    ΣTP_C3=[sum(TP_C3_list[0:i+1]) for i in range(len(TP_C3_list))]
    ΣFP_C3=[sum(FP_C3_list[0:i+1]) for i in range(len(FP_C3_list))]
    ΣFN_C3=[sum(FN_C3_list[0:i+1]) for i in range(len(FN_C3_list))]

    ΣTP_C=[sum(TP_C_list[0:i+1]) for i in range(len(TP_C_list))]
    ΣFP_C=[sum(FP_C_list[0:i+1]) for i in range(len(FP_C_list))]
    ΣFN_C=[sum(FN_C_list[0:i+1]) for i in range(len(FN_C_list))]
    #for element in 

    ΣTP_easy=[sum(TP_easy_list[0:i+1]) for i in range(len(TP_easy_list))]
    ΣFP_easy=[sum(FP_easy_list[0:i+1]) for i in range(len(FP_easy_list))]
    ΣFN_easy=[sum(FN_easy_list[0:i+1]) for i in range(len(FN_easy_list))]

    ΣTP_moderate=[sum(TP_moderate_list[0:i+1]) for i in range(len(TP_moderate_list))]
    ΣFP_moderate=[sum(FP_moderate_list[0:i+1]) for i in range(len(FP_moderate_list))]
    ΣFN_moderate=[sum(FN_moderate_list[0:i+1]) for i in range(len(FN_moderate_list))]

    ΣTP_hard=[sum(TP_hard_list[0:i+1]) for i in range(len(TP_hard_list))]
    ΣFP_hard=[sum(FP_hard_list[0:i+1]) for i in range(len(FP_hard_list))]
    ΣFN_hard=[sum(FN_hard_list[0:i+1]) for i in range(len(FN_hard_list))]

    print('Check: ',len(TP_A1_list)==n_frames)
    ΣA1_precision=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFP_A1[i])  if int(ΣTP_A1[i]+ΣFP_A1[i]) is not 0 else 0 for i in range(n_frames)]
    ΣA1_recall=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFN_A1[i]) if int(ΣTP_A1[i]+ΣFN_A1[i]) is not 0 else 0 for i in range(n_frames)  ]

    ΣA2_precision=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFP_A2[i]) if int(ΣTP_A2[i]+ΣFP_A2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣA2_recall=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFN_A2[i])  if int(ΣTP_A2[i]+ΣFN_A2[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣA3_precision=[ΣTP_A3[i]/(ΣTP_A3[i]+ΣFP_A3[i]) if int(ΣTP_A3[i]+ΣFP_A3[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣA3_recall=[ΣTP_A3[i]/(ΣTP_A3[i]+ΣFN_A3[i])  if int(ΣTP_A3[i]+ΣFN_A3[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣA_precision=[ΣTP_A[i]/(ΣTP_A[i]+ΣFP_A[i]) if int(ΣTP_A[i]+ΣFP_A[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣA_recall=[ΣTP_A[i]/(ΣTP_A[i]+ΣFN_A[i])  if int(ΣTP_A[i]+ΣFN_A[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB1_precision=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFP_B1[i]) if int(ΣTP_B1[i]+ΣFP_B1[i]) is not 0 else 0 for i in range(n_frames)  ]
    ΣB1_recall=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFN_B1[i]) if int(ΣTP_B1[i]+ΣFN_B1[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB2_precision=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFP_B2[i]) if int(ΣTP_B2[i]+ΣFP_B2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣB2_recall=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFN_B2[i]) if int(ΣTP_B2[i]+ΣFN_B2[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB3_precision=[ΣTP_B3[i]/(ΣTP_B3[i]+ΣFP_B3[i]) if int(ΣTP_B3[i]+ΣFP_B3[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣB3_recall=[ΣTP_B3[i]/(ΣTP_B3[i]+ΣFN_B3[i]) if int(ΣTP_B3[i]+ΣFN_B3[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB_precision=[ΣTP_B[i]/(ΣTP_B[i]+ΣFP_B[i]) if int(ΣTP_B[i]+ΣFP_B[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣB_recall=[ΣTP_B[i]/(ΣTP_B[i]+ΣFN_B[i])  if int(ΣTP_B[i]+ΣFN_B[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC1_precision=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFP_C1[i]) if int(ΣTP_C1[i]+ΣFP_C1[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC1_recall=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFN_C1[i]) if int(ΣTP_C1[i]+ΣFN_C1[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC2_precision=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFP_C2[i]) if int(ΣTP_C2[i]+ΣFP_C2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC2_recall=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFN_C2[i]) if int(ΣTP_C2[i]+ΣFN_C2[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC3_precision=[ΣTP_C3[i]/(ΣTP_C3[i]+ΣFP_C3[i]) if int(ΣTP_C3[i]+ΣFP_C3[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC3_recall=[ΣTP_C3[i]/(ΣTP_C3[i]+ΣFN_C3[i]) if int(ΣTP_C3[i]+ΣFN_C3[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC_precision=[ΣTP_C[i]/(ΣTP_C[i]+ΣFP_C[i]) if int(ΣTP_C[i]+ΣFP_C[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC_recall=[ΣTP_C[i]/(ΣTP_C[i]+ΣFN_C[i])  if int(ΣTP_C[i]+ΣFN_C[i]) is not 0 else 0 for i in range(n_frames) ]

    Σeasy_precision=[ΣTP_easy[i]/(ΣTP_easy[i]+ΣFP_easy[i]) if int(ΣTP_easy[i]+ΣFP_easy[i]) is not 0 else 0 for i in range(n_frames) ]
    Σeasy_recall=[ΣTP_easy[i]/(ΣTP_easy[i]+ΣFN_easy[i])  if int(ΣTP_easy[i]+ΣFN_easy[i]) is not 0 else 0 for i in range(n_frames) ]

    Σmoderate_precision=[ΣTP_moderate[i]/(ΣTP_moderate[i]+ΣFP_moderate[i]) if int(ΣTP_moderate[i]+ΣFP_moderate[i]) is not 0 else 0 for i in range(n_frames) ]
    Σmoderate_recall=[ΣTP_moderate[i]/(ΣTP_moderate[i]+ΣFN_moderate[i])  if int(ΣTP_moderate[i]+ΣFN_moderate[i]) is not 0 else 0 for i in range(n_frames) ]

    Σhard_precision=[ΣTP_hard[i]/(ΣTP_hard[i]+ΣFP_hard[i]) if int(ΣTP_hard[i]+ΣFP_hard[i]) is not 0 else 0 for i in range(n_frames) ]
    Σhard_recall=[ΣTP_hard[i]/(ΣTP_hard[i]+ΣFN_hard[i])  if int(ΣTP_hard[i]+ΣFN_hard[i]) is not 0 else 0 for i in range(n_frames) ]

    A1_frame_precision=[TP_A1_list[i]/(TP_A1_list[i]+FP_A1_list[i]) if int(TP_A1_list[i]+FP_A1_list[i]) is not 0 else 0 for i in range(n_frames)]
    A1_mean_precision=[sum(A1_frame_precision[0:i+1])/len(A1_frame_precision[0:i+1]) for i in range(n_frames)]
    A1_frame_recall=[TP_A1_list[i]/(TP_A1_list[i]+FN_A1_list[i]) if int(TP_A1_list[i]+FN_A1_list[i]) is not 0 else 0 for i in range(n_frames)]
    A1_mean_recall=[sum(A1_frame_recall[0:i+1])/len(A1_frame_recall[0:i+1]) for i in range(n_frames)]

    A2_frame_precision=[TP_A2_list[i]/(TP_A2_list[i]+FP_A2_list[i]) if int(TP_A2_list[i]+FP_A2_list[i]) is not 0 else 0 for i in range(n_frames)]
    A2_mean_precision=[sum(A2_frame_precision[0:i+1])/len(A2_frame_precision[0:i+1]) for i in range(n_frames)]
    A2_frame_recall=[TP_A2_list[i]/(TP_A2_list[i]+FN_A2_list[i]) if int(TP_A2_list[i]+FN_A2_list[i]) is not 0 else 0 for i in range(len(TP_A2_list))]
    A2_mean_recall=[sum(A2_frame_recall[0:i+1])/len(A2_frame_recall[0:i+1]) for i in range(n_frames)]

    A3_frame_precision=[TP_A3_list[i]/(TP_A3_list[i]+FP_A3_list[i]) if int(TP_A3_list[i]+FP_A3_list[i]) is not 0 else 0 for i in range(n_frames)]
    A3_mean_precision=[sum(A3_frame_precision[0:i+1])/len(A3_frame_precision[0:i+1]) for i in range(n_frames)]
    A3_frame_recall=[TP_A3_list[i]/(TP_A3_list[i]+FN_A3_list[i]) if int(TP_A3_list[i]+FN_A3_list[i]) is not 0 else 0 for i in range(len(TP_A3_list))]
    A3_mean_recall=[sum(A3_frame_recall[0:i+1])/len(A3_frame_recall[0:i+1]) for i in range(n_frames)]

    A_frame_precision=[TP_A_list[i]/(TP_A_list[i]+FP_A_list[i]) if int(TP_A_list[i]+FP_A_list[i]) is not 0 else 0 for i in range(n_frames)]
    A_mean_precision=[sum(A_frame_precision[0:i+1])/len(A_frame_precision[0:i+1]) for i in range(n_frames)]
    A_frame_recall=[TP_A_list[i]/(TP_A_list[i]+FN_A_list[i]) if int(TP_A_list[i]+FN_A_list[i]) is not 0 else 0 for i in range(len(TP_A_list))]
    A_mean_recall=[sum(A_frame_recall[0:i+1])/len(A_frame_recall[0:i+1]) for i in range(n_frames)]

    B1_frame_precision=[TP_B1_list[i]/(TP_B1_list[i]+FP_B1_list[i]) if int(TP_B1_list[i]+FP_B1_list[i]) is not 0 else 0 for i in range(n_frames)]
    B1_mean_precision=[sum(B1_frame_precision[0:i+1])/len(B1_frame_precision[0:i+1]) for i in range(n_frames)]
    B1_frame_recall=[TP_B1_list[i]/(TP_B1_list[i]+FN_B1_list[i]) if int(TP_B1_list[i]+FN_B1_list[i]) is not 0 else 0 for i in range(len(TP_B1_list))]
    B1_mean_recall=[sum(B1_frame_recall[0:i+1])/len(B1_frame_recall[0:i+1]) for i in range(n_frames)]

    B2_frame_precision=[TP_B2_list[i]/(TP_B2_list[i]+FP_B2_list[i]) if int(TP_B2_list[i]+FP_B2_list[i]) is not 0 else 0 for i in range(n_frames)]
    B2_mean_precision=[sum(B2_frame_precision[0:i+1])/len(B2_frame_precision[0:i+1]) for i in range(n_frames)]
    B2_frame_recall=[TP_B2_list[i]/(TP_B2_list[i]+FN_B2_list[i]) if int(TP_B2_list[i]+FN_B2_list[i]) is not 0 else 0 for i in range(len(TP_B2_list))]
    B2_mean_recall=[sum(B2_frame_recall[0:i+1])/len(B2_frame_recall[0:i+1]) for i in range(n_frames)]

    B3_frame_precision=[TP_B3_list[i]/(TP_B3_list[i]+FP_B3_list[i]) if int(TP_B3_list[i]+FP_B3_list[i]) is not 0 else 0 for i in range(n_frames)]
    B3_mean_precision=[sum(B3_frame_precision[0:i+1])/len(B3_frame_precision[0:i+1]) for i in range(n_frames)]
    B3_frame_recall=[TP_B3_list[i]/(TP_B3_list[i]+FN_B3_list[i]) if int(TP_B3_list[i]+FN_B3_list[i]) is not 0 else 0 for i in range(len(TP_B3_list))]
    B3_mean_recall=[sum(B3_frame_recall[0:i+1])/len(B3_frame_recall[0:i+1]) for i in range(n_frames)]

    B_frame_precision=[TP_B_list[i]/(TP_B_list[i]+FP_B_list[i]) if int(TP_B_list[i]+FP_B_list[i]) is not 0 else 0 for i in range(n_frames)]
    B_mean_precision=[sum(B_frame_precision[0:i+1])/len(B_frame_precision[0:i+1]) for i in range(n_frames)]
    B_frame_recall=[TP_B_list[i]/(TP_B_list[i]+FN_B_list[i]) if int(TP_B_list[i]+FN_B_list[i]) is not 0 else 0 for i in range(len(TP_B_list))]
    B_mean_recall=[sum(B_frame_recall[0:i+1])/len(B_frame_recall[0:i+1]) for i in range(n_frames)]

    C1_frame_precision=[TP_C1_list[i]/(TP_C1_list[i]+FP_C1_list[i]) if int(TP_C1_list[i]+FP_C1_list[i]) is not 0 else 0 for i in range(n_frames)]
    C1_mean_precision=[sum(C1_frame_precision[0:i+1])/len(C1_frame_precision[0:i+1]) for i in range(n_frames)]
    C1_frame_recall=[TP_C1_list[i]/(TP_C1_list[i]+FN_C1_list[i]) if int(TP_C1_list[i]+FN_C1_list[i]) is not 0 else 0 for i in range(len(TP_C1_list))]
    C1_mean_recall=[sum(C1_frame_recall[0:i+1])/len(C1_frame_recall[0:i+1]) for i in range(n_frames)]

    C2_frame_precision=[TP_C2_list[i]/(TP_C2_list[i]+FP_C2_list[i]) if int(TP_C2_list[i]+FP_C2_list[i]) is not 0 else 0 for i in range(n_frames)]
    C2_mean_precision=[sum(C2_frame_precision[0:i+1])/len(C2_frame_precision[0:i+1]) for i in range(n_frames)]
    C2_frame_recall=[TP_C2_list[i]/(TP_C2_list[i]+FN_C2_list[i]) if int(TP_C2_list[i]+FN_C2_list[i]) is not 0 else 0 for i in range(len(TP_C2_list))]
    C2_mean_recall=[sum(C2_frame_recall[0:i+1])/len(C2_frame_recall[0:i+1]) for i in range(n_frames)]

    C3_frame_precision=[TP_C3_list[i]/(TP_C3_list[i]+FP_C3_list[i]) if int(TP_C3_list[i]+FP_C3_list[i]) is not 0 else 0 for i in range(n_frames)]
    C3_mean_precision=[sum(C3_frame_precision[0:i+1])/len(C3_frame_precision[0:i+1]) for i in range(n_frames)]
    C3_frame_recall=[TP_C3_list[i]/(TP_C3_list[i]+FN_C3_list[i]) if int(TP_C3_list[i]+FN_C3_list[i]) is not 0 else 0 for i in range(len(TP_C3_list))]
    C3_mean_recall=[sum(C3_frame_recall[0:i+1])/len(C3_frame_recall[0:i+1]) for i in range(n_frames)]

    C_frame_precision=[TP_C_list[i]/(TP_C_list[i]+FP_C_list[i]) if int(TP_C_list[i]+FP_C_list[i]) is not 0 else 0 for i in range(n_frames)]
    C_mean_precision=[sum(C_frame_precision[0:i+1])/len(C_frame_precision[0:i+1]) for i in range(n_frames)]
    C_frame_recall=[TP_C_list[i]/(TP_C_list[i]+FN_C_list[i]) if int(TP_C_list[i]+FN_C_list[i]) is not 0 else 0 for i in range(len(TP_C_list))]
    C_mean_recall=[sum(C_frame_recall[0:i+1])/len(C_frame_recall[0:i+1]) for i in range(n_frames)]

    gt_class_A_count=[class_count[0] for class_count in gt_total_class_count]
    gt_class_B_count=[class_count[1] for class_count in gt_total_class_count]
    gt_class_C_count=[class_count[2] for class_count in gt_total_class_count]

    filtered_gt_class_A_count=[class_count[0] for class_count in filtered_gt_class_count]
    filtered_gt_class_B_count=[class_count[1] for class_count in filtered_gt_class_count]
    filtered_gt_class_C_count=[class_count[2] for class_count in filtered_gt_class_count]

    ignored_gt_class_A_count=[class_count[0] for class_count in ignored_gt_count]
    ignored_gt_class_B_count=[class_count[1] for class_count in ignored_gt_count]
    ignored_gt_class_C_count=[class_count[2] for class_count in ignored_gt_count]



    gt_class_A1_count=[lst[0] for lst in gt_difficulty_class_count]
    gt_class_A2_count=[lst[1] for lst in gt_difficulty_class_count]
    gt_class_A3_count=[lst[2] for lst in gt_difficulty_class_count]

    gt_class_B1_count=[lst[3] for lst in gt_difficulty_class_count]
    gt_class_B2_count=[lst[4] for lst in gt_difficulty_class_count]
    gt_class_B3_count=[lst[5] for lst in gt_difficulty_class_count]

    gt_class_C1_count=[lst[6] for lst in gt_difficulty_class_count]
    gt_class_C2_count=[lst[7] for lst in gt_difficulty_class_count]
    gt_class_C3_count=[lst[8] for lst in gt_difficulty_class_count]

    pred_class_A_count=[class_count[0] for class_count in pred_total_class_count]
    pred_class_B_count=[class_count[1] for class_count in pred_total_class_count]
    pred_class_C_count=[class_count[2] for class_count in pred_total_class_count]


    filtered_pred_class_A_count=[class_count[0] for class_count in pred_filtered_class_count]
    filtered_pred_class_B_count=[class_count[1] for class_count in pred_filtered_class_count]
    filtered_pred_class_C_count=[class_count[2] for class_count in pred_filtered_class_count]

    ignored_pred_class_A_count=[class_count[0] for class_count in ignored_pred_count]
    ignored_pred_class_B_count=[class_count[1] for class_count in ignored_pred_count]
    ignored_pred_class_C_count=[class_count[2] for class_count in ignored_pred_count]

    pred_class_A1_count=[lst[0] for lst in pred_difficulty_class_count]
    pred_class_A2_count=[lst[1] for lst in pred_difficulty_class_count]
    pred_class_A3_count=[lst[2] for lst in pred_difficulty_class_count]

    pred_class_B1_count=[lst[3] for lst in pred_difficulty_class_count]
    pred_class_B2_count=[lst[4] for lst in pred_difficulty_class_count]
    pred_class_B3_count=[lst[5] for lst in pred_difficulty_class_count]

    pred_class_C1_count=[lst[6] for lst in pred_difficulty_class_count]
    pred_class_C2_count=[lst[7] for lst in pred_difficulty_class_count]
    pred_class_C3_count=[lst[8] for lst in pred_difficulty_class_count]

    ΣA_gt_count=[sum(gt_class_A_count[0:i+1]) for i in range(len(gt_class_A_count))]
    ΣB_gt_count=[sum(gt_class_B_count[0:i+1]) for i in range(len(gt_class_B_count))]
    ΣC_gt_count=[sum(gt_class_C_count[0:i+1]) for i in range(len(gt_class_C_count))]

    ΣA_filtered_gt_count=[sum(filtered_gt_class_A_count[0:i+1]) for i in range(len(filtered_gt_class_A_count))]
    ΣB_filtered_gt_count=[sum(filtered_gt_class_B_count[0:i+1]) for i in range(len(filtered_gt_class_B_count))]
    ΣC_filtered_gt_count=[sum(filtered_gt_class_C_count[0:i+1]) for i in range(len(filtered_gt_class_C_count))]

    ΣA1_gt_count=[sum(gt_class_A1_count[0:i+1]) for i in range(len(gt_class_A1_count))]
    ΣA2_gt_count=[sum(gt_class_A2_count[0:i+1]) for i in range(len(gt_class_A2_count))]
    ΣA3_gt_count=[sum(gt_class_A3_count[0:i+1]) for i in range(len(gt_class_A3_count))]

    ΣB1_gt_count=[sum(gt_class_B1_count[0:i+1]) for i in range(len(gt_class_B1_count))]
    ΣB2_gt_count=[sum(gt_class_B2_count[0:i+1]) for i in range(len(gt_class_B2_count))]
    ΣB3_gt_count=[sum(gt_class_B3_count[0:i+1]) for i in range(len(gt_class_B3_count))]

    ΣC1_gt_count=[sum(gt_class_C1_count[0:i+1]) for i in range(len(gt_class_C1_count))]
    ΣC2_gt_count=[sum(gt_class_C2_count[0:i+1]) for i in range(len(gt_class_C2_count))]
    ΣC3_gt_count=[sum(gt_class_C3_count[0:i+1]) for i in range(len(gt_class_C3_count))]

    ΣA_pred_count=[sum(pred_class_A_count[0:i+1]) for i in range(len(pred_class_A_count))]
    ΣB_pred_count=[sum(pred_class_B_count[0:i+1]) for i in range(len(pred_class_B_count))]
    ΣC_pred_count=[sum(pred_class_C_count[0:i+1]) for i in range(len(pred_class_C_count))]

    ΣA1_pred_count=[sum(pred_class_A1_count[0:i+1]) for i in range(len(pred_class_A1_count))]
    ΣA2_pred_count=[sum(pred_class_A2_count[0:i+1]) for i in range(len(pred_class_A2_count))]
    ΣA3_pred_count=[sum(pred_class_A3_count[0:i+1]) for i in range(len(pred_class_A3_count))]

    ΣB1_pred_count=[sum(pred_class_B1_count[0:i+1]) for i in range(len(pred_class_B1_count))]
    ΣB2_pred_count=[sum(pred_class_B2_count[0:i+1]) for i in range(len(pred_class_B2_count))]
    ΣB3_pred_count=[sum(pred_class_B3_count[0:i+1]) for i in range(len(pred_class_B3_count))]

    ΣC1_pred_count=[sum(pred_class_C1_count[0:i+1]) for i in range(len(pred_class_C1_count))]
    ΣC2_pred_count=[sum(pred_class_C2_count[0:i+1]) for i in range(len(pred_class_C2_count))]
    ΣC3_pred_count=[sum(pred_class_C2_count[0:i+1]) for i in range(len(pred_class_C2_count))]

    

    classA1_array=np.column_stack((TP_A1_list,FP_A1_list,FN_A1_list,A1_frame_precision,A1_frame_recall,ΣTP_A1,ΣFP_A1,ΣFN_A1,ΣA1_precision,ΣA1_recall,A1_mean_precision,A1_mean_recall))
    classA1_data=pd.DataFrame(classA1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classA1_data,os.path.join(path,'Cars Easy.png'),max_rows=-1)
    classA1_data.to_excel(os.path.join(path,'Cars Easy.xlsx'))

    # print("ΣA1_precision[-1]: ",ΣA1_precision[-1])
    # print("ΣA1_recall[-1]: ",ΣA1_recall[-1])

    # print("ΣA1_recall: \n",ΣA1_recall)
    # print("len(ΣA1_recall): ",len(ΣA1_recall))
    classA2_array=np.column_stack((TP_A2_list,FP_A2_list,FN_A2_list,A2_frame_precision,A2_frame_recall,ΣTP_A2,ΣFP_A2,ΣFN_A2,ΣA2_precision,ΣA2_recall,A2_mean_precision,A2_mean_recall))
    classA2_data=pd.DataFrame(classA2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classA2_data,os.path.join(path,'Cars Moderate.png'),max_rows=-1)
    classA2_data.to_excel(os.path.join(path,'Cars Moderate.xlsx'))

    classA3_array=np.column_stack((TP_A3_list,FP_A3_list,FN_A3_list,A3_frame_precision,A3_frame_recall,ΣTP_A3,ΣFP_A3,ΣFN_A3,ΣA3_precision,ΣA3_recall,A3_mean_precision,A3_mean_recall))
    classA3_data=pd.DataFrame(classA3_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classA3_data,os.path.join(path,'Cars Moderate.png'),max_rows=-1)
    classA3_data.to_excel(os.path.join(path,'Cars Hard.xlsx'))

    classA_array=np.column_stack((TP_A_list,FP_A_list,FN_A_list,A_frame_precision,A_frame_recall,ΣTP_A,ΣFP_A,ΣFN_A,ΣA_precision,ΣA_recall,A_mean_precision,A_mean_recall))
    classA_data=pd.DataFrame(classA_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classA_data,os.path.join(path,'Cars Moderate.png'),max_rows=-1)
    classA_data.to_excel(os.path.join(path,'Cars Overall.xlsx'))

    

    classB1_array=np.column_stack((TP_B1_list,FP_B1_list,FN_B1_list,B1_frame_precision,B1_frame_recall,ΣTP_B1,ΣFP_B1,ΣFN_B1,ΣB1_precision,ΣB1_recall,B1_mean_precision,B1_mean_recall))
    classB1_data=pd.DataFrame(classB1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classB1_data,os.path.join(path,'Cyclists Easy.png'),max_rows=-1)
    classB1_data.to_excel(os.path.join(path,'Cyclists Easy.xlsx'))

    classB2_array=np.column_stack((TP_B2_list,FP_B2_list,FN_B2_list,B2_frame_precision,B2_frame_recall,ΣTP_B2,ΣFP_B2,ΣFN_B2,ΣB2_precision,ΣB2_recall,B2_mean_precision,B2_mean_recall))
    classB2_data=pd.DataFrame(classB2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classB2_data,os.path.join(path,'Cyclists Moderate.png'),max_rows=-1)
    classB2_data.to_excel(os.path.join(path,'Cyclists Moderate.xlsx'))

    classB3_array=np.column_stack((TP_B3_list,FP_B3_list,FN_B3_list,B3_frame_precision,B3_frame_recall,ΣTP_B3,ΣFP_B3,ΣFN_B3,ΣB3_precision,ΣB3_recall,B3_mean_precision,B3_mean_recall))
    classB3_data=pd.DataFrame(classB3_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classB3_data,os.path.join(path,'Cyclists Moderate.png'),max_rows=-1)
    classB3_data.to_excel(os.path.join(path,'Cyclists Hard.xlsx'))

    classB_array=np.column_stack((TP_B_list,FP_B_list,FN_B_list,B_frame_precision,B_frame_recall,ΣTP_B,ΣFP_B,ΣFN_B,ΣB_precision,ΣB_recall,B_mean_precision,B_mean_recall))
    classB_data=pd.DataFrame(classB_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classB_data,os.path.join(path,'Cars Moderate.png'),max_rows=-1)
    classB_data.to_excel(os.path.join(path,'Cyclists Overall.xlsx'))

    classC1_array=np.column_stack((TP_C1_list,FP_C1_list,FN_C1_list,C1_frame_precision,C1_frame_recall,ΣTP_C1,ΣFP_C1,ΣFN_C1,ΣC1_precision,ΣC1_recall,C1_mean_precision,C1_mean_recall))
    classC1_data=pd.DataFrame(classC1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classC1_data,os.path.join(path,'Pedestrian Easy.png'),max_rows=-1)
    classC1_data.to_excel(os.path.join(path,'Pedestrian Easy.xlsx'))

    classC2_array=np.column_stack((TP_C2_list,FP_C2_list,FN_C2_list,C2_frame_precision,C2_frame_recall,ΣTP_C2,ΣFP_C2,ΣFN_C2,ΣC2_precision,ΣC2_recall,C2_mean_precision,C2_mean_recall))
    classC2_data=pd.DataFrame(classC2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classC2_data,os.path.join(path,'Pedestrian Hard.png'),max_rows=-1)
    classC2_data.to_excel(os.path.join(path,'Pedestrian Hard.xlsx'))

    classC3_array=np.column_stack((TP_C3_list,FP_C3_list,FN_C3_list,C3_frame_precision,C3_frame_recall,ΣTP_C3,ΣFP_C3,ΣFN_C3,ΣC3_precision,ΣC3_recall,C3_mean_precision,C3_mean_recall))
    classC3_data=pd.DataFrame(classC3_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classC3_data,os.path.join(path,'Pedestrian Hard.png'),max_rows=-1)
    classC3_data.to_excel(os.path.join(path,'Pedestrian Hard.xlsx'))

    classC_array=np.column_stack((TP_C_list,FP_C_list,FN_C_list,C_frame_precision,C_frame_recall,ΣTP_C,ΣFP_C,ΣFN_C,ΣC_precision,ΣC_recall,C_mean_precision,C_mean_recall))
    classC_data=pd.DataFrame(classC_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    #dfi.export(classC_data,os.path.join(path,'Cars Moderate.png'),max_rows=-1)
    classC_data.to_excel(os.path.join(path,'Pedestrians Overall.xlsx'))

    #table_data=table_data.style.set_table_styles([{"selector":"thead","props":"font-size:20pt"}])
#     table_data = table_data.data.style.set_properties(**{
#     'font-size': '30pt',
# })
    #
    # Checks

    # All Data in Table + Checks
    A_array=np.concatenate((classA1_array,classA2_array,classA3_array,classA_array),axis=1)#,classB1_array,classB2_array,classC1_array,classC2_array))
    A_array_with_checks=np.column_stack((A_array,gt_class_A_count,filtered_gt_class_A_count,gt_class_A1_count,gt_class_A2_count,gt_class_A3_count,ignored_gt_class_A_count,pred_class_A_count,filtered_pred_class_A_count,pred_class_A1_count,pred_class_A2_count,pred_class_A3_count,ignored_pred_class_A_count,ΣA_gt_count,ΣA_filtered_gt_count,ΣA1_gt_count,ΣA2_gt_count,ΣA3_gt_count,ΣA_pred_count,ΣA1_pred_count,ΣA2_pred_count,ΣA3_pred_count))
    A_data=pd.DataFrame(A_array_with_checks,columns=field_name_cars)
    A_data.to_excel(os.path.join(path,'Cars.xlsx'),sheet_name='metrics')
    #dfi.export(A_data,os.path.join(path,'Cars.png'),max_rows=-1,max_cols=-1)

    # Customize Worksheet

    enhance_excelfile(A_data,path,'Enhanced_Cars.xlsx')


    # writer = pd.ExcelWriter(os.path.join(path,'enhanced_Cars.xlsx'), engine='xlsxwriter')
    # A_data.to_excel(writer, index=False, sheet_name='metrics')
    # #
    # workbook = writer.book
    # worksheet = writer.sheets['metrics']


    # #Set header formating
    # header_format = workbook.add_format({
    #         "valign": "vcenter",
    #         "align": "center",
    #         "bg_color": "#951F06",
    #         "bold": True,
    #         'font_color': '#FFFFFF'
    #     })

    

    # #add title
    # #merge cells
    # # format = workbook.add_format()
    # # format.set_font_size(30)
    # # format.set_font_color("#333333")
    # #worksheet.merge_range('A1:AS1', title, format)
    # # puting it all together
    # # Write the column headers with the defined format.
    # # for col_num, col_value in enumerate(A_data.columns.values):
    # #     #print(col_num, value)
    # #     #for row_num,row_value in enumerate(A_data.rows.values):
    # #     worksheet.write(4, col_num, col_value)

    # easy_cols=['A','B','C']
    # moderate_cols=['M','N','O']
    # hard_cols=['Y','Z','AA']
 

    # total_cols=easy_cols+moderate_cols+hard_cols

    # easy_format = workbook.add_format({'bg_color': 'blue'})
    # moderate_format=workbook.add_format({'bg_color': 'purple'})
    # hard_format=workbook.add_format({'bg_color': 'orange'})

    # gt_stats_format=workbook.add_format({'bg_color': 'green'})
    # pred_stats_format=workbook.add_format({'bg_color': 'red'})
    
    # worksheet.set_column('A1:C1', None, easy_format)
    # worksheet.set_column(12, 14, None, moderate_format)
    # worksheet.set_column(24, 26, None, hard_format)

    # worksheet.set_column(36, 41, None, gt_stats_format)
    # worksheet.set_column(42, 47, None, pred_stats_format)




    # # Adjust the column width.
    # worksheet.set_column('A:BF', 30)



    # writer.save()





    # All Data in Table + Checks
    B_array=np.concatenate((classB1_array,classB2_array,classB3_array,classB_array),axis=1)#,classB1_array,classB2_array,classC1_array,classC2_array))
    B_array_with_checks=np.column_stack((B_array,gt_class_B_count,filtered_gt_class_B_count,gt_class_B1_count,gt_class_B2_count,gt_class_B3_count,ignored_gt_class_B_count,pred_class_B_count,filtered_pred_class_B_count,pred_class_B1_count,pred_class_B2_count,pred_class_B3_count,ignored_pred_class_B_count,ΣB_gt_count,ΣB_filtered_gt_count,ΣB1_gt_count,ΣB2_gt_count,ΣB3_gt_count,ΣB_pred_count,ΣB1_pred_count,ΣB2_pred_count,ΣB3_pred_count))
    B_data=pd.DataFrame(B_array_with_checks,columns=field_name_cyclists)
    B_data.to_excel(os.path.join(path,'Cyclists.xlsx'))
    #dfi.export(A_data,os.path.join(path,'Cars.png'),max_rows=-1,max_cols=-1)

    enhance_excelfile(B_data,path,'Enhanced_Cyclists.xlsx')

    # All Data in Table + Checks
    C_array=np.concatenate((classC1_array,classC2_array,classC3_array,classC_array),axis=1)#,classC1_array,classC2_array,classC1_array,classC2_array))
    C_array_with_checks=np.column_stack((C_array,gt_class_C_count,filtered_gt_class_C_count,gt_class_C1_count,gt_class_C2_count,gt_class_C3_count,ignored_gt_class_C_count,pred_class_C_count,filtered_pred_class_C_count,pred_class_C1_count,pred_class_C2_count,pred_class_C3_count,ignored_pred_class_C_count,ΣC_gt_count,ΣC_filtered_gt_count,ΣC1_gt_count,ΣC2_gt_count,ΣC3_gt_count,ΣC_pred_count,ΣC1_pred_count,ΣC2_pred_count,ΣC3_pred_count))
    C_data=pd.DataFrame(C_array_with_checks,columns=field_name_pedestrians)
    C_data.to_excel(os.path.join(path,'Pedestrians.xlsx'))

    enhance_excelfile(C_data,path,'Enhanced_Pedestrians.xlsx')


    # B_array=np.concatenate((classB1_array,classB2_array),axis=1)#,classB1_array,classB2_array,classC1_array,classC2_array))
    # B_array_with_checks=np.column_stack((B_array,gt_class_B_count,gt_class_B1_count,gt_class_B2_count,pred_class_B_count,pred_class_B1_count,pred_class_B2_count,ΣB_gt_count,ΣB1_gt_count,ΣB2_gt_count,ΣB_pred_count,ΣB1_pred_count,ΣB2_pred_count))
    # B_data=pd.DataFrame(B_array_with_checks,columns=field_name_cyclists)
    # B_data.to_excel(os.path.join(path,'Cyclists.xlsx'))
    # #dfi.export(B_data,os.path.join(path,'Cyclists.png'),max_rows=-1,max_cols=-1)

    # C_array=np.concatenate((classC1_array,classC2_array),axis=1)#,classB1_array,classB2_array,classC1_array,classC2_array))
    # C_array_with_checks=np.column_stack((C_array,gt_class_C_count,gt_class_C1_count,gt_class_C2_count,pred_class_B_count,pred_class_C1_count,pred_class_C2_count,ΣC_gt_count,ΣC1_gt_count,ΣC2_gt_count,ΣC_pred_count,ΣC1_pred_count,ΣC2_pred_count))
    # C_data=pd.DataFrame(C_array_with_checks,columns=field_name_pedestrians)
    # C_data.to_excel(os.path.join(path,'Pedestrians.xlsx'))
    # #dfi.export(C_data,os.path.join(path,'Pedestrians.png'),max_rows=-1,max_cols=-1)

    car_metrics=[ΣA1_precision[-1],ΣA1_recall[-1],ΣA2_precision[-1],ΣA2_recall[-1],ΣA3_precision[-1],ΣA3_recall[-1],ΣA_precision[-1],ΣA_recall[-1]]
    pedestrian_metrics=[ΣC1_precision[-1],ΣC1_recall[-1],ΣC2_precision[-1],ΣC2_recall[-1],ΣC3_precision[-1],ΣC3_recall[-1],ΣC_precision[-1],ΣC_recall[-1]]
    cyclist_metrics=[ΣC1_precision[-1],ΣC1_recall[-1],ΣC2_precision[-1],ΣC2_recall[-1],ΣC3_precision[-1],ΣC3_recall[-1],ΣC_precision[-1],ΣC_recall[-1]]
    difficulty_metrics=[Σeasy_precision[-1],Σeasy_recall[-1],Σmoderate_precision[-1],Σmoderate_recall[-1],Σhard_precision[-1],Σhard_recall[-1]]

    #print("Car Metrics: ",car_metrics)
    easy_objects_count=ΣA1_gt_count[-1]+ΣC1_gt_count[-1]

    moderate_objects_count=ΣA2_gt_count[-1]+ΣC2_gt_count[-1]

    hard_objects_count=ΣA3_gt_count[-1]+ΣC3_gt_count[-1]

    overall_difficulty_objects_count=ΣA_filtered_gt_count[-1]+ΣC_filtered_gt_count[-1]
    #ΣA_gt_count[-1]+ΣC_gt_count[-1]

    n_objects_difficulty=[easy_objects_count,moderate_objects_count,hard_objects_count,overall_difficulty_objects_count]
    n_objects_classes=[ΣA1_gt_count[-1],ΣA2_gt_count[-1],ΣA3_gt_count[-1],ΣA_filtered_gt_count[-1],ΣC1_gt_count[-1],ΣC2_gt_count[-1],ΣC3_gt_count[-1],ΣC_filtered_gt_count[-1]]

    return car_metrics,pedestrian_metrics,cyclist_metrics,difficulty_metrics,n_objects_classes,n_objects_difficulty

def enhance_excelfile(class_data,path,enhanced_filename):

    writer = pd.ExcelWriter(os.path.join(path,enhanced_filename), engine='xlsxwriter')
    class_data.to_excel(writer, index=False, sheet_name='metrics')
    #
    workbook = writer.book
    worksheet = writer.sheets['metrics']

    # Adjust the column width.
    worksheet.set_column('A:BQ', 30)


    #Set header formating
    header_format = workbook.add_format({
            "valign": "vcenter",
            "align": "center",
            "bg_color": "#951F06",
            "bold": True,
            'font_color': '#FFFFFF'
        })

    



    easy_cols=['A','B','C']
    moderate_cols=['M','N','O']
    hard_cols=['Y','Z','AA']
 

    total_cols=easy_cols+moderate_cols+hard_cols

    easy_format = workbook.add_format({'bg_color': 'blue'})
    moderate_format=workbook.add_format({'bg_color': 'purple'})
    hard_format=workbook.add_format({'bg_color': 'orange'})

    gt_stats_format=workbook.add_format({'bg_color': 'green'})
    pred_stats_format=workbook.add_format({'bg_color': 'red'})
    
    worksheet.set_column('A1:C1', 30, easy_format)
    worksheet.set_column('M1:O1', 30, moderate_format)
    worksheet.set_column('Y1:AA1', 30, hard_format)

    worksheet.set_column('AK1:AP1', 30, gt_stats_format)
    worksheet.set_column('AQ1:AV1', 30, pred_stats_format)




    # Adjust the column width.
    worksheet.set_column('A:BQ', 30)



    writer.save()
    

def tabularize(TP_A1_list,FP_A1_list,FN_A1_list,table_name):
    
    ΣTP_A1=[sum(TP_A1_list[0:i+1]) for i in range(len(TP_A1_list))]
    ΣFP_A1=[sum(FP_A1_list[0:i+1]) for i in range(len(FP_A1_list))]
    ΣFN_A1=[sum(FN_A1_list[0:i+1]) for i in range(len(FN_A1_list))]



    ΣA1_precision=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFP_A1[i])  if (ΣTP_A1[i]+ΣFP_A1[i]) is not 0 else 0 for i in range(len(TP_A1_list))]
    ΣA1_recall=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFN_A1[i]) if (ΣTP_A1[i]+ΣFN_A1[i]) is not 0 else 0 for i in range(len(TP_A1_list))  ]


    A1_frame_precision=[TP_A1_list[i]/(TP_A1_list[i]+FP_A1_list[i]) if (TP_A1_list[i]+FP_A1_list[i]) is not 0 else 0 for i in range(len(TP_A1_list))]
    A1_mean_precision=[sum(A1_frame_precision[0:i+1])/len(A1_frame_precision[0:i+1]) for i in range(len(TP_A1_list))]
    A1_frame_recall=[TP_A1_list[i]/(TP_A1_list[i]+FN_A1_list[i]) if (TP_A1_list[i]+FN_A1_list[i]) is not 0 else 0 for i in range(len(TP_A1_list))]
    A1_mean_recall=[sum(A1_frame_recall[0:i+1])/len(A1_frame_recall[0:i+1]) for i in range(len(TP_A1_list))]


    classA1_array=np.column_stack((TP_A1_list,FP_A1_list,FN_A1_list,A1_frame_precision,A1_frame_recall,ΣTP_A1,ΣFP_A1,ΣFN_A1,ΣA1_precision,ΣA1_recall,A1_mean_precision,A1_mean_recall))
    classA1_data=pd.DataFrame(classA1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classA1_data,"/home/hasan/perception-validation-verification/unit_test_figs/"+table_name+".png")




def get_gt_classes_boxes(groundtruth):
    gt_classes=[groundtruth_detection_list[0]  for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare']
    gt_boxs=[(Point(float(groundtruth_detection_list[4]),float(groundtruth_detection_list[5])),Point(float(groundtruth_detection_list[6]),float(groundtruth_detection_list[7]))) for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare' ]

    return gt_classes,gt_boxs

def get_gt_truncation_occlusion(groundtruth):
    gt_truncation=[float(groundtruth_detection_list[1])  for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare']
    gt_occlusion=[int(groundtruth_detection_list[2])  for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare']
    
    

    return gt_truncation,gt_occlusion

def get_pred_classes_boxes(smoke_predictions_list):
    pred_classes=[yolo_get_key(prediction_list[0]) if type(smoke_predictions_list[0])==list else smoke_predictions_list[0] for prediction_list in smoke_predictions_list]
    pred_boxs=[(Point(float(prediction_list[2]),float(prediction_list[3])),Point(float(prediction_list[4]),float(prediction_list[5])))  if type(smoke_predictions_list[0])==list else (Point(float(smoke_predictions_list[2]),float(smoke_predictions_list[3])),Point(float(smoke_predictions_list[4]),float(smoke_predictions_list[5])))  for prediction_list in smoke_predictions_list ]

    
    return pred_classes,pred_boxs

def smoke_get_n_classes(predictions_list):
    n_cars=0
    n_bikes=0
    n_pedestrians=0

    for i in range(len(predictions_list)):
        if predictions_list[i][0]==0.0:
            n_cars+=1
        elif predictions_list[i][0]==1.0:
            n_bikes+=1
        elif predictions_list[i][0]==2.0:
            n_pedestrians+=1

    return (n_cars,n_pedestrians,n_bikes)

def yolo_get_n_classes(predictions_list):
    n_cars=0
    n_bikes=0
    n_pedestrians=0
    for i in range(len(predictions_list)):
        if predictions_list[i]=='car' or predictions_list[i]=='truck' or predictions_list[i]=='bus':
            n_cars+=1
        elif predictions_list[i]=='bicycle' or  predictions_list[i]=='motorbike' :
            n_bikes+=1
        elif predictions_list[i]=='person':
            n_pedestrians+=1
        else:
            pass

    n_detections=n_cars+n_bikes+n_pedestrians

    return (n_detections,n_cars,n_pedestrians,n_bikes)


# # def precision(predictions,ground_truth):
# #     pass

# Python program to check if rectangles overlap
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



 
# Returns true if two rectangles(l1, r1)
# and (l2, r2) overlap
def rectanges_overlap(l1, r1, l2, r2):
    overlap_condition=None
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        #print('False 1')
        overlap_condition=1

        return False,overlap_condition
     
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        #print('False 2')
        overlap_condition=2
        return False,overlap_condition
 
    # If one rectangle is above other
    if r1.y < l2.y or r2.y < l1.y:
        # print('L1.Y: ',l1.y)
        # print('R2.Y: ',r2.y)

        # print('L2.Y: ',l2.y)
        # print('R1.Y: ',r1.y)
        overlap_condition=3

        #print('False 3')
        return False,overlap_condition
 
    #print('rectangles overlap')
    return True,overlap_condition

def boxes_in_list_do_not_intersect(l1,r1,l2,r2):
    overlap_condition=None
     
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        #print('False 1')
        overlap_condition=1

        return False,overlap_condition
     
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        #print('False 2')
        overlap_condition=2
        return False,overlap_condition
 
    # If one rectangle is above other
    if r1.y < l2.y or r2.y < l1.y:
        # print('L1.Y: ',l1.y)
        # print('R2.Y: ',r2.y)

        # print('L2.Y: ',l2.y)
        # print('R1.Y: ',r1.y)
        overlap_condition=3

        print('False 3')
        return False,overlap_condition
 
    print('rectangles pverlap')
    return True,overlap_condition


def get_IoU(box1,box2):

    # box (Top Left X,Top Left Y,Bottom Right X,Bottom Right Y)

    Pt1=box1[0]
    Pt2=box1[1]

    Pt3=box2[0]
    Pt4=box2[1]


    # l1=Point(x1,y1)
    # r1=Point(x2,y2)

    # l2=Point(x3,y3)
    # r2=Point(x4,y4)


    #print('IoU input in function: ',(Pt1,Pt2,Pt3,Pt4))
    boxs_overlap,overlap_condition=rectanges_overlap(Pt1,Pt2,Pt3,Pt4)
    #print('IoU boxs overlap in function: ',boxs_overlap)
    if boxs_overlap==True:


        x_inter1=max(Pt1.x,Pt3.x)
        y_inter1=max(Pt1.y,Pt3.y)

        x_inter2=min(Pt2.x,Pt4.x)
        y_inter2=min(Pt2.y,Pt4.y)

        # print('(Y2,Y4): ',(Pt2.y,Pt4.y))

        # print('Intersection Point 1: ',(x_inter1,y_inter1))
        # print('Intersection Point 2: ',(x_inter2,y_inter2))

        width_inter=abs(x_inter2-x_inter1)
        height_inter=abs(y_inter2-y_inter1)

        area_inter=width_inter*height_inter

        # print('Width Intersection: ',width_inter)
        # print('Height Intersection: ',height_inter)

        

        width_box1=abs(Pt2.x-Pt1.x)
        height_box1=abs(Pt2.y-Pt1.y)

        # print('Width Box 1: ',width_box1)
        # print('Height Box 1: ',height_box1)

        width_box2=abs(Pt4.x-Pt3.x)
        height_box2=abs(Pt4.y-Pt3.y)

        area_box1=width_box1*height_box1
        area_box2=width_box2*height_box2

        # print('Area Box 1: ',area_box1)
        # print('Area Box 2: ',area_box2)


        area_union=area_box1+area_box2-area_inter

        # print('Area Intersection: ',area_inter)
        # print('Area Union: ',area_union)

        iou=area_inter/area_union

    else:
        # print('Rectangles Do not intersect')
        iou=0
    
    return iou

def get_key(val):
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }
    for key, value in TYPE_ID_CONVERSION.items():
        if val == value:
            return key
 
    return "DontCare"


def get_evaluation_metrics(groundtruth,predictions):
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }

    TP=0
    for i in range(len(groundtruth)):
        true_pt1=Point(float(groundtruth[i][3]),float(groundtruth[i][2]))
        true_pt2=Point(float(groundtruth[i][5]),float(groundtruth[i][4]))
        true_class=groundtruth[i][1]

        for j in range(len(predictions)):

            pred_pt1=Point(predictions[j][2],predictions[j][3])
            pred_pt2=Point(predictions[j][4],predictions[j][5])
            pred_class_ID=predictions[j][0]
            pred_class_string=get_key(pred_class_ID,TYPE_ID_CONVERSION)

            print('Pred',(pred_pt1,pred_pt2))
            print('Truth', (true_pt1,true_pt2))
            IoU=get_IoU((pred_pt1,pred_pt2),(true_pt1,true_pt2))

            if true_class=='Car':

                if IoU>=0.7 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass
            
            elif true_class=='Cyclist':
                if IoU>=0.5 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass

            elif true_class=='Pedestrian':
                if IoU>=0.5 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass

            else:
                pass

    return TP


def plot_groundtruth(img,groundtruth):
    thickness=2
    groundtruth_img=img
    gt_classes,gt_boxs=get_gt_classes_boxes(groundtruth)
    easy_indix_list=get_box_difficulty_index(gt_boxs,gt_classes,'Car','Easy')
    moderate_indix_list=get_box_difficulty_index(gt_boxs,gt_classes,'Car','Moderate')
    for i in range(len(groundtruth)):

        if groundtruth[i][0]=='Car'or groundtruth[i][0]=='Cyclist' or groundtruth[i][0]=='Pedestrian':
            if groundtruth[i][0]=='Car':

                if i in easy_indix_list:
                    color=(0,255,0)
                else:
                    color=(0,122,0)
            else:
                color=(0,255,0)

        elif groundtruth[i][0]=='DontCare':
            color=(255,255,255)


        else:
            color=(0,255,255)

        TL=(int(float(groundtruth[i][4])),int(float(groundtruth[i][5])))
        BR=(int(float(groundtruth[i][6])),int(float(groundtruth[i][7])))

        groundtruth_img=cv2.rectangle(groundtruth_img,TL,BR,color,thickness)

    
    return groundtruth_img

def plot_prediction(img,predictions):
    thickness=2
    prediction_img=img
    for i in range(len(predictions)):

        color=(0,0,255)
        TL=(int(float(predictions[i][2])),int(float(predictions[i][3])))
        BR=(int(float(predictions[i][4])),int(float(predictions[i][5])))

        prediction_img=cv2.rectangle(prediction_img,TL,BR,color,thickness)

    
    return prediction_img

class metrics_evaluator:
    def __init__(self,n_frames,results_path):
        self.labels=['Car','Cyclist','Pedestrian']
        self.difficulties=['Easy','Moderate','Hard']
        self.data=[]#np.zeros((n_frames ,len(self.difficulties)*len(self.labels)*3))
        self.frame_id=0
        self.n_frames=n_frames
        self.gt_class_difficulty_count=[]
        self.gt_class_total_count=[]
        self.pred_class_difficulty_count=[]
        self.pred_class_total_total_count=[]
        self.results_path=results_path

        

    def get_boxs_classes_occlusion_truncation(self,groundtruth,smoke_predictions_list):
        self.gt_classes,self.gt_boxs=get_gt_classes_boxes(groundtruth)
        self.pred_classes,self.pred_boxs=get_pred_classes_boxes(smoke_predictions_list)
        self.gt_truncation,self.gt_occlusion=get_gt_truncation_occlusion(groundtruth)

    def get_metrics(self):
        data_row=[]
        gt_class_difficulty_count_row=[]
        gt_class_total_count_per_frame=[]
        pred_class_difficulty_count=[]
        pred_class_total_count_per_frame=[]
        for l in self.labels:
            gt_class_total_count_per_frame.append(self.gt_classes.count(l))
            pred_class_total_count_per_frame.append(self.pred_classes.count(l))
        self.gt_class_total_count.append(gt_class_total_count_per_frame)
        self.pred_class_total_total_count.append(pred_class_total_count_per_frame)



        for label in self.labels:
            for difficulty in self.difficulties:
                class_gt_instances,class_pred_instances,TP,FP,FN=get_metrics_label_specific_2(label,difficulty,self.labels,self.gt_classes,self.pred_classes,self.gt_boxs,self.pred_boxs,self.gt_truncation,self.gt_occlusion)
                data_row.extend([TP,FP,FN])
                gt_class_difficulty_count_row.append(class_gt_instances)
                pred_class_difficulty_count.append(class_pred_instances)

        
        self.data.append(data_row)
        self.gt_class_difficulty_count.append(gt_class_difficulty_count_row)
        self.pred_class_difficulty_count.append(pred_class_difficulty_count)

        self.frame_id+=1

    def tabularize(self):
        if self.frame_id==self.n_frames:
            self.data=np.array(self.data)
            print('Data: ',self.data)
            print('Data Array Size',self.data.shape)
            tabularize_metrics(self.data,self.gt_class_total_count,self.gt_class_difficulty_count,self.pred_class_total_total_count,self.pred_class_difficulty_count,self.n_frames,self.results_path)
        else:
            pass

    def evaluate_metrics(self,groundtruth,smoke_predictions_list):
        self.get_boxs_classes_occlusion_truncation(groundtruth,smoke_predictions_list)
        self.get_metrics()
        self.tabularize()


def yolo_2_smoke_output_format(boxs,classes,scores):

    output_per_frame=[]

    for label,box,score in zip(classes,boxs,scores):
        if label=='car' or label=='truck':
            class_id=0

        elif label=='bicycle':
            class_id=1

        elif label=='person':
            class_id=2
        
        else:
            class_id=-1

        x1,y1,x2,y2=yolobbox2bbox(box[0],box[1],box[2],box[3])


        output_per_frame.append([class_id,-9,x1,y1,x2,y2,-9,-9,-9,-9,-9,-9,-9,score])

    return output_per_frame


def modified_yolo_2_smoke_output_format(boxs,classes):

    output_per_frame=[]

    for label,box in zip(classes,boxs):
        if label=='car' or label=='truck':
            class_id=0

        elif label=='bicycle':
            class_id=1

        elif label=='person':
            class_id=2
        
        else:
            class_id=-1

        x1,y1,x2,y2=yolobbox2bbox(box[0],box[1],box[2],box[3])


        output_per_frame.append([class_id,-9,x1,y1,x2,y2,-9,-9,-9,-9,-9,-9,-9,-9])

    return output_per_frame

    

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x, y
    x2, y2 = x+w, y+h
    return x1, y1, x2, y2


def convert_prediction_text_format(prediction):
    class_in_text_format=get_key(int(prediction[0]))

    data=[class_in_text_format,
                0,
                0,
                prediction[1],
                prediction[2],
                prediction[3],
                prediction[4],
                prediction[5],
                prediction[6],
                prediction[7],
                prediction[8],
                prediction[9],
                prediction[10],
                prediction[11],
                prediction[12],
                prediction[13]
                ]
    return data




def read_prediction(predictions_folder_path,fileid):

    TYPE_ID_CONVERSION = {
                        'Car': 0,
                        'Cyclist': 1,
                        'Pedestrian': 2,
                        'DontCare' : -1
                        }
    with open(os.path.join(predictions_folder_path,str(fileid).zfill(6)+'.txt'),'r') as f:
        reader=csv.reader(f)
        rows=list(reader)


    smoke_predictions_read_from_file=[[float(row[0].split(' ')[i])  if i!=0 else TYPE_ID_CONVERSION[row[0].split(' ')[i] ]for i in [0]+list(range(3,len(row[0].split(' '))))] for row in rows]

    return smoke_predictions_read_from_file


def write_prediction(predictions_folder_path,fileid,predictions_list):
    with open(os.path.join(predictions_folder_path,str(fileid).zfill(6)+'.txt'),'w') as f:
        writer=csv.writer(f,delimiter=' ')

        
        for j,prediction in enumerate(predictions_list):
            print('Prediction: ',prediction)
            print('Items in Prediction: ',len(prediction))
            writer.writerow(convert_prediction_text_format(prediction))

    return True


def get_class_AP(results_path,label):

    if label=='Car':
        AP_file='plot/car_detection.txt'

    elif label=='Pedestrian':
        AP_file='plot/pedestrian_detection.txt'

    else:
        pass

    try:

        with open(os.path.join(results_path,AP_file),'r') as f:
            reader=csv.reader(f)
            data=list(reader)


        data=[row[0].split(' ') for row in data]

        column1=[float(row[0]) for row in data]
        column2=[float(row[1]) for row in data]
        column3=[float(row[2]) for row in data]
        column4=[float(row[3]) for row in data]

        class_easy_AP=100*sum(column2[1:])/40
        class_moderate_AP=100*sum(column3[1:])/40
        class_hard_AP=100*sum(column4[1:])/40
    
    except:
        print('No {}s Detected'.format(label))
        class_easy_AP=0#100*sum(column2[1:])/40
        class_moderate_AP=0#100*sum(column3[1:])/40
        class_hard_AP=0#100*sum(column4[1:])/40


    return class_easy_AP,class_moderate_AP,class_hard_AP



def construct_dataframe(cars_AP,pedestrians_AP,car_metrics,pedestrian_metrics,difficulty_metrics,n_objects_classes,n_objects_difficulties):

    cars_easy_AP,cars_moderate_AP,cars_hard_AP=cars_AP[0],cars_AP[1],cars_AP[2]
    pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=pedestrians_AP[0],pedestrians_AP[1],pedestrians_AP[2]

    # easy_car_metrics=
    # moderate_car_metrics=
    # hard_car_metrics=


    # easy_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_easy_AP,car_metrics[0],car_metrics[0])
    # moderate_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_easy_AP,car_metrics[2],car_metrics[3])
    # hard_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_hard_AP,car_metrics[4],car_metrics[5])
    # overall_cars='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # cars=[easy_cars,moderate_cars,hard_cars,overall_cars]


    # easy_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_easy_AP,pedestrian_metrics[0],pedestrian_metrics[1])
    # moderate_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_easy_AP,pedestrian_metrics[2],pedestrian_metrics[3])
    # hard_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_hard_AP,pedestrian_metrics[4],pedestrian_metrics[5])
    # overall_pedestrians='AP: {} - Precision: {} - Recall: {}        '.format(0,0,0)
    # pedestrians=[easy_pedestrians,moderate_pedestrians,hard_pedestrians,overall_pedestrians]

    # easy_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # moderate_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # hard_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # overall_classes='AP: {} - Precision: {} - Recall: {}        '.format(0,0,0)

    if n_objects_classes[3]!=0:

        ratio_easy_cars_to_total_cars=n_objects_classes[0]/n_objects_classes[3]
        ratio_moderate_cars_to_total_cars=n_objects_classes[1]/n_objects_classes[3]
        ratio_hard_cars_to_total_cars=n_objects_classes[2]/n_objects_classes[3]
        

    else:

        ratio_easy_cars_to_total_cars=0
        ratio_moderate_cars_to_total_cars=0
        ratio_hard_cars_to_total_cars=0


    if n_objects_classes[7]!=0:
        ratio_easy_pedestrians_to_total_pedestrians=n_objects_classes[4]/n_objects_classes[7]
        ratio_moderate_pedestrians_to_total_pedestrians=n_objects_classes[5]/n_objects_classes[7]
        ratio_hard_pedestrians_to_total_pedestrians=n_objects_classes[6]/n_objects_classes[7]
    else:
        ratio_easy_pedestrians_to_total_pedestrians=0
        ratio_moderate_pedestrians_to_total_pedestrians=0
        ratio_hard_pedestrians_to_total_pedestrians=0



    overall_cars_AP=(cars_easy_AP*ratio_easy_cars_to_total_cars) +(cars_moderate_AP*ratio_moderate_cars_to_total_cars)+(cars_hard_AP*ratio_hard_cars_to_total_cars)
    overall_predestrians_AP=(pedestrian_easy_AP*ratio_easy_pedestrians_to_total_pedestrians)+(pedestrian_moderate_AP*ratio_moderate_pedestrians_to_total_pedestrians)+(pedestrian_hard_AP*ratio_hard_pedestrians_to_total_pedestrians)

    easy_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_easy_AP,car_metrics[0]*100,car_metrics[1]*100)
    moderate_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_moderate_AP,car_metrics[2]*100,car_metrics[3]*100)
    hard_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_hard_AP,car_metrics[4]*100,car_metrics[5]*100)
    overall_cars=' {:.2f} - {:.2f} - {:.2f} '.format(overall_cars_AP,car_metrics[6]*100,car_metrics[7]*100)
    cars=[easy_cars,moderate_cars,hard_cars,overall_cars,n_objects_classes[3]]


    easy_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_easy_AP,pedestrian_metrics[0]*100,pedestrian_metrics[1]*100)
    moderate_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_moderate_AP,pedestrian_metrics[2]*100,pedestrian_metrics[3]*100)
    hard_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_hard_AP,pedestrian_metrics[4]*100,pedestrian_metrics[5]*100)
    overall_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(overall_predestrians_AP,pedestrian_metrics[6]*100,pedestrian_metrics[7]*100)
    pedestrians=[easy_pedestrians,moderate_pedestrians,hard_pedestrians,overall_pedestrians,n_objects_classes[7]]


    if n_objects_difficulties[0]!=0:


        ratio_easy_cars_to_easy_objects=n_objects_classes[0]/n_objects_difficulties[0]
        ratio_easy_pedestrians_to_easy_objects=n_objects_classes[4]/n_objects_difficulties[0]

    else:
        ratio_easy_cars_to_easy_objects=0
        ratio_easy_pedestrians_to_easy_objects=0

    

    if n_objects_difficulties[1]!=0:


        ratio_moderate_cars_to_moderate_objects=n_objects_classes[1]/n_objects_difficulties[1]
        ratio_moderate_pedestrians_to_moderate_objects=n_objects_classes[5]/n_objects_difficulties[1]

    else:
        ratio_moderate_cars_to_moderate_objects=0
        ratio_moderate_pedestrians_to_moderate_objects=0

    

    if n_objects_difficulties[2]!=0:

        ratio_hard_cars_to_hard_objects=n_objects_classes[2]/n_objects_difficulties[2]
        ratio_hard_pedestrians_to_hard_objects=n_objects_classes[6]/n_objects_difficulties[2]

    else:
        ratio_hard_cars_to_hard_objects=0
        ratio_hard_pedestrians_to_hard_objects=0

    AP_weighted_average_easy=(cars_easy_AP*ratio_easy_cars_to_easy_objects) + (pedestrian_easy_AP*ratio_easy_pedestrians_to_easy_objects)
    AP_weighted_average_moderate=(cars_moderate_AP*ratio_moderate_cars_to_moderate_objects) + (pedestrian_moderate_AP*ratio_moderate_pedestrians_to_moderate_objects)
    AP_weighted_average_hard=(cars_hard_AP*ratio_hard_cars_to_hard_objects) + (pedestrian_hard_AP*ratio_hard_pedestrians_to_hard_objects)

    easy_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_easy,difficulty_metrics[0]*100,difficulty_metrics[1]*100)
    moderate_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_moderate,difficulty_metrics[2]*100,difficulty_metrics[3]*100)
    hard_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_hard,difficulty_metrics[4]*100,difficulty_metrics[5]*100)


    ratio_cars_to_total_objects=n_objects_classes[3]/(n_objects_classes[7]+n_objects_classes[3])
    ratio_pedestrians_to_total_objects=n_objects_classes[7]/(n_objects_classes[7]+n_objects_classes[3])
    overall_precision=(car_metrics[6]*ratio_cars_to_total_objects)+ (pedestrian_metrics[6]*ratio_pedestrians_to_total_objects)
    overall_recall=(car_metrics[7]*ratio_cars_to_total_objects)+ (pedestrian_metrics[7]*ratio_pedestrians_to_total_objects)
    overall_AP=(overall_cars_AP*ratio_cars_to_total_objects)+(overall_predestrians_AP*ratio_pedestrians_to_total_objects)





    overall_classes=' {:.2f} - {:.2f} - {:.2f} '.format(overall_AP,overall_precision*100,overall_recall*100)
    classes=[easy_classes,moderate_classes,hard_classes,overall_classes,(n_objects_classes[7]+n_objects_classes[3])]

    column_headers=['Class/Difficulty','Easy','Moderate','Hard','Overall Difficulties','Class # Objects']
    row_headers=['Car','Pedestrian','Cars and Pedestrians','Difficulty # Objects']

    # data=np.zeros((3,4))

    ar1=np.array(cars)
    ar2=np.array(pedestrians)
    ar3=np.array(classes)
    ar4=np.array([n_objects_difficulties[0],n_objects_difficulties[1],n_objects_difficulties[2],n_objects_difficulties[3],'Ø'])

    data=np.row_stack((ar1,ar2,ar3,ar4))

    ARRAY=np.column_stack((row_headers,data))

    ARRAY=np.row_stack((column_headers,ARRAY))

    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('colheader_justify', 'center')

    df=pd.DataFrame(ARRAY)

    df.columns = df.iloc[0]
    df = df[1:]

    # dfStyler = df.style.set_properties(**{'text-align': 'center'})
    # dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    # [dict(selector='th', props=[('text-align', 'center')])]
    
    df1 = df.style.set_table_styles([{"selector": "", "props": [("border", "1px solid")]},
        {"selector": "tbody td", "props": [("border", "1px solid")]},
        {"selector": "th", "props": [("border", "1px solid")]},
        {"selector": "th", "props": [("text-align", "center")]},
        ])
    df1.set_caption('Object Detector Metrics Evaluation [Average Precision - Precision - Recall ]')
    df1.set_properties(**{'text-align': 'center'}).hide_index()
    #df1.style.set_table_attributes("style='display:inline'").set_caption('Caption table')

    # df2=df1.style.set_table_styles(
    #     [{"selector": "", "props": [("border", "1px solid")]},
    #       {"selector": "tbody td", "props": [("border", "1px solid")]},
    #      {"selector": "th", "props": [("border", "1px solid")]}])


   
    print(df)


    # AP - Precision - Recall

    metrics_results={
              'Easy Cars':[cars_easy_AP,car_metrics[0]*100,car_metrics[1]*100],
              'Moderate Cars':[cars_moderate_AP,car_metrics[2]*100,car_metrics[3]*100],
              'Hard Cars':[cars_hard_AP,car_metrics[4]*100,car_metrics[5]*100],
              'Overall Cars':[overall_cars_AP,car_metrics[6]*100,car_metrics[7]*100],
              'Pedestrians Easy':[pedestrian_easy_AP,pedestrian_metrics[0]*100,pedestrian_metrics[1]*100],
              'Pedestrians Moderate':[pedestrian_moderate_AP,pedestrian_metrics[2]*100,pedestrian_metrics[3]*100],
              'Pedestrians Hard':[pedestrian_hard_AP,pedestrian_metrics[4]*100,pedestrian_metrics[5]*100],
              'Overall Pedestrians':[overall_predestrians_AP,pedestrian_metrics[6]*100,pedestrian_metrics[7]*100],
              'Overall Easy' : [AP_weighted_average_easy,difficulty_metrics[0]*100,difficulty_metrics[1]*100],
              'Overall Moderate' : [AP_weighted_average_moderate,difficulty_metrics[2]*100,difficulty_metrics[3]*100],
              'Overall Hard' : [AP_weighted_average_hard,difficulty_metrics[4]*100,difficulty_metrics[5]*100],
              'Overall Overall' : [overall_AP,overall_precision*100,overall_recall*100]

            }

    bar_metrics=pd.DataFrame(metrics_results)

    # bar_metrics.plot(kind='bar',title="SMOKE Evaluation Results")

    # plt.xlabel("Difficulties")
    # plt.ylabel("Percentage %")
    # plt.xticks(ticks=[0,1,2],labels=['AP','Precision','Recall'])



    return df1,bar_metrics



def construct_dataframe_v2(cars_AP,pedestrians_AP,car_metrics,pedestrian_metrics,difficulty_metrics,n_objects_classes,n_objects_difficulties):

    cars_easy_AP,cars_moderate_AP,cars_hard_AP=cars_AP[0],cars_AP[1],cars_AP[2]
    pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=pedestrians_AP[0],pedestrians_AP[1],pedestrians_AP[2]

    # easy_car_metrics=
    # moderate_car_metrics=
    # hard_car_metrics=


    # easy_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_easy_AP,car_metrics[0],car_metrics[0])
    # moderate_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_easy_AP,car_metrics[2],car_metrics[3])
    # hard_cars='AP: {} - Precision: {} - Recall: {}      '.format(cars_hard_AP,car_metrics[4],car_metrics[5])
    # overall_cars='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # cars=[easy_cars,moderate_cars,hard_cars,overall_cars]


    # easy_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_easy_AP,pedestrian_metrics[0],pedestrian_metrics[1])
    # moderate_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_easy_AP,pedestrian_metrics[2],pedestrian_metrics[3])
    # hard_pedestrians='AP: {} - Precision: {} - Recall: {}       '.format(pedestrian_hard_AP,pedestrian_metrics[4],pedestrian_metrics[5])
    # overall_pedestrians='AP: {} - Precision: {} - Recall: {}        '.format(0,0,0)
    # pedestrians=[easy_pedestrians,moderate_pedestrians,hard_pedestrians,overall_pedestrians]

    # easy_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # moderate_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # hard_classes='AP: {} - Precision: {} - Recall: {}       '.format(0,0,0)
    # overall_classes='AP: {} - Precision: {} - Recall: {}        '.format(0,0,0)

    if n_objects_classes[3]!=0:

        ratio_easy_cars_to_total_cars=n_objects_classes[0]/n_objects_classes[3]
        ratio_moderate_cars_to_total_cars=n_objects_classes[1]/n_objects_classes[3]
        ratio_hard_cars_to_total_cars=n_objects_classes[2]/n_objects_classes[3]
        

    else:

        ratio_easy_cars_to_total_cars=0
        ratio_moderate_cars_to_total_cars=0
        ratio_hard_cars_to_total_cars=0


    if n_objects_classes[7]!=0:
        ratio_easy_pedestrians_to_total_pedestrians=n_objects_classes[4]/n_objects_classes[7]
        ratio_moderate_pedestrians_to_total_pedestrians=n_objects_classes[5]/n_objects_classes[7]
        ratio_hard_pedestrians_to_total_pedestrians=n_objects_classes[6]/n_objects_classes[7]
    else:
        ratio_easy_pedestrians_to_total_pedestrians=0
        ratio_moderate_pedestrians_to_total_pedestrians=0
        ratio_hard_pedestrians_to_total_pedestrians=0



    overall_cars_AP=(cars_easy_AP*ratio_easy_cars_to_total_cars) +(cars_moderate_AP*ratio_moderate_cars_to_total_cars)+(cars_hard_AP*ratio_hard_cars_to_total_cars)
    overall_predestrians_AP=(pedestrian_easy_AP*ratio_easy_pedestrians_to_total_pedestrians)+(pedestrian_moderate_AP*ratio_moderate_pedestrians_to_total_pedestrians)+(pedestrian_hard_AP*ratio_hard_pedestrians_to_total_pedestrians)

    easy_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_easy_AP,car_metrics[0]*100,car_metrics[1]*100)
    moderate_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_moderate_AP,car_metrics[2]*100,car_metrics[3]*100)
    hard_cars=' {:.2f} - {:.2f} - {:.2f} '.format(cars_hard_AP,car_metrics[4]*100,car_metrics[5]*100)
    overall_cars=' {:.2f} - {:.2f} - {:.2f} '.format(overall_cars_AP,car_metrics[6]*100,car_metrics[7]*100)
    cars=[easy_cars,moderate_cars,hard_cars,overall_cars,n_objects_classes[3]]


    easy_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_easy_AP,pedestrian_metrics[0]*100,pedestrian_metrics[1]*100)
    moderate_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_moderate_AP,pedestrian_metrics[2]*100,pedestrian_metrics[3]*100)
    hard_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(pedestrian_hard_AP,pedestrian_metrics[4]*100,pedestrian_metrics[5]*100)
    overall_pedestrians=' {:.2f} - {:.2f} - {:.2f} '.format(overall_predestrians_AP,pedestrian_metrics[6]*100,pedestrian_metrics[7]*100)
    pedestrians=[easy_pedestrians,moderate_pedestrians,hard_pedestrians,overall_pedestrians,n_objects_classes[7]]


    if n_objects_difficulties[0]!=0:


        ratio_easy_cars_to_easy_objects=n_objects_classes[0]/n_objects_difficulties[0]
        ratio_easy_pedestrians_to_easy_objects=n_objects_classes[4]/n_objects_difficulties[0]

    else:
        ratio_easy_cars_to_easy_objects=0
        ratio_easy_pedestrians_to_easy_objects=0

    

    if n_objects_difficulties[1]!=0:


        ratio_moderate_cars_to_moderate_objects=n_objects_classes[1]/n_objects_difficulties[1]
        ratio_moderate_pedestrians_to_moderate_objects=n_objects_classes[5]/n_objects_difficulties[1]

    else:
        ratio_moderate_cars_to_moderate_objects=0
        ratio_moderate_pedestrians_to_moderate_objects=0

    

    if n_objects_difficulties[2]!=0:

        ratio_hard_cars_to_hard_objects=n_objects_classes[2]/n_objects_difficulties[2]
        ratio_hard_pedestrians_to_hard_objects=n_objects_classes[6]/n_objects_difficulties[2]

    else:
        ratio_hard_cars_to_hard_objects=0
        ratio_hard_pedestrians_to_hard_objects=0

    AP_weighted_average_easy=(cars_easy_AP*ratio_easy_cars_to_easy_objects) + (pedestrian_easy_AP*ratio_easy_pedestrians_to_easy_objects)
    AP_weighted_average_moderate=(cars_moderate_AP*ratio_moderate_cars_to_moderate_objects) + (pedestrian_moderate_AP*ratio_moderate_pedestrians_to_moderate_objects)
    AP_weighted_average_hard=(cars_hard_AP*ratio_hard_cars_to_hard_objects) + (pedestrian_hard_AP*ratio_hard_pedestrians_to_hard_objects)

    easy_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_easy,difficulty_metrics[0]*100,difficulty_metrics[1]*100)
    moderate_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_moderate,difficulty_metrics[2]*100,difficulty_metrics[3]*100)
    hard_classes=' {:.2f} - {:.2f} - {:.2f} '.format(AP_weighted_average_hard,difficulty_metrics[4]*100,difficulty_metrics[5]*100)


    ratio_cars_to_total_objects=n_objects_classes[3]/(n_objects_classes[7]+n_objects_classes[3])
    ratio_pedestrians_to_total_objects=n_objects_classes[7]/(n_objects_classes[7]+n_objects_classes[3])
    overall_precision=(car_metrics[6]*ratio_cars_to_total_objects)+ (pedestrian_metrics[6]*ratio_pedestrians_to_total_objects)
    overall_recall=(car_metrics[7]*ratio_cars_to_total_objects)+ (pedestrian_metrics[7]*ratio_pedestrians_to_total_objects)
    overall_AP=(overall_cars_AP*ratio_cars_to_total_objects)+(overall_predestrians_AP*ratio_pedestrians_to_total_objects)





    overall_classes=' {:.2f} - {:.2f} - {:.2f} '.format(overall_AP,overall_precision*100,overall_recall*100)
    classes=[easy_classes,moderate_classes,hard_classes,overall_classes,(n_objects_classes[7]+n_objects_classes[3])]

    column_headers=['Class/Difficulty','Easy','Moderate','Hard','Overall Difficulties','Class # Objects']
    row_headers=['Car','Pedestrian','Cars and Pedestrians','Difficulty # Objects']

    # data=np.zeros((3,4))

    ar1=np.array(cars)
    ar2=np.array(pedestrians)
    ar3=np.array(classes)
    ar4=np.array([n_objects_difficulties[0],n_objects_difficulties[1],n_objects_difficulties[2],n_objects_difficulties[3],'Ø'])

    data=np.row_stack((ar1,ar2,ar3,ar4))

    ARRAY=np.column_stack((row_headers,data))

    ARRAY=np.row_stack((column_headers,ARRAY))

    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('colheader_justify', 'center')

    df=pd.DataFrame(ARRAY)

    df.columns = df.iloc[0]
    df = df[1:]

    # dfStyler = df.style.set_properties(**{'text-align': 'center'})
    # dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    # [dict(selector='th', props=[('text-align', 'center')])]
    
    df1 = df.style.set_table_styles([{"selector": "", "props": [("border", "1px solid")]},
        {"selector": "tbody td", "props": [("border", "1px solid")]},
        {"selector": "th", "props": [("border", "1px solid")]},
        {"selector": "th", "props": [("text-align", "center")]},
        ])
    df1.set_caption('Object Detector Metrics Evaluation [Average Precision - Precision - Recall ]')
    df1.set_properties(**{'text-align': 'center'}).hide_index()
    #df1.style.set_table_attributes("style='display:inline'").set_caption('Caption table')

    # df2=df1.style.set_table_styles(
    #     [{"selector": "", "props": [("border", "1px solid")]},
    #       {"selector": "tbody td", "props": [("border", "1px solid")]},
    #      {"selector": "th", "props": [("border", "1px solid")]}])


   
    print(df)


    # AP - Precision - Recall

    # metrics_results={
    #           'Easy Cars':[cars_easy_AP,car_metrics[0]*100,car_metrics[1]*100],
    #           'Moderate Cars':[cars_moderate_AP,car_metrics[2]*100,car_metrics[3]*100],
    #           'Hard Cars':[cars_hard_AP,car_metrics[4]*100,car_metrics[5]*100],
    #           'Overall Cars':[overall_cars_AP,car_metrics[6]*100,car_metrics[7]*100],
    #           'Pedestrians Easy':[pedestrian_easy_AP,pedestrian_metrics[0]*100,pedestrian_metrics[1]*100],
    #           'Pedestrians Moderate':[pedestrian_moderate_AP,pedestrian_metrics[2]*100,pedestrian_metrics[3]*100],
    #           'Pedestrians Hard':[pedestrian_hard_AP,pedestrian_metrics[4]*100,pedestrian_metrics[5]*100],
    #           'Overall Pedestrians':[overall_predestrians_AP,pedestrian_metrics[6]*100,pedestrian_metrics[7]*100],
    #           'Overall Easy' : [AP_weighted_average_easy,difficulty_metrics[0]*100,difficulty_metrics[1]*100],
    #           'Overall Moderate' : [AP_weighted_average_moderate,difficulty_metrics[2]*100,difficulty_metrics[3]*100],
    #           'Overall Hard' : [AP_weighted_average_hard,difficulty_metrics[4]*100,difficulty_metrics[5]*100],
    #           'Overall Overall' : [overall_AP,overall_precision*100,overall_recall*100]

    #         }

    # Easy - Moderate - Hard - Overall
    metrics_results={

              'Cars AP':[cars_easy_AP,cars_moderate_AP,cars_hard_AP,overall_cars_AP],
              'Pedestrians AP':[pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP,overall_cars_AP],
              'All Classes AP' : [AP_weighted_average_easy,AP_weighted_average_moderate,AP_weighted_average_hard,overall_AP],


              'Cars Precision':[car_metrics[0]*100,car_metrics[2]*100,car_metrics[4]*100,car_metrics[6]*100],
              'Pedestrians Precision':[pedestrian_metrics[0]*100,pedestrian_metrics[2]*100,pedestrian_metrics[4]*100,pedestrian_metrics[6]*100],
              'All Classes Precision' : [difficulty_metrics[0]*100,difficulty_metrics[2]*100,difficulty_metrics[4]*100,overall_precision*100],

              'Cars Recall':[car_metrics[1]*100,car_metrics[3]*100,car_metrics[5]*100,car_metrics[7]*100],
              'Pedestrians Recall':[pedestrian_metrics[1]*100,pedestrian_metrics[3]*100,pedestrian_metrics[5]*100,pedestrian_metrics[7]*100],
              'All Classes Recall' : [difficulty_metrics[1]*100,difficulty_metrics[3]*100,difficulty_metrics[5]*100,overall_recall*100],

            }

    bar_metrics=pd.DataFrame(metrics_results)

    # bar_metrics.plot(kind='bar',title="SMOKE Evaluation Results")

    # plt.xlabel("Difficulties")
    # plt.ylabel("Percentage %")
    # plt.xticks(ticks=[0,1,2],labels=['AP','Precision','Recall'])



    return df1,bar_metrics