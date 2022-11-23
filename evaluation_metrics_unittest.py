import cv2
import numpy as np
import random

from torch import mode
import pandas as pd
from evaluation_toolbox import rectanges_overlap,Point,get_IoU,get_key
import dataframe_image as dfi
import os

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

def get_metrics(gt_classes,predicted_classes,labels):
    gt_label_count=[]
    pred_label_count=[]
    for label in labels:
        gt_label_count.append(gt_classes.count(label))
        pred_label_count.append(predicted_classes.count(label))

    if len(predicted_classes)>len(gt_classes):
        FP=len(predicted_classes)-len(gt_classes)
    else:
        FP=0
    
    TP=0

    if len(gt_classes)>len(predicted_classes):
        indix_range=len(predicted_classes)

    else:
        indix_range=len(gt_classes)

    for i in range(indix_range):
        if gt_classes[i]==predicted_classes[i]:
            TP=TP+1

        else:
            FP=FP+1


    gt_ROIs=len(gt_classes)
    predicted_ROIs=len(pred_boxs)
    FN=len(gt_boxs)-TP

    return gt_ROIs,predicted_ROIs,TP,FP,FN

def get_metrics_label_specific(label,difficulty,labels,gt_classes,predicted_classes,gt_boxs,pred_boxs):
    gt_label_count=[]
    pred_label_count=[]
    for l in labels:
        gt_label_count.append(gt_classes.count(l))
        pred_label_count.append(predicted_classes.count(l))

    i=labels.index(label)
    print('i: ',i)
    gt_instances_label=gt_label_count[i]
    predicted_instances_label=pred_label_count[i]

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

    gt_instances_label=len(gt_classes)
    predicted_instances_label=len(predicted_classes)


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

    if len(gt_classes)>len(pred_classes):
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


    

def visualize_metrics_calculations(gt_ROIs_n,predicted_ROIs_n,TP,FP,FN):
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

    if predicted_ROIs_n==TP+FP:
        check_precision=True
    else:
        check_precision=False

    if gt_ROIs_n==TP+FN:
        check_recall=True
    else:
        check_recall=False

    


    evaluation_metrics_window=np.zeros((img_height,img_width,3))
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '# Ground Truth Objects: '+str(gt_ROIs_n), (0,30), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '# Predicted Objects: '+str(predicted_ROIs_n), (0,60), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'TP: '+str(TP), (0,90), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'FP: '+str(FP), (0,120), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, 'FN: '+str(FN), (0,150), font, fontScale, red_color, thickness, cv2.LINE_AA)
    #print("The band for {} is {} out of 10".format(name, band))
    precision_string_expression="Precision = TP/(TP+FP) = {} / {}  Check = {}".format(TP,TP+FP,check_precision)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, precision_string_expression, (0,180), font, fontScale, red_color, thickness, cv2.LINE_AA)
    recall_string_expression="Recall = TP/(TP+FN) = {} / {}  Check = {}".format(TP,TP+FN,check_recall)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, recall_string_expression, (0,210), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '(TP+FP) = # Predicted Objects', (0,240), font, fontScale, red_color, thickness, cv2.LINE_AA)
    evaluation_metrics_window = cv2.putText(evaluation_metrics_window, '(TP+FN) = # Ground Truth Objects', (0,270), font, fontScale, red_color, thickness, cv2.LINE_AA)


    return evaluation_metrics_window


    

def get_box_difficulty_index(gt_boxs,gt_classes,label,difficulty):

    easy_difficulty_index_list=[]
    moderate_difficulty_index_list=[]

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
        elif 5<box_height<40:
           # moderate_difficulty_index_list.append(i)

            if gt_classes[i]==label:
                moderate_difficulty_index_list.append(i)
            else:
                pass
        else:
            pass

    if difficulty=='Easy':
        return easy_difficulty_index_list
    else:
        return moderate_difficulty_index_list


def tabularize_metrics(TP_A1_list,FP_A1_list,FN_A1_list,TP_A2_list,FP_A2_list,FN_A2_list,TP_B1_list,FP_B1_list,FN_B1_list,TP_B2_list,FP_B2_list,FN_B2_list,TP_C1_list,FP_C2_list,FN_C2_list,n_frames):
    ΣTP_A1=[sum(TP_A1_list[0:i+1]) for i in range(len(TP_A1_list))]
    ΣFP_A1=[sum(FP_A1_list[0:i+1]) for i in range(len(FP_A1_list))]
    ΣFN_A1=[sum(FN_A1_list[0:i+1]) for i in range(len(FN_A1_list))]

    ΣTP_A2=[sum(TP_A2_list[0:i+1]) for i in range(len(TP_A2_list))]
    ΣFP_A2=[sum(FP_A2_list[0:i+1]) for i in range(len(FP_A2_list))]
    ΣFN_A2=[sum(FN_A2_list[0:i+1]) for i in range(len(FN_A2_list))]

    ΣTP_B1=[sum(TP_B1_list[0:i+1]) for i in range(len(TP_B1_list))]
    ΣFP_B1=[sum(FP_B1_list[0:i+1]) for i in range(len(FP_B1_list))]
    ΣFN_B1=[sum(FN_B1_list[0:i+1]) for i in range(len(FN_B1_list))]

    ΣTP_B2=[sum(TP_B2_list[0:i+1]) for i in range(len(TP_B2_list))]
    ΣFP_B2=[sum(FP_B2_list[0:i+1]) for i in range(len(FP_B2_list))]
    ΣFN_B2=[sum(FN_B2_list[0:i+1]) for i in range(len(FN_B2_list))]

    ΣTP_C1=[sum(TP_C1_list[0:i+1]) for i in range(len(TP_C1_list))]
    ΣFP_C1=[sum(FP_C1_list[0:i+1]) for i in range(len(FP_C1_list))]
    ΣFN_C1=[sum(FN_C1_list[0:i+1]) for i in range(len(FN_C1_list))]

    ΣTP_C2=[sum(TP_C2_list[0:i+1]) for i in range(len(TP_C2_list))]
    ΣFP_C2=[sum(FP_C2_list[0:i+1]) for i in range(len(FP_C2_list))]
    ΣFN_C2=[sum(FN_C2_list[0:i+1]) for i in range(len(FN_C2_list))]

    ΣA1_precision=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFP_A1[i])  if (ΣTP_A1[i]+ΣFP_A1[i]) is not 0 else 0 for i in range(n_frames)]
    ΣA1_recall=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFN_A1[i]) if (ΣTP_A1[i]+ΣFN_A1[i]) is not 0 else 0 for i in range(n_frames)  ]

    ΣA2_precision=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFP_A2[i]) if (ΣTP_A2[i]+ΣFP_A2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣA2_recall=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFN_A2[i])  if (ΣTP_A2[i]+ΣFN_A2[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB1_precision=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFP_B1[i]) if (ΣTP_B1[i]+ΣFP_B1[i]) is not 0 else 0 for i in range(n_frames)  ]
    ΣB1_recall=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFN_B1[i]) if (ΣTP_B1[i]+ΣFN_B1[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣB2_precision=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFP_B2[i]) if (ΣTP_B2[i]+ΣFP_B2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣB2_recall=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFN_B2[i]) if (ΣTP_B2[i]+ΣFN_B2[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC1_precision=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFP_C1[i]) if (ΣTP_C1[i]+ΣFP_C1[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC1_recall=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFN_C1[i]) if (ΣTP_C1[i]+ΣFN_C1[i]) is not 0 else 0 for i in range(n_frames) ]

    ΣC2_precision=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFP_C2[i]) if (ΣTP_C2[i]+ΣFP_C2[i]) is not 0 else 0 for i in range(n_frames) ]
    ΣC2_recall=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFN_C2[i]) if (ΣTP_C2[i]+ΣFN_C2[i]) is not 0 else 0 for i in range(n_frames) ]

    A1_frame_precision=[TP_A1_list[i]/(TP_A1_list[i]+FP_A1_list[i]) if (TP_A1_list[i]+FP_A1_list[i]) is not 0 else 0 for i in range(n_frames)]
    A1_mean_precision=[sum(A1_frame_precision[0:i+1])/len(A1_frame_precision[0:i+1]) for i in range(n_frames)]
    A1_frame_recall=[TP_A1_list[i]/(TP_A1_list[i]+FN_A1_list[i]) if (TP_A1_list[i]+FN_A1_list[i]) is not 0 else 0 for i in range(len(TP_A1_list))]
    A1_mean_recall=[sum(A1_frame_recall[0:i+1])/len(A1_frame_recall[0:i+1]) for i in range(n_frames)]

    A2_frame_precision=[TP_A2_list[i]/(TP_A2_list[i]+FP_A2_list[i]) if (TP_A2_list[i]+FP_A2_list[i]) is not 0 else 0 for i in range(n_frames)]
    A2_mean_precision=[sum(A2_frame_precision[0:i+1])/len(A2_frame_precision[0:i+1]) for i in range(n_frames)]
    A2_frame_recall=[TP_A2_list[i]/(TP_A2_list[i]+FN_A2_list[i]) if (TP_A2_list[i]+FN_A2_list[i]) is not 0 else 0 for i in range(len(TP_A2_list))]
    A2_mean_recall=[sum(A2_frame_recall[0:i+1])/len(A2_frame_recall[0:i+1]) for i in range(n_frames)]

    B1_frame_precision=[TP_B1_list[i]/(TP_B1_list[i]+FP_B1_list[i]) if (TP_B1_list[i]+FP_B1_list[i]) is not 0 else 0 for i in range(n_frames)]
    B1_mean_precision=[sum(B1_frame_precision[0:i+1])/len(B1_frame_precision[0:i+1]) for i in range(n_frames)]
    B1_frame_recall=[TP_B1_list[i]/(TP_B1_list[i]+FN_B1_list[i]) if (TP_B1_list[i]+FN_B1_list[i]) is not 0 else 0 for i in range(len(TP_B1_list))]
    B1_mean_recall=[sum(B1_frame_recall[0:i+1])/len(B1_frame_recall[0:i+1]) for i in range(n_frames)]

    B2_frame_precision=[TP_B2_list[i]/(TP_B2_list[i]+FP_B2_list[i]) if (TP_B2_list[i]+FP_B2_list[i]) is not 0 else 0 for i in range(n_frames)]
    B2_mean_precision=[sum(B2_frame_precision[0:i+1])/len(B2_frame_precision[0:i+1]) for i in range(n_frames)]
    B2_frame_recall=[TP_B2_list[i]/(TP_B2_list[i]+FN_B2_list[i]) if (TP_B2_list[i]+FN_B2_list[i]) is not 0 else 0 for i in range(len(TP_B2_list))]
    B2_mean_recall=[sum(B2_frame_recall[0:i+1])/len(B2_frame_recall[0:i+1]) for i in range(n_frames)]

    C1_frame_precision=[TP_C1_list[i]/(TP_C1_list[i]+FP_C1_list[i]) if (TP_C1_list[i]+FP_C1_list[i]) is not 0 else 0 for i in range(n_frames)]
    C1_mean_precision=[sum(C1_frame_precision[0:i+1])/len(C1_frame_precision[0:i+1]) for i in range(n_frames)]
    C1_frame_recall=[TP_C1_list[i]/(TP_C1_list[i]+FN_C1_list[i]) if (TP_C1_list[i]+FN_C1_list[i]) is not 0 else 0 for i in range(len(TP_C1_list))]
    C1_mean_recall=[sum(C1_frame_recall[0:i+1])/len(C1_frame_recall[0:i+1]) for i in range(n_frames)]

    C2_frame_precision=[TP_C2_list[i]/(TP_C2_list[i]+FP_C2_list[i]) if (TP_C2_list[i]+FP_C2_list[i]) is not 0 else 0 for i in range(n_frames)]
    C2_mean_precision=[sum(C2_frame_precision[0:i+1])/len(C2_frame_precision[0:i+1]) for i in range(n_frames)]
    C2_frame_recall=[TP_C2_list[i]/(TP_C2_list[i]+FN_C2_list[i]) if (TP_C2_list[i]+FN_C2_list[i]) is not 0 else 0 for i in range(len(TP_C2_list))]
    C2_mean_recall=[sum(C2_frame_recall[0:i+1])/len(C2_frame_recall[0:i+1]) for i in range(n_frames)]

    classA1_array=np.column_stack((TP_A1_list,FP_A1_list,FN_A1_list,A1_frame_precision,A1_frame_recall,ΣTP_A1,ΣFP_A1,ΣFN_A1,ΣA1_precision,ΣA1_recall,A1_mean_precision,A1_mean_recall))
    classA1_data=pd.DataFrame(classA1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classA1_data,"/home/hasan/perception-validation-verification/unit_test_figs/A1.png")

    classA2_array=np.column_stack((TP_A2_list,FP_A2_list,FN_A2_list,A2_frame_precision,A2_frame_recall,ΣTP_A2,ΣFP_A2,ΣFN_A2,ΣA2_precision,ΣA2_recall,A2_mean_precision,A2_mean_recall))
    classA2_data=pd.DataFrame(classA2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classA2_data,"/home/hasan/perception-validation-verification/unit_test_figs/A2.png")

    classB1_array=np.column_stack((TP_B1_list,FP_B1_list,FN_B1_list,B1_frame_precision,B1_frame_recall,ΣTP_B1,ΣFP_B1,ΣFN_B1,ΣB1_precision,ΣB1_recall,B1_mean_precision,B1_mean_recall))
    classB1_data=pd.DataFrame(classB1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classB1_data,"/home/hasan/perception-validation-verification/unit_test_figs/B1.png")

    classB2_array=np.column_stack((TP_B2_list,FP_B2_list,FN_B2_list,B2_frame_precision,B2_frame_recall,ΣTP_B2,ΣFP_B2,ΣFN_B2,ΣB2_precision,ΣB2_recall,B2_mean_precision,B2_mean_recall))
    classB2_data=pd.DataFrame(classB2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classB2_data,"/home/hasan/perception-validation-verification/unit_test_figs/B2.png")

    classC1_array=np.column_stack((TP_C1_list,FP_C1_list,FN_C1_list,C1_frame_precision,C1_frame_recall,ΣTP_C1,ΣFP_C1,ΣFN_C1,ΣC1_precision,ΣC1_recall,C1_mean_precision,C1_mean_recall))
    classC1_data=pd.DataFrame(classC1_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classC1_data,"/home/hasan/perception-validation-verification/unit_test_figs/C1.png")

    classC2_array=np.column_stack((TP_C2_list,FP_C2_list,FN_C2_list,C2_frame_precision,C2_frame_recall,ΣTP_C2,ΣFP_C2,ΣFN_C2,ΣC2_precision,ΣC2_recall,C2_mean_precision,C2_mean_recall))
    classC2_data=pd.DataFrame(classC2_array,columns=['TP','FP','FN','Frame Precision','Frame Recall','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall','Mean Precision','Mean Recall'])
    dfi.export(classC2_data,"/home/hasan/perception-validation-verification/unit_test_figs/C2.png")


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


def read_groundtruth(gt_folder,fileid):
    boxs_groundtruth_file=os.path.join(gt_folder,str(fileid).zfill(6)+'.txt')
    with open(boxs_groundtruth_file,'r') as file:
        firstLine_elements = len(file.readline().split())
        boxs_groundtruth_string=file.read()



    groundtruth_list=boxs_groundtruth_string.split()
    groundtruth = [groundtruth_list[x:x+firstLine_elements] for x in range(0,int(len(groundtruth_list)),firstLine_elements)]

    return groundtruth

def get_gt_classes_boxes(groundtruth):
    gt_classes=[groundtruth_detection_list[0]  for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare']
    gt_boxs=[(Point(float(groundtruth_detection_list[4]),float(groundtruth_detection_list[5])),Point(float(groundtruth_detection_list[6]),float(groundtruth_detection_list[7]))) for groundtruth_detection_list in groundtruth if groundtruth_detection_list[0]!='DontCare' ]

    return gt_classes,gt_boxs

def get_pred_classes_boxes(smoke_predictions_list):
    pred_classes=[get_key(prediction_list[0]) if type(smoke_predictions_list[0])==list else smoke_predictions_list[0] for prediction_list in smoke_predictions_list]
    pred_boxs=[(Point(float(prediction_list[2]),float(prediction_list[3])),Point(float(prediction_list[4]),float(prediction_list[5])))  if type(smoke_predictions_list[0])==list else (Point(float(smoke_predictions_list[2]),float(smoke_predictions_list[3])),Point(float(smoke_predictions_list[4]),float(smoke_predictions_list[5])))  for prediction_list in smoke_predictions_list ]

    
    return pred_classes,pred_boxs




#########################################################################################################





img_width=800
img_height=400
# OK SEEDS 7,11,14,16
# 7: 5GT 4P
# 11: 6GT 4P 3TP 1FP
# 14: 3GT 1P 1TP 0FP
# 16: 5GT 6P 2TP 4FP
# 19: 3GT 5P 1TP 4FP
# 26: 4GT 3P 2TP 1FP
# 38: 6GT 7P 4TP 3FP
# 40: 6GT 6P 3TP 3FP
TP_A1_list=[]
FP_A1_list=[]
FN_A1_list=[]

TP_A2_list=[]
FP_A2_list=[]
FN_A2_list=[]

TP_B1_list=[]
FP_B1_list=[]
FN_B1_list=[]

TP_B2_list=[]
FP_B2_list=[]
FN_B2_list=[]

TP_C1_list=[]
FP_C1_list=[]
FN_C1_list=[]

TP_C2_list=[]
FP_C2_list=[]
FN_C2_list=[]


n_frames=5
for i in range(n_frames):
    random.seed(i)
    img=np.zeros((img_height,img_width,3))
    #evaluation_metrics_window=img




    n_gt_detections=random.randint(3,6)
    labels=['A','B','C']
    gt_classes=get_gt_classes(labels,n_gt_detections)
    gt_boxs,groundtruth_img,gt_boxs_dim=get_plot_groundtruth(img,gt_classes)

    total_predictions_n=random.randint(n_gt_detections-2,n_gt_detections+2)

    pred_classes=get_predicted_classes(labels,total_predictions_n)
    pred_boxs,output_img=get_plot_predictions(gt_boxs,gt_boxs_dim,(img_width,img_height),groundtruth_img,pred_classes,gt_classes,total_predictions_n)
    #pred_classes=get_predicted_classes(labels,n_gt_detections)

    print('Predicted Classes: ',pred_classes)
    print('Ground Truth Classes: ',gt_classes)

    print('Total Predictions: ',len(pred_classes))
    print('Total GT Detections: ',len(gt_classes))

    gt_ROIs_n,predicted_ROIs_n,TP,FP,FN=get_metrics(gt_classes,pred_classes,labels)

    print('Gt Boxs: ',gt_boxs)
    classA1_gt_instances,classA1_pred_instances,TP_A1,FP_A1,FN_A1=get_metrics_label_specific('A','Easy',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)
    classA2_gt_instances,classA2_pred_instances,TP_A2,FP_A2,FN_A2=get_metrics_label_specific('A','Moderate',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)

    classB1_gt_instances,classB1_pred_instances,TP_B1,FP_B1,FN_B1=get_metrics_label_specific('B','Easy',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)
    classB2_gt_instances,classB2_pred_instances,TP_B2,FP_B2,FN_B2=get_metrics_label_specific('B','Moderate',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)


    classC1_gt_instances,classC1_pred_instances,TP_C1,FP_C1,FN_C1=get_metrics_label_specific('C','Easy',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)
    classC2_gt_instances,classC2_pred_instances,TP_C2,FP_C2,FN_C2=get_metrics_label_specific('C','Moderate',labels,gt_classes,pred_classes,gt_boxs,pred_boxs)


    classA1_metrics_window=visualize_metrics_calculations(classA1_gt_instances,classA1_pred_instances,TP_A1,FP_A1,FN_A1)
    classA2_metrics_window=visualize_metrics_calculations(classA2_gt_instances,classA2_pred_instances,TP_A2,FP_A2,FN_A2)

    classB1_metrics_window=visualize_metrics_calculations(classB1_gt_instances,classB1_pred_instances,TP_B1,FP_B1,FN_B1)
    classB2_metrics_window=visualize_metrics_calculations(classB2_gt_instances,classB2_pred_instances,TP_B2,FP_B2,FN_B2)


    classC1_metrics_window=visualize_metrics_calculations(classC1_gt_instances,classC1_pred_instances,TP_C1,FP_C1,FN_C1)
    classC2_metrics_window=visualize_metrics_calculations(classC2_gt_instances,classC2_pred_instances,TP_C2,FP_C2,FN_C2)

    total_metrics_window=visualize_metrics_calculations(gt_ROIs_n,predicted_ROIs_n,TP,FP,FN)


    cv2.imwrite('/home/hasan/perception-validation-verification/unit_test_figs/randomseed'+str(i)+'.png',output_img)

    TP_A1_list.append(TP_A1)
    FP_A1_list.append(FP_A1)
    FN_A1_list.append(FN_A1)

    TP_A2_list.append(TP_A2)
    FP_A2_list.append(FP_A2)
    FN_A2_list.append(FN_A2)

    TP_B1_list.append(TP_B1)
    FP_B1_list.append(FP_B1)
    FN_B1_list.append(FN_B1)

    TP_B2_list.append(TP_B2)
    FP_B2_list.append(FP_B2)
    FN_B2_list.append(FN_B2)

    TP_C1_list.append(TP_C1)
    FP_C1_list.append(TP_C1)
    FN_C1_list.append(TP_C1)

    TP_C2_list.append(TP_C2)
    FP_C2_list.append(FP_C2)
    FN_C2_list.append(FN_C2)

tabularize_metrics(TP_A1_list,FP_A1_list,FN_A1_list,TP_A2_list,FP_A2_list,FN_A2_list,TP_B1_list,FP_B1_list,FN_B1_list,TP_B2_list,FP_B2_list,FN_B2_list,TP_C1_list,FP_C2_list,FN_C2_list,n_frames)
# ΣTP_A1=[sum(TP_A1_list[0:i+1]) for i in range(len(TP_A1_list))]
# ΣFP_A1=[sum(FP_A1_list[0:i+1]) for i in range(len(FP_A1_list))]
# ΣFN_A1=[sum(FN_A1_list[0:i+1]) for i in range(len(FN_A1_list))]

# ΣTP_A2=[sum(TP_A2_list[0:i+1]) for i in range(len(TP_A2_list))]
# ΣFP_A2=[sum(FP_A2_list[0:i+1]) for i in range(len(FP_A2_list))]
# ΣFN_A2=[sum(FN_A2_list[0:i+1]) for i in range(len(FN_A2_list))]

# ΣTP_B1=[sum(TP_B1_list[0:i+1]) for i in range(len(TP_B1_list))]
# ΣFP_B1=[sum(FP_B1_list[0:i+1]) for i in range(len(FP_B1_list))]
# ΣFN_B1=[sum(FN_B1_list[0:i+1]) for i in range(len(FN_B1_list))]

# ΣTP_B2=[sum(TP_B2_list[0:i+1]) for i in range(len(TP_B2_list))]
# ΣFP_B2=[sum(FP_B2_list[0:i+1]) for i in range(len(FP_B2_list))]
# ΣFN_B2=[sum(FN_B2_list[0:i+1]) for i in range(len(FN_B2_list))]

# ΣTP_C1=[sum(TP_C1_list[0:i+1]) for i in range(len(TP_C1_list))]
# ΣFP_C1=[sum(FP_C1_list[0:i+1]) for i in range(len(FP_C1_list))]
# ΣFN_C1=[sum(FN_C1_list[0:i+1]) for i in range(len(FN_C1_list))]

# ΣTP_C2=[sum(TP_C2_list[0:i+1]) for i in range(len(TP_C2_list))]
# ΣFP_C2=[sum(FP_C2_list[0:i+1]) for i in range(len(FP_C2_list))]
# ΣFN_C2=[sum(FN_C2_list[0:i+1]) for i in range(len(FN_C2_list))]

# ΣA1_precision=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFP_A1[i])  if (ΣTP_A1[i]+ΣFP_A1[i]) is not 0 else 0 for i in range(n_frames)]
# ΣA1_recall=[ΣTP_A1[i]/(ΣTP_A1[i]+ΣFN_A1[i]) if (ΣTP_A1[i]+ΣFN_A1[i]) is not 0 else 0 for i in range(n_frames)  ]

# ΣA2_precision=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFP_A2[i]) if (ΣTP_A2[i]+ΣFP_A2[i]) is not 0 else 0 for i in range(n_frames) ]
# ΣA2_recall=[ΣTP_A2[i]/(ΣTP_A2[i]+ΣFN_A2[i])  if (ΣTP_A2[i]+ΣFN_A2[i]) is not 0 else 0 for i in range(n_frames) ]

# ΣB1_precision=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFP_B1[i]) if (ΣTP_B1[i]+ΣFP_B1[i]) is not 0 else 0 for i in range(n_frames)  ]
# ΣB1_recall=[ΣTP_B1[i]/(ΣTP_B1[i]+ΣFN_B1[i]) if (ΣTP_B1[i]+ΣFN_B1[i]) is not 0 else 0 for i in range(n_frames) ]

# ΣB2_precision=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFP_B2[i]) if (ΣTP_B2[i]+ΣFP_B2[i]) is not 0 else 0 for i in range(n_frames) ]
# ΣB2_recall=[ΣTP_B2[i]/(ΣTP_B2[i]+ΣFN_B2[i]) if (ΣTP_B2[i]+ΣFN_B2[i]) is not 0 else 0 for i in range(n_frames) ]

# ΣC1_precision=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFP_C1[i]) if (ΣTP_C1[i]+ΣFP_C1[i]) is not 0 else 0 for i in range(n_frames) ]
# ΣC1_recall=[ΣTP_C1[i]/(ΣTP_C1[i]+ΣFN_C1[i]) if (ΣTP_C1[i]+ΣFN_C1[i]) is not 0 else 0 for i in range(n_frames) ]

# ΣC2_precision=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFP_C2[i]) if (ΣTP_C2[i]+ΣFP_C2[i]) is not 0 else 0 for i in range(n_frames) ]
# ΣC2_recall=[ΣTP_C2[i]/(ΣTP_C2[i]+ΣFN_C2[i]) if (ΣTP_C2[i]+ΣFN_C2[i]) is not 0 else 0 for i in range(n_frames) ]

# classA1_array=np.column_stack((TP_A1_list,FP_A1_list,FN_A1_list,ΣTP_A1,ΣFP_A1,ΣFN_A1,ΣA1_precision,ΣA1_recall))
# classA1_data=pd.DataFrame(classA1_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classA1_data,"/home/hasan/perception-validation-verification/unit_test_figs/A1.png")

# classA2_array=np.column_stack((TP_A2_list,FP_A2_list,FN_A2_list,ΣTP_A2,ΣFP_A2,ΣFN_A2,ΣA2_precision,ΣA2_recall))
# classA2_data=pd.DataFrame(classA2_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classA2_data,"/home/hasan/perception-validation-verification/unit_test_figs/A2.png")

# classB1_array=np.column_stack((TP_B1_list,FP_B1_list,FN_B1_list,ΣTP_B1,ΣFP_B1,ΣFN_B1,ΣB1_precision,ΣB1_recall))
# classB1_data=pd.DataFrame(classB1_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classB1_data,"/home/hasan/perception-validation-verification/unit_test_figs/B1.png")

# classB2_array=np.column_stack((TP_B2_list,FP_B2_list,FN_B2_list,ΣTP_B2,ΣFP_B2,ΣFN_B2,ΣB2_precision,ΣB2_recall))
# classB2_data=pd.DataFrame(classB2_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classB2_data,"/home/hasan/perception-validation-verification/unit_test_figs/B2.png")

# classC1_array=np.column_stack((TP_C1_list,FP_C1_list,FN_C1_list,ΣTP_C1,ΣFP_C1,ΣFN_C1,ΣC1_precision,ΣC1_recall))
# classC1_data=pd.DataFrame(classC1_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classC1_data,"/home/hasan/perception-validation-verification/unit_test_figs/C1.png")

# classC2_array=np.column_stack((TP_C2_list,FP_C2_list,FN_C2_list,ΣTP_C2,ΣFP_C2,ΣFN_C2,ΣC2_precision,ΣC2_recall))
# classC2_data=pd.DataFrame(classC2_array,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN','ΣPrecision','ΣRecall'])
# dfi.export(classC2_data,"/home/hasan/perception-validation-verification/unit_test_figs/C2.png")

#print(classA1_data)
#     # random_class_prediction=get_random_class_prediction(labels)
#     # pred_classes.append(random_class_prediction)


    
    
    # print('iou: ',str(i)+' '+str(iou))



# # Make your windows
# cv2.namedWindow('Output Img')
# cv2.namedWindow('Total Evaluation Metrics')
# cv2.namedWindow('Class A1 Metrics',cv2.WINDOW_NORMAL)
# cv2.namedWindow('Class A2 Metrics',cv2.WINDOW_NORMAL)

# cv2.namedWindow('Class B1 Metrics',cv2.WINDOW_NORMAL)
# cv2.namedWindow('Class B2 Metrics',cv2.WINDOW_NORMAL)

# cv2.namedWindow('Class C1 Metrics',cv2.WINDOW_NORMAL)
# cv2.namedWindow('Class C2 Metrics',cv2.WINDOW_NORMAL)

# cv2.imshow('Output Img',output_img)
# cv2.imshow('Total Evaluation Metrics',total_metrics_window)
# cv2.imshow('Class A1 Metrics',classA1_metrics_window)
# cv2.imshow('Class A2 Metrics',classA2_metrics_window)

# cv2.imshow('Class B1 Metrics',classB1_metrics_window)
# cv2.imshow('Class B2 Metrics',classB2_metrics_window)

# cv2.imshow('Class C1 Metrics',classC1_metrics_window)
# cv2.imshow('Class C2 Metrics',classC2_metrics_window)

# # Then move your windows to where you want them
# cv2.moveWindow('Output Img', 0,0)
# cv2.moveWindow('Total Evaluation Metrics', 1000, 10)
# cv2.moveWindow('Class A1 Metrics', 0,600)
# cv2.moveWindow('Class B1 Metrics', 700,600)
# cv2.moveWindow('Class C1 Metrics', 1400,600)
# cv2.waitKey(0)


