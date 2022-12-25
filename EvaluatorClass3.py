

from metrics_functions_from_evaluation_script import tabularize_metrics,Point,get_pred_classes_boxes,get_IoU,get_class_AP,construct_dataframe_v2
import numpy as np
import cv2
import subprocess
import os
import dataframe_image as dfi 
import matplotlib.pyplot as plt
# Helper Class that makes it easier to hold data

class DifficultyHolder(object):
    def __init__(self,difficulty):
        self.difficulty=difficulty

        self.TP=0
        self.FP=0
        self.FN=0

class ClassHolder(object):
    def __init__(self,label):
        self.label=label

        self.easy=DifficultyHolder("Easy")
        self.moderate=DifficultyHolder("Moderate")
        self.hard=DifficultyHolder("Hard")

class MetricsHolder(object):
    def __init__(self):

        self.cars=ClassHolder("Car")
        self.pedestrians=ClassHolder("Pedestrian")
        self.cyclists=ClassHolder("Cyclist")

    # Add TP/FP/FN to correct category based on label and difficulty
    def add_to_metrics(self,metrics,label,difficulty):
        if label=="Car":
            if difficulty=="Easy":
                self.cars.easy.TP+=metrics[0]
                self.cars.easy.FP+=metrics[1]
                self.cars.easy.FN+=metrics[2]
            elif difficulty=="Moderate":
                self.cars.moderate.TP+=metrics[0]
                self.cars.moderate.FP+=metrics[1]
                self.cars.moderate.FN+=metrics[2]
            else:
                self.cars.hard.TP+=metrics[0]
                self.cars.hard.FP+=metrics[1]
                self.cars.hard.FN+=metrics[2]
        elif label=="Pedestrian":
            if difficulty=="Easy":
                self.pedestrians.easy.TP+=metrics[0]
                self.pedestrians.easy.FP+=metrics[1]
                self.pedestrians.easy.FN+=metrics[2]
            elif difficulty=="Moderate":
                self.pedestrians.moderate.TP+=metrics[0]
                self.pedestrians.moderate.FP+=metrics[1]
                self.pedestrians.moderate.FN+=metrics[2]
            else:
                self.pedestrians.hard.TP+=metrics[0]
                self.pedestrians.hard.FP+=metrics[1]
                self.pedestrians.hard.FN+=metrics[2]
        else:
            if difficulty=="Easy":
                self.cyclists.easy.TP+=metrics[0]
                self.cyclists.easy.FP+=metrics[1]
                self.cyclists.easy.FN+=metrics[2]
            elif difficulty=="Moderate":
                self.cyclists.moderate.TP+=metrics[0]
                self.cyclists.moderate.FP+=metrics[1]
                self.cyclists.moderate.FN+=metrics[2]
            else:
                self.cyclists.hard.TP+=metrics[0]
                self.cyclists.hard.FP+=metrics[1]
                self.cyclists.hard.FN+=metrics[2]

    # Put all data in a list needed for metrics Tabularization
    def get_data_row(self):

        datarow=[self.cars.easy.TP,self.cars.easy.FP,self.cars.easy.FN,
                self.cars.moderate.TP,self.cars.moderate.FP,self.cars.moderate.FN,
                self.cars.hard.TP,self.cars.hard.FP,self.cars.hard.FN,
                self.cyclists.easy.TP,self.cyclists.easy.FP,self.cyclists.easy.FN,
                self.cyclists.moderate.TP,self.cyclists.moderate.FP,self.cyclists.moderate.FN,
                self.cyclists.hard.TP,self.cyclists.hard.FP,self.cyclists.hard.FN,
                self.pedestrians.easy.TP,self.pedestrians.easy.FP,self.pedestrians.easy.FN,
                self.pedestrians.moderate.TP,self.pedestrians.moderate.FP,self.pedestrians.moderate.FN,
                self.pedestrians.hard.TP,self.pedestrians.hard.FP,self.pedestrians.hard.FN]

        return datarow

    # Store number of total objects, filtered objects, ignored objects for every label and difficulty for Groundtruth & Predictions
    # Useful for debugging
    def store_stats(self,label,difficulty,
                    filtered_gt_for_debugging_purposes_indices,ignored_label_specific_gt_indices,
                    ignored_difficulty_specific_pred_indices,valid_difficulty_specific_pred_indices,
                    pred_label_difficulty_indices):

        if label=='Car':

            self.car_filtered_gt_instances=len(filtered_gt_for_debugging_purposes_indices)

            self.ignored_gt_car_indices=ignored_label_specific_gt_indices
            self.ignored_pred_car_indices=ignored_difficulty_specific_pred_indices#[index for index in self.ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Car']
            self.filtered_pred_car_count=len(valid_difficulty_specific_pred_indices)

            if difficulty=='Easy':
                self.easy_car_pred_indices=pred_label_difficulty_indices
                #print('Easy Car Pred Indices: ',easy_car_pred_indices)
            elif difficulty=='Moderate':
                self.moderate_car_pred_indices=pred_label_difficulty_indices
                #print('Moderate Car Pred Indices: ',moderate_car_pred_indices)
            elif difficulty=='Hard':
                self.hard_car_pred_indices=pred_label_difficulty_indices
                #print('Hard Car Pred Indices: ',hard_car_pred_indices)

        # Get Pedestrian Difficulty Stats
        elif label=='Pedestrian':
            self.pedestrian_filtered_gt_instances=len(filtered_gt_for_debugging_purposes_indices)

            self.ignored_gt_pedestrian_indices=ignored_label_specific_gt_indices
            self.ignored_pred_pedestrian_indices=ignored_difficulty_specific_pred_indices#[index for index in ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Pedestrian']
            self.filtered_pred_pedestrian_count=len(valid_difficulty_specific_pred_indices)
            if difficulty=='Easy':
                self.easy_pedestrian_pred_indices=pred_label_difficulty_indices
            elif difficulty=='Moderate':
                self.moderate_pedestrian_pred_indices=pred_label_difficulty_indices
            elif difficulty=='Hard':
                self.hard_pedestrian_pred_indices=pred_label_difficulty_indices

        # Get Cyclist Difficulty Stats
        elif label=='Cyclist':
            self.cyclist_filtered_gt_instances=len(filtered_gt_for_debugging_purposes_indices)


            self.ignored_gt_cyclist_indices=ignored_label_specific_gt_indices
            self.ignored_pred_cyclist_indices=ignored_difficulty_specific_pred_indices#[index for index in ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Cyclist']
            self.filtered_pred_cyclist_count=len(valid_difficulty_specific_pred_indices)

            if difficulty=='Easy':
                self.easy_cylist_pred_indices=pred_label_difficulty_indices
            elif difficulty=='Moderate':
                self.moderate_cylist_pred_indices=pred_label_difficulty_indices
            elif difficulty=='Hard':
                self.hard_cylist_pred_indices=pred_label_difficulty_indices



def get_gt_classes_boxes(groundtruth):
    gt_classes=[groundtruth_detection_list[0]  for groundtruth_detection_list in groundtruth]
    gt_boxs=[(Point(float(groundtruth_detection_list[4]),float(groundtruth_detection_list[5])),Point(float(groundtruth_detection_list[6]),float(groundtruth_detection_list[7]))) for groundtruth_detection_list in groundtruth]

    return gt_classes,gt_boxs

def get_gt_truncation_occlusion(groundtruth):
    gt_truncation=[float(groundtruth_detection_list[1])  for groundtruth_detection_list in groundtruth ]
    gt_occlusion=[int(groundtruth_detection_list[2])  for groundtruth_detection_list in groundtruth]
    

    return gt_truncation,gt_occlusion


def get_gt_difficulty(gt_box,gt_truncation,gt_occlusion):

    box_height=gt_box[1].y-gt_box[0].y
    truncation=gt_truncation
    occlusion=gt_occlusion
    if box_height>=40 and occlusion==0 and truncation<=0.15:
        #easy_difficulty_index_list.append(i)

        return 'Easy'
    elif box_height>=25 and occlusion in [0,1] and truncation<=0.3:

        return 'Moderate'


    elif box_height>=25 and occlusion in [0,1,2] and truncation<=0.5:

        return 'Hard'

    else:
        return 'Ignored'


# Visualize Groundtruth 2D BBox
def plot_groundtruth(img,groundtruth):
    thickness=2
    groundtruth_img=img
    gt_classes,gt_boxs=get_gt_classes_boxes(groundtruth)
    gt_truncations,gt_occlusions=get_gt_truncation_occlusion(groundtruth)

    for gt_class,gt_box,gt_truncation,gt_occlusion in zip(gt_classes,gt_boxs,gt_truncations,gt_occlusions):

        gt_difficulty=get_gt_difficulty(gt_box,gt_truncation,gt_occlusion)


        if gt_class=='DontCare':
            color=(255,255,255)

        elif gt_class in ['Car','Cyclist','Pedestrian']:

            if gt_difficulty=='Ignored':
                color=(0,255,255)
            else:
                color=(0,255,0)


        else:
            color=(221,160,221)

        print('TL: ',((gt_box[0].x),(gt_box[0].y)))
        TL=(int(gt_box[0].x),int(gt_box[0].y))
        BR=(int(gt_box[1].x),int(gt_box[1].y))

        groundtruth_img=cv2.rectangle(groundtruth_img,TL,BR,color,thickness)

    
    return groundtruth_img


# Class For Evaluating Object Detector 
class metrics_evaluator:
    def __init__(self,object_detector,n_frames,logs_path,results_path):
        # Initial Config
        self.object_detector=object_detector
        self.labels=['Car','Cyclist','Pedestrian']
        self.difficulties=['Easy','Moderate','Hard']
        self.data=[]
        self.frame_id=0
        self.n_frames=n_frames

        # Placeholders for storing info
        self.gt_class_difficulty_count=[]
        self.filtered_gt_instances_count=[]
        self.gt_class_total_count=[]
        self.pred_class_difficulty_count=[]
        self.pred_class_total_count=[]
        self.filtered_pred_instances_count=[]
        self.logs_path=logs_path
        self.results_path=results_path
        self.ignored_gt_count=[]
        self.ignored_pred_count=[]







    
    # Cleans data according to label and difficulty provided as input
    def clean_data(self,groundtruth,predictions,label,difficulty):
        # Clean groundtruth
        # Get GT classes,boxs,truncation,occlusion in a list.
        self.gt_classes,self.gt_boxs=get_gt_classes_boxes(groundtruth)
        gt_truncations,gt_occlusions=get_gt_truncation_occlusion(groundtruth)

        # Placeholders for storing info
        valid_label_specific_gt_indices=[]
        dontcare_gt_indices=[]
        invalid_label_specific_indices=[]
        neighbouring_class_indices=[]

        self.filtered_gt_for_debugging_purposes_indices=[]


        # Stores valid gt index in list for debugging purposes
        for i,gt_box in enumerate(self.gt_boxs):
            computed_difficulty=get_gt_difficulty(gt_box=gt_box,gt_truncation=gt_truncations[i],gt_occlusion=gt_occlusions[i])

            if self.gt_classes[i]==label:
            # Filter By Difficulty
                if computed_difficulty=='Ignored':
                    pass
                    #print('gt ',i,' is ignored ')
                    
                else:
                    self.filtered_gt_for_debugging_purposes_indices.append(i)
                    #print('gt ',i,' is valid')
            else:
                pass


        # Sort groundtruth into valid_label_specific list,dontcare list, neighbouring_class list, and invalid_label_specific list
        for i,gt_class in enumerate(self.gt_classes):
            # Label is current class being evaluated
            if gt_class==label:
                valid_label_specific_gt_indices.append(i)
            # DontCare Classes due to small box height
            elif gt_class=='DontCare':
                dontcare_gt_indices.append(i)
            # Neighbouring Classes such as Van for Car and Person Sitting for Person
            elif gt_class=='Van' or gt_class=='Person Sitting':
                neighbouring_class_indices.append(i)
            # Groundtruth with label other than one currently being evaluated --> invalid
            else:
                invalid_label_specific_indices.append(i)

        
        valid_label_and_difficulty_specific_gt_indices=[]
        ignored_label_specific_gt_indices=[]

        # Previously we only filter according to label, now we filter according to difficulty
        # We only iterate over groundtruth with valid labels (Ones already added into valid_label_specific_gt_indices list)
        for valid_label_index in valid_label_specific_gt_indices:
            # Get groundtruth difficulty from box height, truncation and occlusion.
            computed_difficulty=get_gt_difficulty(gt_box=self.gt_boxs[valid_label_index],
                                                  gt_truncation=gt_truncations[valid_label_index],
                                                  gt_occlusion=gt_occlusions[valid_label_index])

            # Filter By Difficulty
            if computed_difficulty==difficulty:
                # If groundtruth matches label and difficulty seeked then add to valid_gt_list
                valid_label_and_difficulty_specific_gt_indices.append(valid_label_index)
            elif computed_difficulty=='Ignored':
                # If groundtruth matches label but not difficulty is ignored due to not fitting in Kitti Criteria add to ignored_gt_list
                ignored_label_specific_gt_indices.append(valid_label_index)

            else:
                # If groundtruth matches label but not difficulty --> pass
                pass


        # Clean Predictions
        self.pred_classes,self.pred_boxes=get_pred_classes_boxes(predictions)

        # Placeholders for predictions
        valid_difficulty_specific_pred_indices=[]
        ignored_difficulty_specific_pred_indices=[]


        for index,box in enumerate(self.pred_boxes):
            box_height=box[1].y-box[0].y

            # Ignore predictions with height less than the minimum
            if box_height>25:
                valid_difficulty_specific_pred_indices.append(index)

            else:
                ignored_difficulty_specific_pred_indices.append(index)
                

        # Define various indices list as global attributes to be used by the evaluator class

        # Groundtruth
        self.relevant_gt_indices=valid_label_and_difficulty_specific_gt_indices
        self.ignored_label_specific_gt_indices=ignored_label_specific_gt_indices
        self.dontcare_gt_indices=dontcare_gt_indices
        self.neighbouring_class_indices=neighbouring_class_indices

        # Predictions
        self.relevant_pred_indices=[index for index in valid_difficulty_specific_pred_indices if self.pred_classes[index]==label]
        self.ignored_difficulty_specific_pred_indices=ignored_difficulty_specific_pred_indices
        self.valid_difficulty_specific_pred_indices=[index for index in valid_difficulty_specific_pred_indices if self.pred_classes[index]==label]


        print("{} {} Relevant Gt Indices: ".format(label,difficulty),self.relevant_gt_indices)
    def get_ignored_gt_indices(self):
        # These are the indices of groundtruth objects that we use to crosscheck a potential fp
        # before we are sure that it's a fp, 
        # These are objects that have a height lower than the minimum height which are still detected by SMOKE
        # Or objects of neighbouring class which are detected by SMOKE as the other class
        all_ignored_indices=[]
        for index_list in [self.metrics_holder.ignored_gt_car_indices,self.metrics_holder.ignored_gt_cyclist_indices,self.metrics_holder.ignored_gt_pedestrian_indices,self.dontcare_gt_indices,self.neighbouring_class_indices]:
            if type(index_list)==list:
                all_ignored_indices.append(index_list)
        flatten_ignored_indices_list=[item for sublist in all_ignored_indices for item in sublist]   

        self.ignored_gt_indices=flatten_ignored_indices_list

    def check_fp_candidate(self,fp_candidate_index): 

        # Check IoU of Potential False Positives with Objects Filtered from Groundtruth
        fp_candidate_IoU_checks=[get_IoU(self.pred_boxes[fp_candidate_index],self.gt_boxs[ignored_index]) for ignored_index in self.ignored_gt_indices ]
        # If IoU is greater than 0.5 then prediction is not a false positive and is actually 
        # a detected object with height below minimum or an object of neighbouring class
        checks=[iou>0.5 for iou in fp_candidate_IoU_checks]

        # If we have at least 1 True in checks than unassigned prediction is not a False Positive
        if True in checks:
            fp_condition=False
        # If we only have False in Checks this means unassigned prediction is a False Positive    
        else:
            fp_condition=True

        return fp_condition

    def get_fp_difficulty_by_height(self,box):

        # FP difficulty category is determined by Bbox height

        box_height=box[1].y-box[0].y

        if box_height>40:
            return "Easy"
        elif 30<box_height<40:
            return "Moderate"
        elif 25<box_height<30:
            return "Hard"
        else:
            return "Ignored"




    def get_label_specific_metrics(self,difficulty):
        # Set Initial Metrics to 0
        TP=0
        FP=0
        FN=0

        # Stores prediction indices of valid label and difficulty
        self.pred_label_difficulty_indices=[]

        initial_relevant_gt_indices=self.relevant_gt_indices.copy()
        initial_relevant_pred_indices=self.relevant_pred_indices.copy()
        for relevant_gt_index in self.relevant_gt_indices:
            # Iterate over all valid predictions wrt to box height = relevant_pred
            for relevant_pred_index in self.relevant_pred_indices:
                # Get Intersection Over Union
                iou=get_IoU(self.gt_boxs[relevant_gt_index],self.pred_boxes[relevant_pred_index])

                # If iou greater than threshold then prediction is a potential TP (if predicted label is correct) and prediction has not been assigned
                if iou>=0.5:
                    # If predicted label matches groundtruth label AND prediction has not been assigned a groundtruth
                    if self.gt_classes[relevant_gt_index]==self.pred_classes[relevant_pred_index] and self.assigned_predictions[relevant_pred_index]==False and self.assigned_groundtruth[relevant_gt_index]==False:
                        # Then prediction is a True Positive
                        TP+=1
                        # Add index of TP prediction to the list self.pred_label_difficulty_indices=[]
                        self.pred_label_difficulty_indices.append(relevant_pred_index)
                        # Set assigned_predictions with index=relevant_pred_index to True since it's a TP
                        self.assigned_predictions[relevant_pred_index]=True
                        self.assigned_groundtruth[relevant_gt_index]=True

                    else:
                        pass
                else:
                    pass


        FP=len(self.pred_label_difficulty_indices)-TP
        FN=len(initial_relevant_gt_indices)-TP

        return len(initial_relevant_gt_indices),len(initial_relevant_pred_indices),TP,FP,FN


    def get_frame_metrics(self,groundtruth,predictions):

        # Initially all predictions are not assigned --> None of the predictions are assigned a gt --> False
        self.assigned_predictions=[False for i in range(len(self.total_pred_classes))]
        self.assigned_groundtruth=[False for i in range(len(self.total_gt_classes))]

        self.gt_class_difficulty_count_row=[]
        #self.pred_class_difficulty_count=[]
        for label in self.labels:
            for difficulty in self.difficulties:
                # Filter Groundtruth and Predictions according to label and difficulty
                self.clean_data(groundtruth,predictions,label,difficulty)
                # Get Metrics according to label and difficulty inputs
                class_gt_instances,class_pred_instances,TP,FP,FN=self.get_label_specific_metrics(difficulty=difficulty)
                # Add Metrics to Metrics Holder
                self.metrics_holder.add_to_metrics([TP,FP,FN],label=label,difficulty=difficulty)

                # Store Objects Stats For Debugging/Logging Purposes
                self.metrics_holder.store_stats(label,difficulty,
                    self.filtered_gt_for_debugging_purposes_indices,self.ignored_label_specific_gt_indices,
                    self.ignored_difficulty_specific_pred_indices,self.valid_difficulty_specific_pred_indices,
                    self.pred_label_difficulty_indices)

                # Add object count to list
                self.gt_class_difficulty_count_row.append(class_gt_instances)
                #self.pred_class_difficulty_count.append(class_pred_instances)
        
        # Get Unassigned Prediction indices to compute False Positives
        unassigned_prediction_indices=[i for i in range(len(self.assigned_predictions)) if self.assigned_predictions[i]==False ]

        # Iterate over every unassigned prediction 
        for unassigned_prediction_index in unassigned_prediction_indices:
            # Get Groundtruth objects to crosscheck potential fp with
            self.get_ignored_gt_indices()
            # Check if Unassigned Prediction is False Positive
            fp_condition=self.check_fp_candidate(fp_candidate_index=unassigned_prediction_index)
            
            if fp_condition==True:
                print("fp found")
                fp_difficulty=self.get_fp_difficulty_by_height(self.pred_boxes[unassigned_prediction_index])
                self.metrics_holder.add_to_metrics([0,1,0],label=self.total_pred_classes[unassigned_prediction_index],difficulty=fp_difficulty)




    def get_metrics_v2(self,groundtruth,predictions):
        data_row=[]
        
        gt_class_total_count_per_frame=[]
        pred_class_total_count_per_frame=[]

        # self.gt_class_difficulty_count_row=[]
        # self.pred_class_difficulty_count=[]

        # Get Classes for Count
        self.total_gt_classes,_=get_gt_classes_boxes(groundtruth)
        self.total_pred_classes,_=get_pred_classes_boxes(predictions)

        for l in self.labels:
            gt_class_total_count_per_frame.append(self.total_gt_classes.count(l))
            pred_class_total_count_per_frame.append(self.total_pred_classes.count(l))
        
        self.gt_class_total_count.append(gt_class_total_count_per_frame)
        self.pred_class_total_count.append(pred_class_total_count_per_frame)






        self.get_frame_metrics(groundtruth,predictions)






        
        self.data.append(self.metrics_holder.get_data_row())
        self.gt_class_difficulty_count.append(self.gt_class_difficulty_count_row)
        self.filtered_gt_instances_count.append([self.metrics_holder.car_filtered_gt_instances,self.metrics_holder.cyclist_filtered_gt_instances,self.metrics_holder.pedestrian_filtered_gt_instances])
        
        self.pred_class_difficulty_count.append([len(self.metrics_holder.easy_car_pred_indices),len(self.metrics_holder.moderate_car_pred_indices),len(self.metrics_holder.hard_car_pred_indices),
                                        len(self.metrics_holder.easy_cylist_pred_indices),len(self.metrics_holder.moderate_cylist_pred_indices),len(self.metrics_holder.hard_cylist_pred_indices),
                                        len(self.metrics_holder.easy_pedestrian_pred_indices),len(self.metrics_holder.moderate_pedestrian_pred_indices),len(self.metrics_holder.hard_pedestrian_pred_indices)])

        #print("pred class difficulty count: ",self.pred_class_difficulty_count)
        self.ignored_gt_count.append([len(self.metrics_holder.ignored_gt_car_indices),len(self.metrics_holder.ignored_gt_cyclist_indices),len(self.metrics_holder.ignored_gt_pedestrian_indices)])

        ####
        self.ignored_pred_count.append([len(self.metrics_holder.ignored_pred_car_indices),len(self.metrics_holder.ignored_pred_cyclist_indices),len(self.metrics_holder.ignored_pred_pedestrian_indices)])
        self.filtered_pred_instances_count.append([self.metrics_holder.filtered_pred_car_count,self.metrics_holder.filtered_pred_cyclist_count,self.metrics_holder.filtered_pred_pedestrian_count])
        

        self.frame_id+=1

        #print("gt class difficulty count: \n",self.gt_class_difficulty_count)

    def tabularize(self):
        if self.frame_id==self.n_frames:
            self.data=np.array(self.data)
            print('Data: ',self.data)
            #print('Data Array Size',self.data.shape)

            self.car_metrics,self.pedestrian_metrics,self.cyclists_metrics,self.difficulty_specific_metrics,self.n_object_classes,self.n_object_difficulties=tabularize_metrics(self.data,self.gt_class_total_count,self.filtered_gt_instances_count,self.gt_class_difficulty_count,self.ignored_gt_count,self.pred_class_total_count,self.filtered_pred_instances_count,self.pred_class_difficulty_count,self.ignored_pred_count,self.n_frames,self.logs_path)

            # print('N Object Classes: ',self.n_object_classes)
            # print("N Object Difficulties: ",self.n_object_difficulties)

        else:
            pass

    def eval_metrics(self,groundtruth,predictions):

        self.metrics_holder=MetricsHolder()

        self.get_metrics_v2(groundtruth,predictions)
        
        self.tabularize()

        if self.frame_id==self.n_frames:
            return self.car_metrics,self.pedestrian_metrics,self.cyclists_metrics,self.difficulty_specific_metrics,self.n_object_classes,self.n_object_difficulties
        else:
            return None,None,None,None,None,None

    def run_kitti_AP_evaluation_executable(self,root_dir,evaluation_executable_path,predictions_foldername):

        #evaluation_executable_path='.\SMOKE\smoke\data\datasets\evaluation\kitti\kitti_eval_40\eval8.exe'
        boxs_groundtruth_path=os.path.join(root_dir,'SMOKE/datasets/kitti/training/label_2')
        command = "{} {} {} {}".format(evaluation_executable_path,boxs_groundtruth_path, self.results_path.replace("/","\\"),predictions_foldername)#"C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\results\\Streamtest_AP_eval2022_12_23_15_05_34"

        # Run evaluation command from terminal
        average_precision_command=subprocess.check_output(command, shell=True, universal_newlines=True).strip()
        print(average_precision_command)

        # # Get AP from generated files (by previous command)
        # cars_easy_AP,cars_moderate_AP,cars_hard_AP=get_class_AP(results_path,'Car')
        # pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=get_class_AP(results_path,'Pedestrian')


        # cars_AP=[cars_easy_AP,cars_moderate_AP,cars_hard_AP]
        # pedestrians_AP=[pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP]

        # # Organize results in clear format + Get weighted average for categories with unavailable info
        # df,bar_metrics=construct_dataframe_v2(cars_AP,pedestrians_AP,metrics_evaluator.car_metrics,metrics_evaluator.pedestrian_metrics,metrics_evaluator.difficulty_metrics,metrics_evaluator.n_object_classes,metrics_evaluator.n_object_difficulties)

    def get_AP(self):
        cars_easy_AP,cars_moderate_AP,cars_hard_AP=get_class_AP(self.results_path,'Car')
        pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP=get_class_AP(self.results_path,'Pedestrian')

        #print('Cars AP: ',)


        self.cars_AP=[cars_easy_AP,cars_moderate_AP,cars_hard_AP]
        self.pedestrians_AP=[pedestrian_easy_AP,pedestrian_moderate_AP,pedestrian_hard_AP]

    def construct_dataframe(self):
        self.get_AP()
        df,self.bar_metrics=construct_dataframe_v2(self.object_detector,self.cars_AP,self.pedestrians_AP,self.car_metrics,self.pedestrian_metrics,self.difficulty_specific_metrics,self.n_object_classes,self.n_object_difficulties)

        dfi.export(df,os.path.join(self.results_path,'{}MetricsTable.png'.format(self.object_detector)))
        self.metrics_img=cv2.imread(os.path.join(self.results_path,'{}MetricsTable.png'.format(self.object_detector)))
        return df.data

    def show_results(self):

        cv2.imshow('Metrics',self.metrics_img)
        cv2.waitKey(0)

        self.bar_metrics.iloc[:,0:3].plot(kind='bar',title="{} AP Evaluation ".format(self.object_detector),figsize=(20, 8))
        plt.legend(loc=(-0.16,0.7))
        plt.xlabel("Metrics")
        plt.ylabel("Percentage %")
        plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
        plt.yticks(range(0,105,5))
        plt.savefig(os.path.join(self.results_path,"{}_bar_metrics_AP.png".format(self.object_detector)),dpi=600,bbox_inches="tight")
        plt.grid(True)
        plt.show()

        self.bar_metrics.iloc[:,3:6].plot(kind='bar',title="{} Precision Evaluation".format(self.object_detector),figsize=(20, 8))
        plt.legend(loc=(-0.16,0.7))
        plt.xlabel("Metrics")
        plt.ylabel("Percentage %")
        plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
        plt.yticks(range(0,105,5))
        plt.savefig(os.path.join(self.results_path,"{}_bar_metrics_precision.png".format(self.object_detector)),dpi=600,bbox_inches="tight")
        plt.grid(True)
        plt.show()

        self.bar_metrics.iloc[:,6:9].plot(kind='bar',title="{} Recall Evaluation".format(self.object_detector),figsize=(20, 8))
        plt.legend(loc=(-0.16,0.7))
        plt.xlabel("Metrics")
        plt.ylabel("Percentage %")
        plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
        plt.yticks(range(0,105,5))
        plt.savefig(os.path.join(self.results_path,"{}_bar_metrics_recall.png".format(self.object_detector)),dpi=600,bbox_inches="tight")
        plt.grid(True)
        plt.show()