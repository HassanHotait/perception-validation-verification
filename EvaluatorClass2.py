

from metrics_functions_from_evaluation_script import tabularize_metrics,Point,get_pred_classes_boxes,get_IoU
import numpy as np
import cv2

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


class metrics_evaluator:
    def __init__(self,n_frames,results_path):
        self.labels=['Car','Cyclist','Pedestrian']
        self.difficulties=['Easy','Moderate','Hard']
        self.data=[]#np.zeros((n_frames ,len(self.difficulties)*len(self.labels)*3))
        self.frame_id=0
        self.n_frames=n_frames
        self.gt_class_difficulty_count=[]
        self.filtered_gt_instances_count=[]
        self.gt_class_total_count=[]
        self.pred_class_difficulty_count=[]
        self.pred_class_total_count=[]
        self.filtered_pred_instances_count=[]
        self.results_path=results_path

        self.ignored_gt_count=[]
        self.ignored_pred_count=[]

    


    def clean_data(self,groundtruth,predictions,label,diffculty):
        self.gt_classes,self.gt_boxs=get_gt_classes_boxes(groundtruth)
        gt_truncations,gt_occlusions=get_gt_truncation_occlusion(groundtruth)

        valid_label_specific_gt_indices=[]
        dontcare_gt_indices=[]
        invalid_label_specific_indices=[]
        neighbouring_class_indices=[]

        self.filtered_gt_for_debugging_purposes_indices=[]


        # Only for debugging purposes

        print('Gt Cars: ',len([lbl for lbl in self.gt_classes if lbl=='Car']))
        
        for i,gt_box in enumerate(self.gt_boxs):
            computed_difficulty=get_gt_difficulty(gt_box=gt_box,gt_truncation=gt_truncations[i],gt_occlusion=gt_occlusions[i])

            if self.gt_classes[i]==label:
            # Filter By Difficulty
                if computed_difficulty=='Ignored':
                    print('gt ',i,' is ignored ')
                    
                else:
                    self.filtered_gt_for_debugging_purposes_indices.append(i)
                    print('gt ',i,' is valid')
            else:
                pass






        for i,gt_class in enumerate(self.gt_classes):
            # Label is current class being evaluated
            if gt_class==label:
                valid_label_specific_gt_indices.append(i)

            # DontCare Classes due to small box height
            elif gt_class=='DontCare':
                dontcare_gt_indices.append(i)

            elif gt_class=='Van' or gt_class=='Person Sitting':
                neighbouring_class_indices.append(i)


            else:
                invalid_label_specific_indices.append(i)

        
        valid_label_and_difficulty_specific_gt_indices=[]
        ignored_label_specific_gt_indices=[]
        for valid_label_index in valid_label_specific_gt_indices:
            computed_difficulty=get_gt_difficulty(gt_box=self.gt_boxs[valid_label_index],gt_truncation=gt_truncations[valid_label_index],gt_occlusion=gt_occlusions[valid_label_index])

            # Filter By Difficulty
            if computed_difficulty==diffculty:
                valid_label_and_difficulty_specific_gt_indices.append(valid_label_index)
            elif computed_difficulty=='Ignored':
                ignored_label_specific_gt_indices.append(valid_label_index)

            else:
                pass


        self.pred_classes,self.pred_boxes=get_pred_classes_boxes(predictions)




        # for i,pred_class in enumerate(self.pred_classes):
        #     if pred_class==label:
        #         valid_label_specific_pred_indices.append(i)

        #     else:
                # invalid_label_specific_pred_indices.append(i)

        #self.valid_label_specific_pred_indices=[]

        valid_difficulty_specific_pred_indices=[]
        ignored_difficulty_specific_pred_indices=[]

        # for index,box in enumerate(self.pred_boxes):
        #     box_height=box[1].y-box[0].y

        #     if box_height>40:
        #         computed_difficulty='Easy'

        #     elif 30<box_height<40:
        #         computed_difficulty='Moderate'

        #     elif 25<box_height<30:
        #         computed_difficulty='Hard'
        
        #     else:
        #         computed_difficulty='Ignored'

            
        #     if computed_difficulty==diffculty:
        #         valid_difficulty_specific_pred_indices.append(index)

        #     elif computed_difficulty=='Ignored':
        #         ignored_difficulty_specific_pred_indices.append(index)

        #     else:
        #         pass
        #######################################################################
        for index,box in enumerate(self.pred_boxes):
            box_height=box[1].y-box[0].y

            
            if box_height>25:
                valid_difficulty_specific_pred_indices.append(index)

            # elif computed_difficulty=='Ignored':
            #     ignored_difficulty_specific_pred_indices.append(index)

            else:
                ignored_difficulty_specific_pred_indices.append(index)
                


        self.relevant_gt_indices=valid_label_and_difficulty_specific_gt_indices
        self.ignored_label_specific_gt_indices=ignored_label_specific_gt_indices
        self.relevant_pred_indices=[index for index in valid_difficulty_specific_pred_indices if self.pred_classes[index]==label]
        self.ignored_difficulty_specific_pred_indices=ignored_difficulty_specific_pred_indices
        self.dontcare_gt_indices=dontcare_gt_indices
        self.neighbouring_class_indices=neighbouring_class_indices

        self.valid_difficulty_specific_pred_indices=[index for index in valid_difficulty_specific_pred_indices if self.pred_classes[index]==label]
        


    def get_label_specific_metrics(self,label,difficulty):
        TP=0
        FP=0
        FN=0

        self.pred_label_difficulty_indices=[]

        for relevant_gt_index in self.relevant_gt_indices:
            for relevant_pred_index in self.relevant_pred_indices:
                iou=get_IoU(self.gt_boxs[relevant_gt_index],self.pred_boxes[relevant_pred_index])

                if iou>=0.5:

                    if self.gt_classes[relevant_gt_index]==self.pred_classes[relevant_pred_index] and self.assigned_predictions[relevant_pred_index]==False:
                        TP+=1

                        self.pred_label_difficulty_indices.append(relevant_pred_index)
                        self.assigned_predictions[relevant_pred_index]=True
                        self.relevant_pred_indices.remove(relevant_pred_index)



                    else:
                        pass

                
                else:
                    pass

        #relevant_label_pred_indices=[index for index in self.relevant_pred_indices if self.pred_classes[index]==label]
        unassigned_pred_indices=[index for index in self.relevant_pred_indices if self.assigned_predictions[index]==False]

        if label=='Car':
            #print('Total Pred Indices: ',self.valid_difficulty_specific_pred_indices+unassigned_pred_indices)
            print('TP Predictions ',difficulty, '  Indices: ',self.pred_label_difficulty_indices)
            print('Unassigned Pred Indices: ',unassigned_pred_indices)

        # categorized_fps=[]
        # easy_fps=[]
        # moderate_fps=[]
        # hard_fps=[]
        # for i in unassigned_pred_indices:
        #     box_height=self.pred_boxes[i][1].y-self.pred_boxes[i][0].y

        #     if box_height>40:
        #         easy_fps.append(i)
            
        #     elif 30<box_height<40:
        #         moderate_fps.append(i)
        #     else:
        #         hard_fps.append(i)

        
        # if difficulty=='Easy':
        #     categorized_fps=easy_fps
        # elif difficulty=='Moderate':
        #     categorized_fps=moderate_fps
        # else:
        #     categorized_fps=hard_fps

        
        # self.pred_label_difficulty_indices+=categorized_fps

        # if label=='Car':
        #     print('Unassigned Pred ',difficulty, ' Indices: ',categorized_fps)
        #     print(difficulty,' Pred Indices: ',self.pred_label_difficulty_indices)
        FP=len(self.pred_label_difficulty_indices)-TP
        FN=len(self.relevant_gt_indices)-TP

        return len(self.relevant_gt_indices),len(self.relevant_pred_indices),TP,FP,FN,unassigned_pred_indices





    def get_metrics(self,groundtruth,predictions):
        data_row=[]
        gt_class_difficulty_count_row=[]
        gt_class_total_count_per_frame=[]
        pred_class_difficulty_count=[]
        pred_class_total_count_per_frame=[]
        total_gt_classes,_=get_gt_classes_boxes(groundtruth)
        total_pred_classes,_=get_pred_classes_boxes(predictions)
        for l in self.labels:
            gt_class_total_count_per_frame.append(total_gt_classes.count(l))
            pred_class_total_count_per_frame.append(total_pred_classes.count(l))
        self.gt_class_total_count.append(gt_class_total_count_per_frame)
        self.pred_class_total_count.append(pred_class_total_count_per_frame)

        self.ignored_gt_car_indices=0
        self.ignored_gt_pedestrian_indices=0
        self.ignored_gt_cyclist_indices=0

        self.ignored_pred_car_indices=0
        self.ignored_pred_pedestrian_indices=0
        self.ignored_pred_cyclist_indices=0

        self.easy_car_pred_indices=[]
        self.moderate_car_pred_indices=[]
        self.hard_car_pred_indices=[]

        self.easy_cyclist_pred_indices=[]
        self.moderate_cyclist_pred_indices=[]
        self.hard_cyclist_pred_indices=[]

        self.easy_pedestrian_pred_indices=[]
        self.moderate_pedestrian_pred_indices=[]
        self.hard_pedestrian_pred_indices=[]

        



        for label in self.labels:
            self.assigned_predictions=[False for i in range(len(total_pred_classes))]
            for difficulty in self.difficulties:
                self.clean_data(groundtruth,predictions,label,difficulty)
                #print('Pred Indices with valid heght: ',self.relevant_pred_indices)
                class_gt_instances,class_pred_instances,TP,FP,FN,unassigned_predictions=self.get_label_specific_metrics(label,difficulty)

                


                data_row.extend([TP,FP,FN])
                gt_class_difficulty_count_row.append(class_gt_instances)
                pred_class_difficulty_count.append(class_pred_instances)

                if label=='Car':

                    self.car_filtered_gt_instances=len(self.filtered_gt_for_debugging_purposes_indices)

                    if difficulty=='Easy':
                        self.easy_car_pred_indices=self.pred_label_difficulty_indices
                        print('Easy Car Pred Indices: ',self.easy_car_pred_indices)
                    elif difficulty=='Moderate':
                        self.moderate_car_pred_indices=self.pred_label_difficulty_indices
                        print('Moderate Car Pred Indices: ',self.moderate_car_pred_indices)
                    elif difficulty=='Hard':
                        self.hard_car_pred_indices=self.pred_label_difficulty_indices
                        print('Hard Car Pred Indices: ',self.hard_car_pred_indices)


                elif label=='Pedestrian':
                    self.pedestrian_filtered_gt_instances=len(self.filtered_gt_for_debugging_purposes_indices)
                    if difficulty=='Easy':
                        self.easy_pedestrian_pred_indices=self.pred_label_difficulty_indices
                    elif difficulty=='Moderate':
                        self.moderate_pedestrian_pred_indices=self.pred_label_difficulty_indices
                    elif difficulty=='Hard':
                        self.hard_pedestrian_pred_indices=self.pred_label_difficulty_indices

                elif label=='Cyclist':
                    self.cyclist_filtered_gt_instances=len(self.filtered_gt_for_debugging_purposes_indices)

                    if difficulty=='Easy':
                        self.easy_cylist_pred_indices=self.pred_label_difficulty_indices
                    elif difficulty=='Moderate':
                        self.moderate_cylist_pred_indices=self.pred_label_difficulty_indices
                    elif difficulty=='Hard':
                        self.hard_cylist_pred_indices=self.pred_label_difficulty_indices


            
            if label=='Car':
                self.ignored_gt_car_indices=self.ignored_label_specific_gt_indices
                self.ignored_pred_car_indices=self.ignored_difficulty_specific_pred_indices#[index for index in self.ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Car']
                filtered_pred_car_count=len(self.valid_difficulty_specific_pred_indices)

            elif label=='Pedestrian':
                self.ignored_gt_pedestrian_indices=self.ignored_label_specific_gt_indices
                self.ignored_pred_pedestrian_indices=[index for index in self.ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Pedestrian']
                filtered_pred_pedestrian_count=len(self.valid_difficulty_specific_pred_indices)

            elif label=='Cyclist':
                self.ignored_gt_cyclist_indices=self.ignored_label_specific_gt_indices
                self.ignored_pred_cyclist_indices=[index for index in self.ignored_difficulty_specific_pred_indices if self.pred_classes[index]=='Cyclist']
                filtered_pred_cyclist_count=len(self.valid_difficulty_specific_pred_indices)

            
            for unassigned_prediction_index in unassigned_predictions:


                # Check IoU of fp candidate with all DontCare Areas

                print('Ignored Car Indices: ',self.ignored_gt_car_indices)
                print('Ignored Cyclist Indices: ',self.ignored_gt_cyclist_indices)
                print('Ignored Pedestrian Indices: ',self.ignored_gt_pedestrian_indices)

                all_ignored_indices=[]
                for index_list in [self.ignored_gt_car_indices,self.ignored_gt_cyclist_indices,self.ignored_gt_pedestrian_indices,self.dontcare_gt_indices,self.neighbouring_class_indices]:
                    if type(index_list)==list:
                        all_ignored_indices.append(index_list)

                flatten_ignored_indices_list=[item for sublist in all_ignored_indices for item in sublist]
                fp_candidate_IoU_checks=[get_IoU(self.pred_boxes[unassigned_prediction_index],self.gt_boxs[ignored_index]) for ignored_index in flatten_ignored_indices_list ]
                checks=[iou>0.5 for iou in fp_candidate_IoU_checks]

                if True in checks:
                    fp_condition=False
                else:
                    fp_condition=True


                if self.pred_classes[unassigned_prediction_index]=='Car':

                    box_height=self.pred_boxes[unassigned_prediction_index][1].y-self.pred_boxes[unassigned_prediction_index][0].y

                    if box_height>40:
                        self.easy_car_pred_indices.append(unassigned_prediction_index)
                        print(len(data_row))
                        print(data_row)
                        if fp_condition==True:
                            data_row[1]+=1
                    
                    elif 30<box_height<40:
                        self.moderate_car_pred_indices.append(unassigned_prediction_index)#self.pred_label_difficulty_indices
                        if fp_condition==True:
                            data_row[4]+=1
                    else:
                        self.hard_car_pred_indices.append(unassigned_prediction_index)
                        if fp_condition==True:
                            data_row[7]+=1
                elif self.pred_classes[unassigned_prediction_index]=='Cyclist':

                    box_height=self.pred_boxes[unassigned_prediction_index][1].y-self.pred_boxes[unassigned_prediction_index][0].y

                    if box_height>40:
                        self.easy_cyclist_pred_indices.append(unassigned_prediction_index)
                        if fp_condition==True:
                            data_row[10]+=1
                    
                    elif 30<box_height<40:
                        self.moderate_cyclist_pred_indices.append(unassigned_prediction_index)#self.pred_label_difficulty_indices
                        if fp_condition==True:
                            data_row[13]+=1
                    else:
                        self.hard_cyclist_pred_indices.append(unassigned_prediction_index)
                        if fp_condition==True:
                            data_row[16]+=1

                else:

                    box_height=self.pred_boxes[unassigned_prediction_index][1].y-self.pred_boxes[unassigned_prediction_index][0].y

                    if box_height>40:
                        self.easy_pedestrian_pred_indices.append(unassigned_prediction_index)
                        if fp_condition==True:
                            data_row[19]+=1
                    
                    elif 30<box_height<40:
                        self.moderate_pedestrian_pred_indices.append(unassigned_prediction_index)#self.pred_label_difficulty_indices
                        if fp_condition==True:
                            data_row[22]+=1
                    else:
                        self.hard_pedestrian_pred_indices.append(unassigned_prediction_index)
                        if fp_condition==True:
                            data_row[25]+=1


        
        self.data.append(data_row)
        self.gt_class_difficulty_count.append(gt_class_difficulty_count_row)
        self.filtered_gt_instances_count.append([self.car_filtered_gt_instances,self.cyclist_filtered_gt_instances,self.pedestrian_filtered_gt_instances])
        
        self.pred_class_difficulty_count.append([len(self.easy_car_pred_indices),len(self.moderate_car_pred_indices),len(self.hard_car_pred_indices),
                                        len(self.easy_cylist_pred_indices),len(self.moderate_cylist_pred_indices),len(self.hard_cylist_pred_indices),
                                        len(self.easy_pedestrian_pred_indices),len(self.moderate_pedestrian_pred_indices),len(self.hard_pedestrian_pred_indices)])

        self.ignored_gt_count.append([len(self.ignored_gt_car_indices),len(self.ignored_gt_cyclist_indices),len(self.ignored_gt_pedestrian_indices)])

        ####
        self.ignored_pred_count.append([len(self.ignored_pred_car_indices),len(self.ignored_pred_cyclist_indices),len(self.ignored_pred_pedestrian_indices)])
        self.filtered_pred_instances_count.append([filtered_pred_car_count,filtered_pred_cyclist_count,filtered_pred_pedestrian_count])
        

        self.frame_id+=1

    def tabularize(self):
        if self.frame_id==self.n_frames:
            self.data=np.array(self.data)
            print('Data: ',self.data)
            print('Data Array Size',self.data.shape)


            self.car_metrics,self.pedestrian_metrics,self.cyclists_metrics,self.difficulty_specific_metrics,self.n_object_classes,self.n_object_difficulties=tabularize_metrics(self.data,self.gt_class_total_count,self.filtered_gt_instances_count,self.gt_class_difficulty_count,self.ignored_gt_count,self.pred_class_total_count,self.filtered_pred_instances_count,self.pred_class_difficulty_count,self.ignored_pred_count,self.n_frames,self.results_path)

        else:
            pass

    def evaluate_metrics(self,groundtruth,predictions):
        self.get_metrics(groundtruth,predictions)
        
        self.tabularize()

        if self.frame_id==self.n_frames:
            return self.car_metrics,self.pedestrian_metrics,self.cyclists_metrics,self.difficulty_specific_metrics,self.n_object_classes,self.n_object_difficulties
        else:
            return None,None,None,None,None,None