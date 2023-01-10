import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def get_calculated_2d_points(K,pose,dtype=int):

    x_3d=pose[0]
    y_3d=pose[1]
    z_3d=pose[2]

    calculated_output=np.dot(K,np.array([[x_3d],[y_3d],[z_3d]]))
    if z_3d!=0:
        calculated_x=calculated_output[0]/z_3d
        # if prediction==True:
        #     print("Float Y Coordinate in Application: ",calculated_output[1]/z_3d)
        calculated_y=calculated_output[1]/z_3d

    float_coordinates=(calculated_x,calculated_y)
    int_coordinates=(int(calculated_x),int(calculated_y))

    if dtype==int:
        return int_coordinates
    else:
        return float_coordinates


def match_pred_2_gt_by_obj_center(pred_obj_center,gt_objs_center):

    error_list=[]
    for gt_obj_center in gt_objs_center:
        x_error=(pred_obj_center[0]-gt_obj_center[0])**2
        y_error=(pred_obj_center[1]-gt_obj_center[1])**2
        error=x_error+y_error
        error_list.append(error)

    gt_object_match_index=error_list.index(min(error_list))

    return gt_object_match_index
class depth_evaluator:
    def __init__(self,n_frames,results_path):
        print("Initialized Depth Evaluator")
        #self.K=K
        self.results_path=results_path
        self.n_frames=n_frames
        self.predicted_depth=[]
        self.groundtruth_depth=[]
        self.pred_gt_matched_depth_list=[]
        self.internal_counter=0

    def match_pred_with_gt_by_object_center(self,K,groundtruth,predictions):
        self.K=K
        self.gt_objs_center=[get_calculated_2d_points(self.K,[obj.tx,obj.ty-(obj.h/2),obj.tz],dtype=int) for obj in groundtruth]
        self.pred_objs_center=[get_calculated_2d_points(self.K,[obj.tx,obj.ty-(obj.h/2),obj.tz],dtype=int) for obj in predictions]

        for i,pred_obj_center in enumerate(self.pred_objs_center):
            print("pred obj center: ",pred_obj_center)
            print("gt objs center list")
            gt_match_index=match_pred_2_gt_by_obj_center(pred_obj_center,self.gt_objs_center)
            self.pred_gt_matched_depth_list.append((predictions[i].tz,groundtruth[gt_match_index].tz))

        self.internal_counter+=1

    def viz_object_centers(self,img):

        for obj_center in self.gt_objs_center:
            img=cv2.circle(img,obj_center , 0, (0,255,0), 4)

        for obj_center in self.pred_objs_center:
                img=cv2.circle(img,obj_center , 0, (0,0,255), 4)

        cv2.imshow("Visualize Obj Centers",img)
        cv2.waitKey(0)

    def plot_eror(self,method):

        if self.internal_counter==self.n_frames:
            x=[item[1] for item in self.pred_gt_matched_depth_list]
            y=[np.abs(item[1]-item[0]) for item in self.pred_gt_matched_depth_list]
            indices_sorted=np.argsort(x)
            x_sorted=[x[i] for i in indices_sorted]
            y_sorted=[y[i] for i in indices_sorted]

            # #plt.figure().clear()
            # #plt.close()
            # fig, ax = plt.subplots()  # Create a figure containing a single axes.
            # #ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # Plot some data on the axes.
            # ax.scatter(x_sorted,y_sorted)
            # plt.ylabel("Error [m]")
            # plt.xlabel("GroundTruth [m]")
            # plt.xlim([0,60])
            # plt.ylim([0,15])
            
            #x.sort()



            y_smoothed=[]
            range_coeffs=[]
            for i in range(0,7):
                
                    
                a,b=np.polyfit(x_sorted[(i*10):(i+1)*10], y_sorted[(i*10):(i+1)*10], 1)
                range_coeffs.append((a,b))
                print("Range: ","{} - {}".format(i*10,(i+1)*10))
                print("(a,b): ",(a,b))
                # y_smooth=a*(i*10)+(b)
                # y_smoothed.append(y_smooth)

            for i,x in enumerate(list(range(0,70,10))):
                if i==0 or i==1:
                    a,b=range_coeffs[0][0],range_coeffs[0][1]
                    y_smooth=(a*x)+(b)
                # elif i==1:
                #     a,b=range_coeffs[i-1][0],range_coeffs[i-1][1]
                #     y_smooth=(a*x)+(b)
                #     print("Range: ","{} - {}".format(i*10,(i+1)*10))
                #     print("(a,b): ",(a,b))
                else:
                    a,b=range_coeffs[i][0],range_coeffs[i][1]
                    y_smooth=(a*x)+(b)


                y_smoothed.append(y_smooth)


            #fig,ax = plt.subplots(nrows=1,ncols=1)
            # ax.plot(list(range(0,70,10)),y_smoothed)
            # #plt.plot(list(range(0,70,10)),y_smoothed)
            # plt.ylabel("Error [m]")
            # plt.xlabel("GroundTruth [m]")
            # plt.xlim([0,60])
            # plt.ylim([0,max(y_smoothed)+1])
            # plt.show()
            #fig.savefig(os.path.join(self.results_path,"Depth Evaluation via Linear Regression.png"))

            average_y=[]
            x_for_average_error=[]

            for i in range(7):
                #print("X Sorted: ",x_sorted)
                #print("Y Sorted: ",y_sorted)
                x_range_indices=[x_sorted.index(x) for x in x_sorted if (0*i)<=x<((i+1)*10)]
                x_for_average_error.append(i)
                y_for_range=[y_sorted[i] for i in x_range_indices]
                #if len(y_for_range)!=0:
                average_for_range=np.mean(y_for_range)
                #if average_for_range=="nan":
                if i==0:
                    print("Method: ",method)
                    print("X Sorted: ",x_sorted)
                    print("mean for range: ",average_for_range)
                # print("x for range: ",[x_sorted[i] for i in x_range_indices])
                # print("y for range: ",y_for_range)
                average_y.append(average_for_range)
                # else:
                #     average_y.append(0)


            average_plot_labels=["{}-{}".format(i*10,(i+1)*10) for i in range(0,7)]
            # fig,ax = plt.subplots(nrows=1,ncols=1)
            # ax.plot(list(range(1,8)),average_y)
            # #plt.plot(list(range(1,8)),average_y)
            # plt.xticks(ticks=list(range(1,8)),labels=average_plot_labels)
            # plt.ylabel("Average Error [m]")
            # plt.xlabel("GroundTruth Range [m]")
            # plt.xlim([0,8])
            # plt.ylim([0,20])
            #plt.show()
            #fig.savefig(os.path.join(self.results_path,"Depth Evaluation via Range Average.png"))

            x_filtered_indices=[x_for_average_error.index(x) for x in x_for_average_error if math.isnan(x)==False]
            x_filtered=[x_for_average_error[i] for i in x_filtered_indices]
            y_filtered=[average_y[i] for i in x_filtered_indices]
            # # Count nan
            count_nan=0
            for y in y_filtered:
                if math.isnan(y)==True:
                    count_nan+=1

            #count_nan=y_filtered.count(math.nan)
            if count_nan!=0:
                x_filtered=x_filtered[count_nan:]
                x_filtered=[x+1 for x in x_filtered]
                y_filtered=[average_y[i] for i in x_filtered_indices]
                y_filtered=y_filtered[count_nan:]
            else:
                y_filtered=[average_y[i] for i in x_filtered_indices]


            return x_filtered,y_filtered
        else:
            return None,None

        
        

        



    

