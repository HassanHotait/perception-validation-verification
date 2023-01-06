from metrics_functions_from_evaluation_script import new_read_groundtruth
import numpy as np
import os
import csv
#from test_manuallly_obtained_matrix import get_calculated_2d_points
from EvaluatorClass3 import new_plot_groundtruth
import cv2
import matplotlib.pyplot as plt

## SMOKE Modules
from SMOKE.smoke.config import cfg
from SMOKE.smoke.engine import default_argument_parser
from SMOKE.tools.my_functions import setup_network,preprocess_then_predict
from SMOKE.smoke.utils.check_point import DetectronCheckpointer
########################################################################

from Dataset2Kitti.SMOKE_Visualizer import detectionInfo,SMOKE_Viz


def get_angle_deg_by_FOV(img_shape,Yc,FOV_v):
    img_height=img_shape[0]

    # Y Coordinate above horizontal centerline
    # print("Input Y Coordinate: ",Yc)
    # print("Img Height Used: ",img_height)
    if Yc<(img_height/2):

        pixel_position_wrt_hor_centerline=(img_height/2)-Yc
        ratio=pixel_position_wrt_hor_centerline/(img_height/2)
        alpha=(90-(FOV_v/2)*ratio)

    else:
        # Y Coordinate Below horizontal centerline

        pixel_position_wrt_hor_centerline=Yc-(img_height/2)
        ratio=pixel_position_wrt_hor_centerline/(img_height/2)
        alpha=-(90-(FOV_v/2)*ratio)

    return alpha

def get_angle_deg_by_dims(depth,height):

    if height!=0:

        if height>0:
            alpha=-np.degrees(np.arctan(depth/height))

        if height<0:
            alpha=np.degrees(np.arctan(depth/np.abs(height)))

    else:
        alpha=0



    return alpha


def match_pred_2_gt_by_obj_center(pred_obj_center,gt_objs_center):

    error_list=[]
    for gt_obj_center in gt_objs_center:
        x_error=(pred_obj_center[0]-gt_obj_center[0])**2
        y_error=(pred_obj_center[1]-gt_obj_center[1])**2
        error=x_error+y_error
        error_list.append(error)

    gt_object_match_index=error_list.index(min(error_list))

    return gt_object_match_index

def get_K(file_name,calib_dir):

    with open(os.path.join(calib_dir, file_name), 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=' ')
                for line, row in enumerate(reader):
                    if row[0] == 'P2:':
                        K = row[1:]
                        K = [float(i) for i in K]
                        #print('K array: ',K)
                        #print('K elements length',len(K))
                        K = np.array(K, dtype=np.float32).reshape(3, 4)
                        P2=K
                        #print('K Shape',)
                        #print('K after reshaping (3,4)',K)
                        K = K[:3, :3]
                        #print('Input K: ',K)
                        break
                    # if 'P2' in line:
                    #     P2=line.split(' ')
                    #     P2 = np.asarray([float(i) for i in P2[1:]])
                    #     P2 = np.reshape(P2, (3, 4))

    return K,P2

def get_calculated_2d_points(K,pose,prediction,dtype=int):

    x_3d=pose[0]
    y_3d=pose[1]
    z_3d=pose[2]

    calculated_output=np.dot(K,np.array([[x_3d],[y_3d],[z_3d]]))
    calculated_x=calculated_output[0]/z_3d
    if prediction==True:
        print("Float Y Coordinate in Application: ",calculated_output[1]/z_3d)
    calculated_y=calculated_output[1]/z_3d

    float_coordinates=(calculated_x,calculated_y)
    int_coordinates=(int(calculated_x),int(calculated_y))

    if dtype==int:
        return int_coordinates
    else:
        return float_coordinates


def get_optimized_depth(error,depth_pred):

    if error>0:
        gain=1.1
    else:
        gain=1.1
    print("Gain: ",gain)

    depth_optimized=depth_pred*gain

    return depth_optimized


alpha_error_list=[]
depth_error_list=[]
depth_optimized_error_list=[]

for i in range(10):
    id=i
    labels_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\Dataset2Kitti\\PrescanRawData_Scenario15\\DataLogger1\\labels_2"
    groundtruth=new_read_groundtruth(gt_folder=labels_path,fileid=id,extension=".csv",dataset="Prescan")

    K=np.array([[935 ,  0  ,       399],
                [  0   ,      935 ,298],
                [  0     ,      0      ,     1       ]],dtype=np.float32)

    K=np.array([[937.5 ,  0  ,      400 ],
                [  0   ,      937.5 ,300],
                [  0     ,      0      ,     1       ]],dtype=np.float32)



    img_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\Dataset2Kitti\\PrescanRawData_Scenario15\\DataLogger1\\images_2\\"+str(id).zfill(6)+".png"
    img=cv2.imread(img_path)
    print("Img Shape: ",img.shape)
    print("K: ",K)
    print("Groundtruth: ",groundtruth)
    print("len(gt): ",len(groundtruth))

    gt_img=new_plot_groundtruth(img,groundtruth=groundtruth)
    camera_height=1.32

    fy=K[1][1]
    cy=K[1][2]

    print("fy: ",fy)
    print("cy: ",cy)



    # Setup SMOKE

    args = default_argument_parser().parse_args()
    model,cfg,gpu_device,cpu_device=setup_network(args)
    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)
    # Predict SMOKE


    smoke_predictions_list,K=preprocess_then_predict(model,cfg,id,img_path,gpu_device,cpu_device,dataset="Prescan")

    detection_info_list=[detectionInfo(pred) for pred in smoke_predictions_list]
    detection_info_list_optimized=[detectionInfo(pred) for pred in smoke_predictions_list]

    


    gt_objects_center=[]
    for obj in groundtruth:

        obj_center=get_calculated_2d_points(K,[obj.tx,obj.ty-(obj.h/2),obj.tz],prediction=False,dtype=float)
        gt_objects_center.append(obj_center)


    print("-------------------------------------- Predictions ----------------------------------------------------")
    pred_obj_center=[]
    for pred_obj in detection_info_list:
        obj_center=get_calculated_2d_points(K,[pred_obj.tx,camera_height-(pred_obj.h/2),pred_obj.tz],prediction=True,dtype=float)
        pred_obj_center.append(obj_center)

    for i,pred_obj in enumerate(detection_info_list):
        # Plot Predicted Obj Center
        (x,y)=(int(pred_obj_center[i][0]),int(pred_obj_center[i][1]))
        gt_img=cv2.circle(gt_img,(x,y) , 0, (0,0,255), 2)
        # Plot Matched GroundTruth Obj Center
        matched_gt_index=match_pred_2_gt_by_obj_center(pred_obj_center[i],gt_objects_center)
        matched_gt=groundtruth[matched_gt_index]
        
        (x,y)=(int(gt_objects_center[matched_gt_index][0]),int(gt_objects_center[matched_gt_index][1]))
        gt_img=cv2.circle(gt_img,(x,y) , 0, (0,255,0), 2)


        # Y Coordinate is calculated according to [X,Y,Z] center of the object,hence Y
        Z_pred_calculation=((camera_height-(pred_obj.h/2))*fy)/(pred_obj_center[i][1]-cy)

        Z_pred_calculation_ideal_Y=((camera_height-(pred_obj.h/2))*fy)/(gt_objects_center[matched_gt_index][1]-cy)
        Z_pred_calculation_ideal_height=((camera_height-(matched_gt.h/2))*fy)/(pred_obj_center[i][1]-cy)
        

        matched_gt_index=match_pred_2_gt_by_obj_center(pred_obj_center[i],gt_objects_center)
        matched_gt=groundtruth[matched_gt_index]

        Z_gt_calculation=((camera_height-(matched_gt.h/2))*fy)/(gt_objects_center[matched_gt_index][1]-cy)

        FOV_v=35.49
        gt_alpha_wrt_cc_by_dims=get_angle_deg_by_dims(matched_gt.tz,matched_gt.ty)
        gt_alpha_wrt_cc_by_fov=get_angle_deg_by_FOV(img.shape,gt_objects_center[matched_gt_index][1],FOV_v=FOV_v)
        print("Groundtruth Depth: ",matched_gt.tz)
        print("Groundtruth Angle wrt Camera Coordinate System: ",gt_alpha_wrt_cc_by_dims)
        print("Groundtruth Angle Calculated wrt Camera Coordinate System: ",gt_alpha_wrt_cc_by_fov)
        pred_alpha_wrt_cc_by_dims=get_angle_deg_by_dims(pred_obj.tz,pred_obj.ty)
        pred_alpha_wrt_cc_by_fov=get_angle_deg_by_FOV(img.shape,pred_obj_center[i][1],FOV_v=FOV_v)
        print("FOV Angle via Prediction Dimensions: ",pred_alpha_wrt_cc_by_dims)
        print("FOV Angle via Predicted Y Coordinate: ",pred_alpha_wrt_cc_by_fov)
        # Truth - Prediction
        alpha_error=pred_alpha_wrt_cc_by_fov-pred_alpha_wrt_cc_by_dims
        alpha_error_list.append(alpha_error)

        depth_optimized=get_optimized_depth(alpha_error,Z_pred_calculation)
        error_optimized=depth_optimized-Z_gt_calculation
        error_pre=Z_pred_calculation-Z_gt_calculation
        depth_error_list.append(Z_pred_calculation)
        print("Pred Depth Optimized: {} - Pred Depth  {}- Groundtruth Depth: {} - Error Optimized: {} + Error Pre: {} - alpha Error: {} ".format(depth_optimized,
                                                                                                    Z_pred_calculation,
                                                                                                    Z_gt_calculation,
                                                                                                    error_optimized,
                                                                                                    error_pre,
                                                                                                    alpha_error))

        optimized_pred=detection_info_list_optimized[i]
        optimized_pred.tz=depth_optimized[0]
        
        print("Predicted Angle Error Input For Controller: ",alpha_error)
        print("Groundtruth Calculated Depth: ",Z_gt_calculation)
        print("Manually Pred Calculated Depth Ideal Y Coordinate: ",Z_pred_calculation_ideal_Y)
        print("Manually Pred Calculated Depth Ideal Object Height: ",Z_pred_calculation_ideal_height)
        print("Manually Pred Calculated Depth: ",Z_pred_calculation)
        print("SMOKE Coder Pred Calculated Depth: ",pred_obj.tz)

        print("Groundtruth Y Coordinate vs Predicted Y Coordinate (SMOKE Coder Post Processing): ","{} vs {} ; Error = {}".format(gt_objects_center[matched_gt_index][1],pred_obj_center[i][1],gt_objects_center[matched_gt_index][1]-pred_obj_center[i][1]))
        print("Groundtruth Height vs Predicted Y Coordinate (SMOKE Coder Post Processing): ","{} vs {} ; Error = {}".format(matched_gt.h,pred_obj.h,matched_gt.h-pred_obj.h))


        
        print("-------------------------------------------------------")
        

    # cv2.imshow("Gt Image",gt_img)
    # cv2.waitKey(0)



    # #frame_detection_info_list=
    # frame=cv2.imread(img_path)
    # viz=SMOKE_Viz(frame,lat_range_m=40,long_range_m=70,scale=10,dataset="Prescan")

    # # Plot Groundtruth
    # viz.draw_3Dbox(K,frame,gt_list=groundtruth)
    # viz.draw_birdeyes(obj_list=groundtruth)

    # # Plot Predictions
    # viz.draw_3Dbox(K,frame,predictions_list=detection_info_list)
    # viz.draw_birdeyes(obj_list=detection_info_list)
    
    # # Plot Optimized Predictions
    # viz.draw_3Dbox(K,frame,predictions_list=detection_info_list_optimized,optimized=True)
    # viz.draw_birdeyes(obj_list=detection_info_list_optimized)

    # viz.show()

import scipy.stats
x=[alpha_error[0] for alpha_error in alpha_error_list]
y=[depth_error[0] for depth_error in depth_error_list]

print(alpha_error_list)
print(depth_error_list)

p=scipy.stats.pearsonr(x, y)    # Pearson's r
s=scipy.stats.spearmanr(x, y)   # Spearman's rho
k=scipy.stats.kendalltau(x, y)  # Kendall's tau

print("Pearson's Correlation Coefficient: ",p)
print("Spearman's Correlation Coefficient: ",s)
print("Kendalls's Correlation Coefficient: ",k)


plt.scatter(x,y)
plt.show()
