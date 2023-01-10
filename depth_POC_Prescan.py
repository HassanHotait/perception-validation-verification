from metrics_functions_from_evaluation_script import new_read_groundtruth,new_plot_prediction,get_dataset_depth_stats
import numpy as np
import os
import csv
#from test_manuallly_obtained_matrix import get_calculated_2d_points
import copy
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
import copy

from SMOKE.smoke.config import cfg

cfg_stats=cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE


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

def get_calculated_2d_points(K,pose,prediction=True,dtype=int):

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


def compute_3Dbox(P2, obj):
        #obj=gt_obj
        # if gt_list==None:
        #obj = GtInfo(line)
        # else:
        #     obj=detectionInfo(gt_list)
        # Draw 2D Bounding Box
        # xmin = int(obj.xmin)
        # xmax = int(obj.xmax)
        # ymin = int(obj.ymin)
        # ymax = int(obj.ymax)
        # width = xmax - xmin
        # height = ymax - ymin
        # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
        # ax.add_patch(box_2d)

        # Draw 3D Bounding Box

        R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                    [0, 1, 0],
                    [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

        x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
        y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
        z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

        x_corners = [i - obj.l / 2 for i in x_corners]
        y_corners = [i - obj.h for i in y_corners]
        z_corners = [i - obj.w / 2 for i in z_corners]

        corners_3D = np.array([x_corners, y_corners, z_corners])
        corners_3D = R.dot(corners_3D)
        corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
        # print("Corners 3D_1 : ",corners_3D_1)
        # print("Corners 3D_1 Shape: ",corners_3D_1.shape)
        # print("P2: ",P2)
        # print("P2 Shape: ",P2.shape)
        corners_2D = P2.dot(corners_3D)
        # print("Dot Product Output: ",corners_2D)
        # print("Dot Product Output Shape: ",corners_2D.shape)
        corners_2D = corners_2D / corners_2D[2]
        corners_2D = corners_2D[:2]
        #print("corners 2D Final Version: ",corners_2D)

        return corners_2D
def get_optimized_depth(error,depth_pred):

    if error>0:
        gain=1.1
    else:
        gain=1.1
    print("Gain: ",gain)

    depth_optimized=depth_pred*gain

    return depth_optimized

def get_optimized_depth_flat_road(K,pred_obj,camera_height):
    fy=K[1][1]
    cy=K[1][2]

    height_relative_to_camera=camera_height-(pred_obj.h/2)
    pixel_y_coordinate=get_calculated_2d_points(K=K,pose=[pred_obj.tx,pred_obj.ty-(pred_obj.h/2),pred_obj.tz],dtype=float)[1][0]

    print("Float Y Coordinate in fnctn: ",pixel_y_coordinate)

    Z=(height_relative_to_camera*fy)/(pixel_y_coordinate-cy)

    optimized_pred_obj=pred_obj
    optimized_pred_obj.tz=Z
    corners_2D=compute_3Dbox(K,pred_obj)
    xmin=min(corners_2D[0])
    ymin=min(corners_2D[1])
    xmax=max(corners_2D[0])
    ymax=max(corners_2D[1])


    optimized_pred_obj.xmin=xmin
    optimized_pred_obj.ymin=ymin
    optimized_pred_obj.xmax=xmax
    optimized_pred_obj.ymax=ymax

    return optimized_pred_obj

def get_optimized_depth_by_stats(cfg_stats,pred_obj,path,fileid="All"):

    depth_ref=cfg_stats
    print("Default Depth Refs: ",depth_ref)
    mean,std,_=get_dataset_depth_stats(labels_path=path,depth_condition=60,extension='.txt',frame_id=fileid)
    new_depth_refs=(mean,std)
    print("Scenario Depths Ref: ",new_depth_refs)


    depths_offset=(pred_obj.tz-depth_ref[0])/depth_ref[1]
    #depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

    depth_optimized=(depths_offset*new_depth_refs[1])+new_depth_refs[0]

    optimized_pred_obj=pred_obj
    optimized_pred_obj.tz=depth_optimized
    corners_2D=compute_3Dbox(K,optimized_pred_obj)
    xmin=min(corners_2D[0])
    ymin=min(corners_2D[1])
    xmax=max(corners_2D[0])
    ymax=max(corners_2D[1])


    optimized_pred_obj.xmin=xmin
    optimized_pred_obj.ymin=ymin
    optimized_pred_obj.xmax=xmax
    optimized_pred_obj.ymax=ymax

    return optimized_pred_obj




alpha_error_list=[]
depth_error_list=[]
depth_optimized_error_list=[]

for j in range(1):
    id=j
    labels_path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\\Dataset2Kitti\\PrescanRawData_Scenario16\\DataLogger1\\label_2"
    groundtruth=new_read_groundtruth(gt_folder=labels_path,fileid=id,extension=".txt",dataset="Prescan")

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
    detection_info_list_optimized_flat_ground=[detectionInfo(copy.copy(pred)) for pred in smoke_predictions_list]
    detection_info_list_optimized_frame_depth_stats=[detectionInfo(copy.copy(pred)) for pred in smoke_predictions_list]




    gt_objects_center=[]
    for obj in groundtruth:

        obj_center=get_calculated_2d_points(K,[obj.tx,obj.ty-(obj.h/2),obj.tz],prediction=False,dtype=float)
        gt_objects_center.append(obj_center)


    print("-------------------------------------- Predictions ----------------------------------------------------")
    pred_obj_center=[]
    for pred_obj in detection_info_list:
        obj_center=get_calculated_2d_points(K,[pred_obj.tx,camera_height-(pred_obj.h/2),pred_obj.tz],prediction=True,dtype=int)
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

        optimized_pred_flat_road= get_optimized_depth_flat_road(K,detection_info_list_optimized_flat_ground[i],camera_height=camera_height)
        detection_info_list_optimized_flat_ground[i]=optimized_pred_flat_road
        output_2d_img=new_plot_prediction(gt_img,[optimized_pred_flat_road],color=(0,0,255))

        optimized_pred_by_frame_depth_stats=get_optimized_depth_by_stats(cfg_stats=cfg_stats,pred_obj=pred_obj,path=labels_path,fileid=j)
        detection_info_list_optimized_frame_depth_stats[i]=optimized_pred_by_frame_depth_stats

        # optimized_pred.tz=get_optimized_depth_flat_road(K=K,pred_obj=pred_obj,camera_height=camera_height)[0]
        # corners_2D=get_optimized_depth_flat_road(K=K,pred_obj=pred_obj,camera_height=camera_height)[1]
        # xmin=min(corners_2D[0])
        # ymin=min(corners_2D[1])
        # xmax=max(corners_2D[0])
        # ymax=max(corners_2D[1])

        # print("(xmin,ymin): ({},{}) - (xmax,ymax) : ({},{})".format(xmin,ymin,xmax,ymax))
        # print("(xmin,ymin): ({},{}) - (xmax,ymax) : ({},{})".format(pred_obj.xmin,pred_obj.ymin,pred_obj.xmax,pred_obj.ymax))

        #rint("Corners 2D: ",corners_2D)

        print("Predicted Angle Error Input For Controller: ",alpha_error)
        print("Groundtruth Calculated Depth: ",Z_gt_calculation)
        print("Manually Pred Calculated Depth Ideal Y Coordinate: ",Z_pred_calculation_ideal_Y)
        print("Manually Pred Calculated Depth Ideal Object Height: ",Z_pred_calculation_ideal_height)
        print("Manually Pred Calculated Depth: ",Z_pred_calculation)
        print("Manually Pred Calculation Depth Function Flat Road: ",optimized_pred_flat_road.tz)
        print("Manually Pred Calculation Depth Function By Stats: ",optimized_pred_by_frame_depth_stats.tz)
        print("SMOKE Coder Pred Calculated Depth: ",pred_obj.tz)

        print("Groundtruth Y Coordinate vs Predicted Y Coordinate (SMOKE Coder Post Processing): ","{} vs {} ; Error = {}".format(gt_objects_center[matched_gt_index][1],pred_obj_center[i][1],gt_objects_center[matched_gt_index][1]-pred_obj_center[i][1]))
        print("Groundtruth Height vs Predicted Y Coordinate (SMOKE Coder Post Processing): ","{} vs {} ; Error = {}".format(matched_gt.h,pred_obj.h,matched_gt.h-pred_obj.h))



        print("-------------------------------------------------------")


    cv2.imshow("2D Overview",output_2d_img)
    cv2.waitKey(0)



    #frame_detection_info_list=
    frame=cv2.imread(img_path)
    viz=SMOKE_Viz(frame,lat_range_m=40,long_range_m=70,scale=10,dataset="Prescan")

    # Plot Groundtruth
    viz.draw_3Dbox(K,frame,gt_list=groundtruth)
    viz.draw_birdeyes(obj_list=groundtruth)

    # Plot Predictions
    viz.draw_3Dbox(K,frame,predictions_list=detection_info_list)
    viz.draw_birdeyes(obj_list=detection_info_list)

    #Plot Optimized Predictions
    viz.draw_3Dbox(K,frame,predictions_list=detection_info_list_optimized_frame_depth_stats,optimized=True)
    viz.draw_birdeyes(obj_list=detection_info_list_optimized_frame_depth_stats)

    viz.show()

# import scipy.stats
# x=[alpha_error[0] for alpha_error in alpha_error_list]
# y=[depth_error[0] for depth_error in depth_error_list]

# print(alpha_error_list)
# print(depth_error_list)

# p=scipy.stats.pearsonr(x, y)    # Pearson's r
# s=scipy.stats.spearmanr(x, y)   # Spearman's rho
# k=scipy.stats.kendalltau(x, y)  # Kendall's tau

# print("Pearson's Correlation Coefficient: ",p)
# print("Spearman's Correlation Coefficient: ",s)
# print("Kendalls's Correlation Coefficient: ",k)


# plt.scatter(x,y)
# plt.show()
