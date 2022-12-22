from re import X
from turtle import width
import torch
import cv2
import time

from smoke.config import cfg
from smoke.engine import (
    default_argument_parser,
    default_setup,
    
)
from  smoke.modeling.detector import build_detection_model

# from smoke.data import build_test_loader
# from smoke.engine.inference import compute_on_dataset,inference
# from smoke.engine.test_net import run_test
# from smoke.data import transforms as T
# from smoke.data.transforms.transforms import ToTensor, Compose,Normalize

from torchvision.transforms import functional as F
from smoke.data.transforms import transforms as T
from smoke.data.transforms import build_transforms
# from smoke.data.build import build_dataset
# from smoke.utils.imports import import_file
#from smoke.utils.visualization import draw_birdeyes
from smoke.structures.params_3d import ParamsList
# from smoke.structures.image_list import to_image_list

import matplotlib.patches as patches
from matplotlib.path import Path

import os
import numpy as np
import csv
from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix)

#from smoke.utils.check_point import DetectronCheckpointer
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import matplotlib



# Import necessary libraries
# import torch
from PIL import Image
# import torchvision.transforms as transforms

from smoke.utils import comm

def setup_network(args):

    cfg = setup(args)
    model = build_detection_model(cfg)

    device = torch.device(cfg.MODEL.DEVICE) 
    model.to(device)
    cpu_device = torch.device("cpu")



    return model,cfg,device,cpu_device


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

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

def preprocess(network_configuration,file_id,image):

    # Get Intrinsic Matrix

    K,P2=get_K(str(file_id).zfill(6)+".txt","/home/hashot51/Projects/perception-validation-verification/SMOKE/datasets/kitti/training/calib")
    # print('Type of K: ',type(K))
    # K=np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=np.float32)

    print('input image size: ',image.size)

    
    target = ParamsList(image_size=image.size  ,is_train=True) # Default (1242.375)
    target.add_field("K", K)



    print('Input Image Original Shape Before Resizing',image.size)

    center = np.array([i / 2 for i in image.size], dtype=np.float32)
    size = np.array([i for i in image.size], dtype=np.float32)

    
    center_size = [center, size]

    img_tensor=transformation(network_configuration,image,center_size,target)

    return img_tensor,target,P2

def preprocess_then_predict(model,network_configuration,file_id,image,gpu_device,cpu_device):
    model.eval()
    img_tensor,target,P2=preprocess(network_configuration,file_id,image)
    
    tuple_target=(target,)
    with torch.no_grad():
        img=img_tensor[0].to(gpu_device)
        output=model(img,targets=tuple_target)
        output = output.to(cpu_device)
    
    smoke_predictions_list=output.tolist()
    print('SMOKE Predictions list in fnctn:',smoke_predictions_list)

    return smoke_predictions_list


def transformation(network_configuration,image,center_size,target):
    input_width = network_configuration.INPUT.WIDTH_TRAIN
    input_height = network_configuration.INPUT.HEIGHT_TRAIN
    output_width = input_width // network_configuration.MODEL.BACKBONE.DOWN_RATIO
    output_height = input_height // network_configuration.MODEL.BACKBONE.DOWN_RATIO


    trans_mat = get_transfrom_matrix(
    center_size,
    [output_width, output_height])
    
    target.add_field("trans_mat", trans_mat)




    trans_affine = get_transfrom_matrix(
        center_size,
        [input_width, input_height]
    )
    trans_affine_inv = np.linalg.inv(trans_affine)

    image_resized = image.transform(
    (input_width, input_height),
    method=Image.AFFINE,
    data=trans_affine_inv.flatten()[:6],
    resample=Image.BILINEAR,
)

    transform=build_transforms(network_configuration,is_train=False)
    img_tensor = transform(image_resized,target=target)

    return img_tensor

def plot_show_2d(predictions_list,image_cv2,fileid):
    detections_n=len(predictions_list)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    
    # fontScale
    fontScale = 0.8
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
   
    for i in range(detections_n):

        start_point=(int(predictions_list[i][2]),int(predictions_list[i][3]))
        end_point=(int(predictions_list[i][4]),int(predictions_list[i][5]))
        image_cv2 = cv2.rectangle(image_cv2, start_point, end_point, (0,0,255), 1)
        # print('Start Point: ',start_point)
        # print('End Point: ',end_point)
        # print('xmin: ',int(predictions_list[i][2]))
        # print('ymin: ',int(predictions_list[i][3]))
        # print('xmax: ',int(predictions_list[i][4]))
        # print('ymax: ',int(predictions_list[i][5]))

        # org
        org = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2))
        #print("Number if Items in Prediction List: ",len(predictions_list[i]))
        h=predictions_list[i][6]
        w=predictions_list[i][7]
        l=predictions_list[i][8]
        #print('(Height,Width,Length)=',(h,w,l))
        x=predictions_list[i][9]
        y=predictions_list[i][10]
        z=predictions_list[i][11]
        orientation=predictions_list[i][12]
        score=predictions_list[i][13]
        #print('Confidence: ',score)
        


        d_birdview=np.sqrt(x**2+z**2)
        d_birdview_rounded = "{:.2f}".format(d_birdview)
        x_rounded="{:.2f}".format(x)
        y_rounded="{:.2f}".format(y)
        z_rounded="{:.2f}".format(z)
        info=(d_birdview_rounded,x_rounded,y_rounded,z_rounded)
        score_rounded="{:.2f}".format(score)
        orientation_rounded="{:.2f}".format(orientation)
        orientation_degrees=np.degrees(orientation)
        orientation_degrees_rounded="{:.2f}".format(orientation_degrees)
        #print('Orientation Degrees: ',orientation_degrees)
        hypothesis=(orientation_degrees_rounded,score_rounded)

        # Using cv2.putText() method
        image_cv2 = cv2.putText(image_cv2,'d: '+str(d_birdview_rounded), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        img_shape=image_cv2.shape
        #print('IMG SHAPE ',img_shape)
        image_cv2=cv2.line(image_cv2, (int(img_shape[1]/2),0),(int(img_shape[1]/2),img_shape[0]), color, thickness) 
        



    cv2.imshow("ID: ",image_cv2)
    #cv2.waitKey(0)


def compute_birdviewbox(prediction_list, shape, scale):
    detections_n=len(prediction_list)
    for i in range(detections_n):
        h = prediction_list[i][7] * scale
        w = prediction_list[i][8] * scale
        l = prediction_list[i][9] * scale
        x = prediction_list[i][10] * scale
        y = prediction_list[i][11] * scale
        z = prediction_list[i][12] * scale
        rot_y = prediction_list[i][13]

        R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                    [np.sin(rot_y), np.cos(rot_y)]])
        t = np.array([x, z]).reshape(1, 2).T

        x_corners = -l/2#[0, l, l, 0]  # -l/2
        z_corners = -w/2#[w, w, 0, 0]  # -w/2


        x_corners += -w / 2
        z_corners += -l / 2

        # bounding box in object coordinate
        corners_2D = np.array([x_corners, z_corners])
        # rotate
        corners_2D = R.dot(corners_2D)
        # translation
        corners_2D = t - corners_2D
        # in camera coordinate
        corners_2D[0] += int(shape/2)
        corners_2D = (corners_2D).astype(np.int16)
        corners_2D = corners_2D.T

        return np.vstack((corners_2D, corners_2D[0,:]))



def draw_birdeyes(ax2, prediction_list, shape):
    # shape = 900
    scale = 1

    pred_corners_2d = compute_birdviewbox(prediction_list, shape, scale)


    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=True, color='green', label='prediction')
    ax2.add_patch(p)

    #return p
    

def matplotlibfig(prediction_list):
    shape=900
    birdimage = np.zeros((shape, shape, 3), np.uint8)
    fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

    # fig.tight_layout()
    gs = GridSpec(1, 4)
    gs.update(wspace=0)  # set the spacing between axes.

    ax = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 3:])

    

    draw_birdeyes(ax2, prediction_list, shape)
    # visualize bird eye view
    ax2.imshow(birdimage, origin='lower')
    plt.show()

def birdeye_view(prediction_list):
    ax = plt.gca()
    initial_transform=ax.transDataextent=[-shape[0]/2, shape[0]/2, -shape[1]/2, shape[1]/2]
    ax.cla() # clear things for fresh plot
    shape=(100,100)
    blank=np.zeros(shape)
    x_list=[]
    y_list=[]
    z_list=[]
    print("Number of Objects Detected: ",len(prediction_list))
    for i in range(len(prediction_list)):
        w=prediction_list[i][7]
        l=prediction_list[i][8]
        x=prediction_list[i][9]
        y=prediction_list[i][10]
        z=prediction_list[i][11]
        orientation=prediction_list[i][12]
        d_birdview=np.sqrt(x**2+z**2)
        print('Center Object: ',(x,z))
        print('Object Width: ',w)
        print('Object Length',l)
        print('Rectangle Left Bottom',(int(x-(l/2)), int(z-(w/2))))
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        # Create a Rectangle patch
        rect = patches.Rectangle(((x-(w/2)), (z-(l/2))), w, l, linewidth=1, edgecolor='r', facecolor='none')
        t2 = matplotlib.transforms.Affine2D().rotate_around(x,z,theta=(orientation-np.radians(90)))+ax.transData
        print('ax.transData: ',ax.transData)
        print('t2',t2)
        rect.set_transform(t2)
        ax.add_patch(rect)

    circle1 = plt.Circle((0, 0), 25, color='b', fill=False)
    circle2 = plt.Circle((0, 0), 12.5, color='b', fill=False)
    circle3 = plt.Circle((0, 0), 6.25, color='b', fill=False)
    circle4 = plt.Circle((0, 0), 3.125, color='b', fill=False)



    ax.imshow(blank,cmap='gray',)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    ax.add_patch(circle4)
    plt.xlim([-shape[0]/2,shape[0]/2])
    plt.ylim([0,shape[1]/2])
    # plt.xlim([-200,200])
    # plt.ylim([-200,200])
    plt.grid(visible=True)
    #plt.scatter(x_list,z_list)
    plt.show()




