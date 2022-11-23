import torch
import cv2
import time
import glob

from smoke.config import cfg
from smoke.engine import (
    default_argument_parser,
    default_setup,
    
)


import os
from smoke.utils.check_point import DetectronCheckpointer
import matplotlib.pyplot as plt
from tools.deep_sort import nn_matching
from tools.deep_sort.tracker import Tracker





# Import necessary libraries
# import torch
from PIL import Image
# import torchvision.transforms as transforms


from tools.my_functions import  preprocess,setup_network
from tools.box import visualize
from tools.deep_sort_processing import get_deepsort_input

torch.cuda.empty_cache()

f=open('record_detections_SMOKE.txt', 'w') 
# Select video or frame
video_path='test_videos/test_video.mp4'
cap = cv2.VideoCapture(video_path)
n_frames=1000

print(type(cap))
fileid=0  # Select Start Frame
test_video=True # True for testing video | False for testing frames from directory


# Setup Network

args = default_argument_parser().parse_args()
model,network_configuration,gpu_device,cpu_device=setup_network(args)

# Load Weights
checkpointer = DetectronCheckpointer(
    cfg, model, save_dir=cfg.OUTPUT_DIR
)
ckpt=cfg.MODEL.WEIGHT
_ = checkpointer.load(ckpt, use_latest=args.ckpt is None)


# Equivalent to model.train(False) for deployment/evaluation
# model.eval()

# Deep Sort Tracker Parameters
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

if test_video==True:
    detections_frames=[]
    birdview_frames=[]
    while cap.isOpened():
        results_dict = {}
        t1 = time.time()
        ret, frame = cap.read()
        if ret==True:
            pilimage=Image.fromarray(frame)
            img_tensor,target,P2=preprocess(network_configuration,fileid,pilimage,test_video)
            tuple_target=(target,)
            with torch.no_grad():
                model.eval()
                output=model.forward(img_tensor[0].to(gpu_device),targets=tuple_target)
                output = output.to(cpu_device)


            results_dict.update({fileid: output })
            #print(str(fileid)+output)
            predictions_list=output.tolist()
            detections_n=len(predictions_list)
            print('smoke output item 0',predictions_list[0][0])
            tracker=get_deepsort_input(predictions_list,frame)
            # for track in tracker.tracks:
            #     print('ID: ',track.track_id)
            # print('deep sort detections: ',deep_sort_detections)    
            # for d in deep_sort_detections:
            #     print('bbox: ',d.tlwh)
            #     print('score: ',d.confidence)
            #     print('class_name: ',d.class_name)
            #     print('feature: ',d.feature)

            birdeye_view_shape=(100,100)
            fig,boundingbox_image,birdview_image=visualize(birdeye_view_shape,predictions_list,P2,pilimage,tracker)
            detections_frames.append(boundingbox_image)
            birdview_frames.append(birdview_image)

            #print("Results Dict: ",results_dict)
            # cv2.imshow('test',birdview_image)
            # cv2.waitKey(0)

        else:
            break

        #ani = animation.FuncAnimation(fig, [detections_frames,birdview_frames], 50, blit=True, interval=10,repeat=False)

        plt.show() 
        

        fileid=fileid+1
        delta_T=time.time()-t1
        fps = 1./(time.time()-t1)
        print("FPS: ",fps)
        print("Results Dict: ",results_dict)
    



else:
    # Dataset Directory to Test
    image_dir="datasets/kitti/testing/image_2/"
    test_dir='/home/hasan/perception-validation-verification/SMOKE/datasets/KITTI_MOD_fixed/training/images/'
    image_dir=test_dir

    #images=[file for file in glob.glob()]
    for filepath in glob.glob(os.path.join(test_dir,'2011_09_26_drive_0001_sync_*.png')):
        print('file-----------------',filepath)
        
        IMAGE=str(fileid).zfill(10)+'.png'
        t1 = time.time()

        results_dict = {}
        
        # Read a PIL image
        filename=test_dir+'2011_09_26_drive_0001_sync_'+IMAGE
        # image_filename=str(fileid).zfill(6)+".png"
        # path_to_image=image_dir+image_filename
        image = Image.open(filename).convert('RGB')
        filename=filename.split('/')[-1]


        img_tensor,target,P2=preprocess(network_configuration,fileid,image,test_video)
        tuple_target=(target,)

        with torch.no_grad():
            model.eval()
            output=model.forward(img_tensor[0].to(gpu_device),targets=tuple_target)
            output = output.to(cpu_device)

        results_dict.update({fileid: output})
        
        predictions_list=output.tolist()
        detections_n=len(predictions_list)
        #print('smoke output item 0',predictions_list[0][0])
        print('detections_n: ',detections_n)
        n_cars=0
        n_bikes=0
        n_pedestrians=0
        for i in range(detections_n):
            if predictions_list[i][0]==0.0:
                n_cars+=1
            elif predictions_list[i][0]==1.0:
                n_bikes+=1
            elif predictions_list[i][0]==2.0:
                n_pedestrians+=1

        f.write(filename+' '+str(detections_n)+' '+str(n_cars)+' '+str(n_pedestrians)+' '+str(n_bikes))
        f.write('\n')



        birdeye_view_shape=(100,100)
        # visualize(birdeye_view_shape,predictions_list,P2,image,tracker)
        # plt.show() 
        
        delta_T=time.time()-t1
        fps = 1./(time.time()-t1)

        fileid=fileid+1


        print("FPS: ",fps)
        print("Results Dict: ",results_dict)



        



        










