from cgi import test
from email.mime import image
import cv2
import numpy as np
import os
import glob


frame_id=0
test_dir='/home/hasan/perception-validation-verification/results/Stream592022-10-06 12:05:01.623736/smoke-image-stream59/'
IMAGE='frame'+str(frame_id)+'.png'
image_path=test_dir+IMAGE
img=cv2.imread(image_path)
print('shape',img.shape)
# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
codec = cv2.VideoWriter_fourcc(*'XVID')

video = cv2.VideoWriter('SMOKE_DEMO.mp4', codec, 20, (img.shape[1],img.shape[0]))

#test_dir='/home/hasan/SMOKE/datasets/KITTI_MOD_fixed/training/images/'

for filepath in glob.glob(os.path.join(test_dir,'*.png')):
    image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')
    print('IMAGE PATH: ',image_path)
    img = cv2.imread(image_path)
    # cv2.imshow('output',img)
    # cv2.waitKey(0)
    video.write(img)
    frame_id+=1

cv2.destroyAllWindows()
video.release()