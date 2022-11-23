import numpy as np
import cv2
import csv
import pandas as pd

data = pd.read_csv('tools/logs/inference/kitti_test/data/000009.txt', sep=" ", header=None)
print(type(data.iloc[0][4]))
# print(type(data[4]))
# print(format(data[4]))
 
# Load an color image in grayscale
img = cv2.imread('000009.png')

print("Image Shape",img.shape)
 

# Pt 1 Vertical
start_point=(int(data.iloc[0][4]),0)
end_point=(int(data.iloc[0][4]),img.shape[0])
color=(255,0,0)
thickness=2
image=cv2.line(img, start_point, end_point, color, thickness) 

# Pt 2 int
start_point=(0,int(data.iloc[0][5]))
end_point=(img.shape[1],int(data.iloc[0][5]))
color=(255,0,0)
thickness=2
image=cv2.line(image, start_point, end_point, color, thickness) 

# Pt 3 Vertical
start_point=(int(data.iloc[0][6]),0)
end_point=(int(data.iloc[0][6]),img.shape[0])
color=(255,0,0)
thickness=2
image=cv2.line(img, start_point, end_point, color, thickness) 

# Pt 4 Horizontal
start_point=(0,int(data.iloc[0][7]))
end_point=(img.shape[1],int(data.iloc[0][7]))
color=(255,0,0)
thickness=2
image=cv2.line(image, start_point, end_point, color, thickness) 





# resized = cv2.resize(img, (1280,384), interpolation = cv2.INTER_CUBIC)




# show image
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()