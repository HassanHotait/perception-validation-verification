from metrics_functions_from_evaluation_script import get_dataset_depth_stats
import math


path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\Dataset2Kitti\\PrescanRawData_Scenario16\\DataLogger1\\label_2"

mean_list=[]
std_list=[]
# for i in range(100):
mean,std,car_dim_reference=get_dataset_depth_stats(labels_path=path,frame_id=62,extension=".txt")
# if math.isnan(mean):
#     print("image with nan depth refs: ",i)
mean_list.append(mean)
std_list.append(std)

print("mean: ",mean_list)
print("std: ",std_list)

#print("Average Car dimensions in dataset: ",car_dim_reference)