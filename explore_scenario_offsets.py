from metrics_functions_from_evaluation_script import get_dataset_depth_stats


path="C:\\Users\\hashot51\\Desktop\\perception-validation-verification\Dataset2Kitti\\PrescanRawData_Scenario15\\DataLogger1\\labels_2"

mean,std,car_dim_reference=get_dataset_depth_stats(labels_path=path,n_frames=1,depth_condition=60)

print("mean: ",mean)
print("std: ",std)

print("Average Car dimensions in dataset: ",car_dim_reference)