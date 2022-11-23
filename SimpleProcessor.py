from processorclass import MobileEyeDataYoloProcessor
from datetime import datetime
import os

automatic_id_selection=False

# Define Paths for Storing Results
stream_id='Calib_Batch2'
session_datetime=str(datetime.now())  

foldername='Stream'+str(stream_id)+session_datetime
root_dir='/home/hasan//perception-validation-verification'

watch_complete_video=True
automatic_id_selection=False

c=MobileEyeDataYoloProcessor(foldername,root_dir,stream_id)
c.use_gpu_switch(True)
c.setup_yolo()
c.create_results_folders()


c.nth_frame=100
c.run_yolo_log_results('2.mp4',watch_complete_video)
c.play_video_with_tracked_objects()

c.relevant_ids=[1,4,15,18,31,39,43,53,54]

c.request_relevant_ids(automatic_id_selection)

# Vid 1
#relevant_ids=[5,22,27,33,47,78,97,105,144,159,163]
# Vid 2q
# relevant_ids=[1,4,7,15,18,31,39,43,53,54]
    

c.filter_objects_by_id_then_log()
c.log_yolo_filtered_objects_analytics()
c.log_Mobileye_CameraTrackedData_analytics()
c.match_yolo_detecions_with_Mobileye_CameraTrackedData_by_timestamps()
#c.log_results_in_video()


c.convert_data_to_column_format_then_filter_reliable_frames()
c.write_video_with_distances()
c.play_results_video()
c.request_frames_to_include()
c.get_K_linear_regression()


