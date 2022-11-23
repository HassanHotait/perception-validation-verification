import os
import csv
import cv2



import torch
import csv
import numpy as np
torch.cuda.empty_cache()
import os
import glob
from datetime import datetime
# from toolbox import get_IoU,Point,smoke_get_n_classes,yolo_get_n_classes
from metrics_functions_from_evaluation_script import plot_groundtruth,plot_prediction,yolo_2_smoke_output_format
import time

from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd


from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
from YOLO_toolbox  import setup_YOLO,setup_tracker,preprocess_predict_YOLO,postprocess_YOLO,yolo_visualize



# 1 Run Yolo on Video
# 2 Store Vehicle Detections (Car/Bus/Truck/.....)
# 3 Save Images with Tracked Objects
# Play Video
# 4 Select Relevant IDS by Watching Video
# 5 Filter Tracked Objects by ID

def find_indices(list,item_to_find):
    indices=[]

    for idx,value in enumerate(list):
        if value==item_to_find:
            indices.append(idx)
    
    return indices


def closest(lst, K):
      
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]



class MobileEyeDataYoloProcessor():
    def __init__(self,foldername,root_dir,stream_id):
        print('HI Im CLass')

        self.vehicle_labels=['car','bus','truck']

        self.relevant_ids=[]

        self.video_path_dictionary={
                                    "1.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-02-31_5/cameraTrackedData.csv",
                                    "2.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-10-31_21/cameraTrackedData.csv",
                                    "3.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-13-01_26/cameraTrackedData.csv",
                                    "4.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-16-01_32/cameraTrackedData.csv",
                                    "5.mp4": "calibration_optimisation_data/Mobileye/2022-06-08-13-22-38_5/cameraTrackedData.csv",
                                    "6.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-19-01_38/cameraTrackedData.csv",
                                    "7.mp4": "calibration_optimisation_data/Mobileye/2022-02-25-12-19-31_39/cameraTrackedData.csv",
                                    }

        self.nth_frame=100
        self.root_dir=root_dir
        self.results_path=os.path.join(root_dir,'results',foldername)
        self.yolo_image_stream_path_tracked=os.path.join(self.results_path,'yolo-image-stream-tracked'+str(stream_id))
        self.yolo_image_stream_path_filtered=os.path.join(self.results_path,'yolo-image-stream-filtered'+str(stream_id))
        self.groundtruth_image_stream_path=os.path.join(self.results_path,'groundtruth-image-stream'+str(stream_id))
        self.logs_path=os.path.join(self.results_path,'logs')


    def create_results_folders(self):
        # Create Folder with Datetime to store results of stream
        os.mkdir(self.results_path)
        # Create Folders in which to store YOLO and SMOKE Frames
        os.mkdir(self.yolo_image_stream_path_tracked)
        os.mkdir(self.yolo_image_stream_path_filtered)
        #os.mkdir(smoke_image_stream_path)
        # Create foler in which to store text file logs
        os.mkdir(self.logs_path)
        os.mkdir(self.groundtruth_image_stream_path)

    def setup_yolo(self):
        # Setup YOLOV3 | Load Weights | Initialize Tracker for Object IDS | Initialize Feature Encoder
        self.yolo,self.class_names=setup_YOLO(path_to_class_names='YOLO_Detection/data/labels/coco.names',
                                    path_to_weights='YOLO_Detection/weights/yolov3.tf')
        model_filename = 'YOLO_Detection/model_data/mars-small128.pb'
        self.tracker,self.encoder=setup_tracker(model_filename)




    def use_gpu_switch(self,flag):

        if flag==False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
        else:
            pass

    def run_yolo_log_results(self,video_filename,complete_video_flag):


        self.video_filename=video_filename

        ##  Setup Logging File
        self.yolo_detections_filename='mobileeye_yolo_detections.csv'
        f = open(os.path.join(self.logs_path,self.yolo_detections_filename), 'w')
        # create the csv writer
        writer = csv.writer(f)
        header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
        writer.writerow(header)



        # Run Yolo On Video
        fileid=0
        # Video Reader
        vid_in = cv2.VideoCapture(os.path.join('test_videos',self.video_filename))
        # Get Frame info for duration calculation
        total_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = vid_in.get(cv2.CAP_PROP_FPS) 

        # Video Writer For Getting Relevant IDS
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps =int(vid_in.get(cv2.CAP_PROP_FPS))
        vid_width,vid_height = int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_out_filepath=os.path.join(self.logs_path,video_filename+'_yolo_tracker_ids.mp4')
        vid_out = cv2.VideoWriter(self.vid_out_filepath, codec, vid_fps, (vid_width, vid_height))


        
        ret,frame=vid_in.read()
        while ret==True:
            print('frame: ',fileid)

            # Run Yolo 
            boxes, scores, classes, nums=preprocess_predict_YOLO(self.yolo,frame)
            boxs,classes,tracker=postprocess_YOLO(self.encoder,self.tracker,self.class_names,frame,boxes,scores,classes)
            yolo_predictions_list=yolo_2_smoke_output_format(boxs=boxs,classes=classes)

            timestamp=(fileid/fps)


            # Log Tracked Vehicles
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update >1:
                    continue
                elif track.get_class() in self.vehicle_labels:
                    row=[fileid,timestamp,track.get_class(),track.track_id,track.to_tlbr()[0],track.to_tlbr()[1],track.to_tlbr()[2],track.to_tlbr()[3]]
                    writer.writerow(row)
                else:
                    pass


            output_img=plot_prediction(frame,yolo_predictions_list)
            yolo_output_img=yolo_visualize(tracker,frame,fileid,timestamp)
            vid_out.write(yolo_output_img)
            cv2.imwrite(os.path.join(self.yolo_image_stream_path_tracked,'frame'+str(fileid)+'.png'),yolo_output_img)
            #yolo_metrics_evaluator.evaluate_metrics(groundtruth,yolo_predictions_list)
            fileid+=1
            ret,frame=vid_in.read()

            if complete_video_flag==False:
                if fileid==self.nth_frame:
                    ret=False
            else:
                pass
        
        f.close()
        vid_in.release()
        vid_out.release()

    def play_video_with_tracked_objects(self):
        desired_fps=30
        
        vid_out = cv2.VideoCapture(self.vid_out_filepath)
        # Get Frame info for duration calculation
        total_frames = int(vid_out.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = vid_out.get(cv2.CAP_PROP_FPS) 


        # Check if camera opened successfully
        if (vid_out.isOpened()== False):
            print("Error opening video file")
        
        # Read until video is completed
        while(vid_out.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = vid_out.read()
            if ret == True:
            # Display the resulting frame
                cv2.imshow('Frame', frame)
                
            # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        
        # Break the loop
            else:
                vid_out = cv2.VideoCapture(self.vid_out_filepath)

            time.sleep(1/desired_fps)
        
        # When everything done, release
        # the video capture object
        vid_out.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()

    def request_relevant_ids(self,automatic_mode):

        self.automatic_mode=automatic_mode

        if automatic_mode==False:
            self.relevant_ids=input('Enter Relevant IDS as integers seperated by commas  ; For Example :1,2,3,4,5,6   \n')


    def filter_objects_by_id_then_log(self):



        header = ['frame id','timestamp','label','tracker id','xmin','ymin','xmax','ymax']
        self.filtered_yolo_detections_filename='mobileeye_yolo_detections_filtered_id.csv'

        outputfile=open(os.path.join(self.logs_path,self.filtered_yolo_detections_filename), 'w')
        writer=csv.writer(outputfile)
        writer.writerow(header)

        with open(os.path.join(self.logs_path,self.yolo_detections_filename), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for i,row in enumerate(csv_reader):
                if i!=0:
                    if self.automatic_mode==False:
                        ids=[eval(id) for id in list(self.relevant_ids.split(','))]
                    else:
                        ids=self.relevant_ids
                    if int(row[3]) in ids:
                        writer.writerow(row)


        outputfile.close()


    def log_yolo_filtered_objects_analytics(self):
        self.yolo_filtered_ids_analytics_filename='yolo_filtered_objects_analytics.csv'
        self.yolo_filtered_objects_analytics_filepath=open(os.path.join(self.logs_path,self.yolo_filtered_ids_analytics_filename),'w')
        writer=csv.writer(self.yolo_filtered_objects_analytics_filepath)
        writer.writerow(['timestamp','frame','# objects','X1_min','Y1_min','X1_max','Y1_max','......','Xn_min','Yn_min','Xn_max','Yn_max'])

        with open(os.path.join(self.logs_path,self.filtered_yolo_detections_filename),'r') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        frames_with_relevant_objects=[int(row[0]) for row in rows[1:]]

        unique_frames=list(set(frames_with_relevant_objects))



        for frame_id in unique_frames:

            print(frame_id)
            indices_of_frame=find_indices(frames_with_relevant_objects,frame_id)
            n_objects=len(indices_of_frame)


            datarow=[]

            for i in indices_of_frame:
                datarow.extend((rows[i+1][4],rows[i+1][5],rows[i+1][6],rows[i+1][7]))



            writer.writerow([rows[indices_of_frame[0]+1][1],rows[indices_of_frame[0]+1][0],n_objects]+datarow)


        self.yolo_filtered_objects_analytics_filepath.close()


    def log_Mobileye_CameraTrackedData_analytics(self):

        self.Mobileye_CameraTrackedData_analytics_filename='Mobileye_CameraTrackedData_analytics.csv'
        self.Mobileye_CameraTrackedData_analytics_filepath=open(os.path.join(self.logs_path,self.Mobileye_CameraTrackedData_analytics_filename), 'w',newline='') 
        writer = csv.writer(self.Mobileye_CameraTrackedData_analytics_filepath)


        with open(self.video_path_dictionary[self.video_filename]) as f_in:
             for line in f_in:
                elements=line.split(',')
                elements[-1] = elements[-1].strip()
                #print('elements[-1: ]',elements[-1])
                indices=find_indices(elements,'1000')
                indices_total=[[i-1,i,i+1] for i in indices]
                indices_total=sum(indices_total,[])
                elements=np.delete(elements, indices_total).tolist()

                if len(elements)==1:
                    objects=0

                elif len(elements)==4:
                    objects=1

                elif len(elements)==7:
                    objects=2

                elif len(elements)==10:
                    objects=3   
                
                elements.insert(1,objects)

                row_data=[float(item) for item in elements]

                lateral_distances=[row_data[4+(3*i)] for i in range(objects)]

                filtered_indices=[row_data.index(lat_object) for lat_object in lateral_distances if not -4<=lat_object<=4 ]

                # indices_total=[[i-2,i,i-1] for i in filtered_indices]
                # indices_total=sum(indices_total,[])
                # filtered_elements=np.delete(row_data, indices_total).tolist()

                indices_to_remove=[[i-2,i-1,i] for i in filtered_indices]

                flatten_list=[item for sublist in indices_to_remove for item in sublist] 

                filtered_row=[item for j,item in enumerate(row_data) if j not in flatten_list]

                print('filtered row: ',filtered_row)
                if len(filtered_row)==2:
                    objects=0

                elif len(filtered_row)==5:
                    objects=1

                elif len(filtered_row)==8:
                    objects=2

                elif len(filtered_row)==11:
                    objects=3   

                print('n objects: ',objects)

                filtered_row[1]=objects
                writer.writerow(filtered_row)
                
                    
        self.Mobileye_CameraTrackedData_analytics_filepath.close()

    def match_yolo_detecions_with_Mobileye_CameraTrackedData_by_timestamps(self):
        camera_height=1.2
        lateral_offset=0
        longitudinal_offset=0.9
        height_offset=-0.2
        # find closest timestamps

        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filename='matched_yolo_detecions_with_Mobileye_CameraTrackedData.csv'
        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filepath=open(os.path.join(self.logs_path, self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filename), 'w',newline='') 
        writer = csv.writer(self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filepath)
        header = ['image_filename', '# objects', 'X1 3D', 'Y1 3D','Z1 3D','........','Xn 3D', 'Yn 3D','Zn 3D','X1 Obj Center 2D','Y1 Obj Center 2D','........','Xn Obj Center 2D','Yn Obj Center 2D']
        writer.writerow(header)


        with open(os.path.join(self.logs_path,self.yolo_filtered_ids_analytics_filename),'r') as f:
            csv_reader = csv.reader(f)
            yolo_rows = list(csv_reader)

        with open(os.path.join(self.logs_path,self.Mobileye_CameraTrackedData_analytics_filename),'r') as f:
            csv_reader = csv.reader(f)
            mobileye_rows = list(csv_reader)

        yolo_timestamps=[float(row[0]) for row in yolo_rows[1:]]
        mobileeye_timestamps=[float(row[0]) for row in mobileye_rows]



        for index,timestamp in enumerate(yolo_timestamps):
            t_closest=closest(mobileeye_timestamps,timestamp)

            mobileye_list_index=mobileeye_timestamps.index(t_closest)

            if yolo_rows[index+1][2]==mobileye_rows[mobileye_list_index][1]:
                #writer.writerow(yolo_rows[index+1])
                #writer.writerow(mobileye_rows[mobileye_list_index])
                ##################################
                yolo_row=yolo_rows[index+1]
                mobileye_row=mobileye_rows[mobileye_list_index]

                n_objects=int(yolo_row[2])
                image_filename=yolo_row[1]

                pose_3d=[]
                image_2d=[]

                if yolo_row[2]==1:
                    pose_3d=[-float(mobileye_row[4]),camera_height,float(mobileye_row[3])]
                    xc=(yolo_row[3]+yolo_row[5])/2
                    yc=(yolo_row[4]+yolo_row[6])/2
                    bbox_height=abs(yolo_row[4]-yolo_row[6])
                    image_2d=[xc,yc+(bbox_height/2)]

                    row=[image_filename,n_objects]+pose_3d+image_2d
                    writer.writerow(row)

                else:
                    for i in range(n_objects):
                        pose_3d.extend((-float(mobileye_row[4+(3*i)]),camera_height+height_offset,float(mobileye_row[3+(3*i)])+longitudinal_offset))

                        xc=(float(yolo_row[3+(4*i)])+float(yolo_row[5+(4*i)]))/2
                        yc=(float(yolo_row[4+(4*i)])+float(yolo_row[6+(4*i)]))/2

                        bbox_height=abs(float(yolo_row[4+(4*i)])-float(yolo_row[6+(4*i)]))


                        image_2d.extend((xc,yc+(bbox_height/2)))

                    # sort detections in correct order

                    # pose_3d=[x1,y1,z1,x2,y2,z2]
                    # image_2d=[x1,y1,x2,y2]
                    lateral_distances=[pose_3d[i] for i in range(0,len(pose_3d),3)]
                    lateral_pixel_points=[image_2d[i] for i in range(0,len(image_2d),2)]

                    print('Lateral Distances: ',lateral_distances)
                    print('Lateral Pixel Points: ',lateral_pixel_points)

                    sorted_lateral_distances=lateral_distances.copy()
                    sorted_lateral_pixel_points=lateral_pixel_points.copy()

                    sorted_lateral_distances.sort()
                    print('Sorted Lateral Distances: ',lateral_distances)
                    sorted_lateral_pixel_points.sort()
                    print('Sorted Lateral Pixel Points: ',sorted_lateral_pixel_points)

                    sorted_pose_3d=[]
                    sorted_image_2d=[]

                    for lat_d,lateral_pixel in zip(sorted_lateral_distances,sorted_lateral_pixel_points):
                        distance_index=pose_3d.index(lat_d)
                        pixel_index=image_2d.index(lateral_pixel)

                        sorted_pose_3d.extend((pose_3d[distance_index],pose_3d[distance_index+1],pose_3d[distance_index+2]))
                        sorted_image_2d.extend((image_2d[pixel_index],image_2d[pixel_index+1]))

                    row=[image_filename,n_objects]+sorted_pose_3d+sorted_image_2d

                    writer.writerow(row)


        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filepath.close()


    def log_results_in_video(self):

        # Video Writer

        frame_id=1
        test_dir=self.yolo_image_stream_path_tracked
        IMAGE='frame'+str(frame_id)+'.png'
        image_path=os.path.join(test_dir,IMAGE)
        img=cv2.imread(image_path)
        print('shape',img.shape)
        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        codec = cv2.VideoWriter_fourcc(*'XVID')

        self.video_results_filename=os.path.join(self.logs_path,'objects_with_distances.mp4')
        video = cv2.VideoWriter(self.video_results_filename, codec, 20, (img.shape[1],img.shape[0]))


        
        with open(os.path.join(self.logs_path,self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filename),'r') as f:
            csv_reader = csv.reader(f)
            rows_final_dataset = list(csv_reader)

        with open(os.path.join(self.logs_path,self.yolo_filtered_ids_analytics_filename),'r') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)


        frames_with_relevant_objects=[int(row[1]) for row in rows[1:]]


        frames_with_relevant_objects_final_dataset=[int(row[0]) for row in rows_final_dataset[1:]]


        print(frames_with_relevant_objects)
        print('---------------------------------------------',len(rows))
        for filepath in glob.glob(os.path.join(test_dir,'*.png')):
            image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')
            print('IMAGE PATH: ',image_path)
            img = cv2.imread(image_path)



            if frame_id in frames_with_relevant_objects:
                #img = cv2.imread(image_path)
                #img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
                index=frames_with_relevant_objects.index(frame_id)
                print('condition true')
                n_objects=int(rows[index+1][2])
                for i in range(n_objects):
                    img = cv2.rectangle(img, (int(float(rows[index+1][3+(4*i)])),int(float(rows[index+1][4+(4*i)]))),(int(float(rows[index+1][5+(4*i)])),int(float(rows[index+1][6+(4*i)]))) , (255,255,0), 10)


            if frame_id in frames_with_relevant_objects_final_dataset:
                #img = cv2.imread(image_path)


                # img=cv2.circle(img, (int(float(rows[frame_id+1][5])),int(float(rows[frame_id+1][6]))), 10, (255,0,0), 3)

                # img=cv2.putText(img,str((rows[frame_id+1][2],rows[frame_id+1][3],rows[frame_id+1][4])),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)


                #indices=find_indices(frames_with_relevant_objects,frame_id)
                index=frames_with_relevant_objects_final_dataset.index(frame_id)
                index_rows=frames_with_relevant_objects.index(frame_id)
                # print('condition true')

                #for i in indices:

                n_objects=int(rows_final_dataset[index+1][1])
                print(n_objects)

                if n_objects==1:
                    x_2d=int(float(rows_final_dataset[index+1][5]))
                    y_2d=int(float(rows_final_dataset[index+1][6]))

                    x_3D=rows_final_dataset[index+1][2]
                    y_3d=rows_final_dataset[index+1][3]
                    z_3d=rows_final_dataset[index+1][4]

                    img=cv2.circle(img,(x_2d,y_2d), 10, (255,0,0), 3)

                    xc=int((float(rows[index_rows+1][3])+float(rows[index_rows+1][5]))/2)
                    yc=int((float(rows[index_rows+1][4])+float(rows[index_rows+1][6]))/2)


                    img=cv2.putText(img,str((x_3D,y_3d,z_3d)),(xc-10,yc),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),2)

                else:
                    print('n objects: ',n_objects)

                    for i in range(n_objects):

                        x_2d=int(float(rows_final_dataset[index+1][2+(3*n_objects)+(2*i)]))
                        y_2d=int(float(rows_final_dataset[index+1][3+(3*n_objects)+(2*i)]))

                        #row_data=[row[0],row[1],row[2+(3*i)],row[3+(3*i)],row[4+(3*i)],row[2+(3*n_objects)+(2*i)],row[3+(3*n_objects)+(2*i)]]

                        x_3D=rows_final_dataset[index+1][2+(3*i)]
                        y_3d=rows_final_dataset[index+1][3+(3*i)]
                        z_3d=rows_final_dataset[index+1][4+(3*i)]

                        img=cv2.circle(img,(x_2d,y_2d), 10, (255,0,0), 3)

                        xc=int((float(rows[index_rows+1][3+(4*i)])+float(rows[index_rows+1][5+(4*i)]))/2)
                        yc=int((float(rows[index_rows+1][4+(4*i)])+float(rows[index_rows+1][6+(4*i)]))/2)



                        img=cv2.putText(img,str((x_3D,y_3d,z_3d)),(xc-10,yc),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),2)
                
            else:
                img=cv2.imread(image_path)
            # cv2.imshow('output',img)
            # cv2.waitKey(0)
            video.write(img)
            frame_id+=1

        cv2.destroyAllWindows()
        video.release()


    


    def request_frames_to_include(self):
        included_frames=input('Enter range of frames to include:  ; For Example:10,300,450,700 is [10,11,........,300],[450,....,700]')

        frames_list=[eval(n) for n in list(included_frames.split(','))] 


        self.list_of_included_frames=[]
        for i in range(0,len(frames_list),2):
            self.list_of_included_frames=list(range(frames_list[i],frames_list[i+1]+1))+self.list_of_included_frames

        self.list_of_included_frames.sort()

    def convert_data_to_column_format_then_filter_reliable_frames(self):

        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filename='matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat.csv'
        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filepath=open(os.path.join(self.logs_path, self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filename), 'w',newline='') 
        writer = csv.writer(self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filepath)
        header = ['image_filename', '# objects', 'X1 3D', 'Y1 3D','Z1 3D','X1 Obj Center 2D','Y1 Obj Center 2D']
        writer.writerow(header)

        with open(os.path.join(self.logs_path, self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_filename)) as f:
            reader=csv.reader(f)

            rows=list(reader)


        for row in rows[1:]:
            n_objects=int(float(row[1]))
            frame_id=int(float(row[0]))



            if n_objects>1:
                
                for i in range(n_objects):

                    row_data=[row[0],row[1],row[2+(3*i)],row[3+(3*i)],row[4+(3*i)],row[2+(3*n_objects)+(2*i)],row[3+(3*n_objects)+(2*i)]]
                    writer.writerow(row_data)

            else:
                writer.writerow(row)






        self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filepath.close()

    def write_video_with_distances(self):

        frame_id=0
        test_dir=self.yolo_image_stream_path_tracked
        IMAGE='frame'+str(frame_id)+'.png'
        image_path=os.path.join(test_dir,IMAGE)
        img=cv2.imread(image_path)


        # choose codec according to format needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        codec = cv2.VideoWriter_fourcc(*'XVID')

        self.video_results_filename=os.path.join(self.logs_path,'objects_with_distances.mp4')
        video = cv2.VideoWriter(self.video_results_filename, codec, 20, (img.shape[1],img.shape[0]))

        with open(os.path.join(self.logs_path,self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filename),'r') as f:
            csv_reader=csv.reader(f)
            rows_column_format=list(csv_reader)

        with open(os.path.join(self.logs_path,self.yolo_filtered_ids_analytics_filename),'r') as f:
            csv_reader = csv.reader(f)
            rows_yolo_boxes = list(csv_reader)

        frames_column_format=[int(float(row[0])) for row in rows_column_format[1:]]
        frames_yolo_boxes=[int(float(row[1])) for row in rows_yolo_boxes[1:]]


        for filepath in glob.glob(os.path.join(test_dir,'*.png')):
            image_path=os.path.join(test_dir,'frame'+str(frame_id)+'.png')
            img = cv2.imread(image_path)

            # Plot Bounding Boxes from YOLO Detections File
            if frame_id in frames_yolo_boxes:
                #img = cv2.imread(image_path)
                #img=cv2.putText(img,'Frame: '+str(frame_id),(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,00,00),4)
                index=frames_yolo_boxes.index(frame_id)
                print('condition true')
                n_objects=int(rows_yolo_boxes[index+1][2])
                for i in range(n_objects):
                    img = cv2.rectangle(img, (int(float(rows_yolo_boxes[index+1][3+(4*i)])),int(float(rows_yolo_boxes[index+1][4+(4*i)]))),(int(float(rows_yolo_boxes[index+1][5+(4*i)])),int(float(rows_yolo_boxes[index+1][6+(4*i)]))) , (255,255,0), 10)



            if frame_id in frames_column_format:
                indices=find_indices(frames_column_format,frame_id)

                for i in indices:

                    x_3D=rows_column_format[i+1][2]
                    y_3d=rows_column_format[i+1][3]
                    z_3d=rows_column_format[i+1][4]

                    x_2d=int(float(rows_column_format[i+1][5]))
                    y_2d=int(float(rows_column_format[i+1][6]))

                    img=cv2.circle(img,(x_2d,y_2d), 10, (255,0,0), 3)
                    img=cv2.putText(img,str((x_3D,y_3d,z_3d)),(x_2d-40,y_2d-20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),2)

                    
            
            video.write(img)
            frame_id+=1

        cv2.destroyAllWindows()
        video.release()




    def play_results_video(self):

        desired_fps=30
        
        vid_out = cv2.VideoCapture(self.video_results_filename)
        # Get Frame info for duration calculation
        total_frames = int(vid_out.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = vid_out.get(cv2.CAP_PROP_FPS) 


        # Check if camera opened successfully
        if (vid_out.isOpened()== False):
            print("Error opening video file")
        
        # Read until video is completed
        while(vid_out.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = vid_out.read()
            if ret == True:
            # Display the resulting frame
                cv2.imshow('Frame', frame)
                
            # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        
        # Break the loop
            else:
                vid_out = cv2.VideoCapture(self.video_results_filename)

            time.sleep(1/desired_fps)
        
        # When everything done, release
        # the video capture object
        vid_out.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()
        
            




    def get_K_linear_regression(self):

        lr=LinearRegression(positive=True)

        df = pd.read_csv(os.path.join(self.logs_path,self.matched_yolo_detecions_with_Mobileye_CameraTrackedData_columnformat_filename))

        filtered_X_3D=[df.iloc[i,2] for i in range(len(df.iloc[:,2])) if df.iloc[i,0] in self.list_of_included_frames]
        filtered_Y_3D=[df.iloc[i,3] for i in range(len(df.iloc[:,3])) if df.iloc[i,0] in self.list_of_included_frames]
        filtered_Z_3D=[df.iloc[i,4] for i in range(len(df.iloc[:,4])) if df.iloc[i,0] in self.list_of_included_frames]

        filtered_X_2D=[df.iloc[i,5] for i in range(len(df.iloc[:,5])) if df.iloc[i,0] in self.list_of_included_frames]
        filtered_Y_2D=[df.iloc[i,6] for i in range(len(df.iloc[:,6])) if df.iloc[i,0] in self.list_of_included_frames]

        X_3D=np.array(filtered_X_3D)
        Y_3D=np.array(filtered_Y_3D)
        Z_3D=np.array(filtered_Z_3D)

        X_2D=np.array(filtered_X_2D)
        Y_2D=np.array(filtered_Y_2D)



        pose=np.column_stack(((X_3D/Z_3D),(Y_3D/Z_3D),Z_3D))

        data_2d=np.column_stack((X_2D,Y_2D))


        lr.fit(pose,data_2d)

        print(lr.coef_)
        print(lr.intercept_)

        fx=lr.coef_[0][0]
        fy=lr.coef_[1][1]

        cx=lr.intercept_[0]
        cy=lr.intercept_[1]

        print('fx: ',fx)
        print('fy: ',fy)
        print('cx: ',cx)
        print('cy: ',cy)


        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

        print(len(X_3D))
        print(K)


    


