import json
import os
import csv


class detectionInfo(object):
    def __init__(self, line):
        #print("length of line: ",len(line))
        self.name = line[0]


        # self.truncation = float(line[1])
        # self.occlusion = int(line[2])

        # # local orientation = alpha + pi/2
        self.alpha = float(line[1])

        # in pixel coordinate
        self.xmin = float(line[2])
        self.ymin = float(line[3])
        self.xmax = float(line[4])
        self.ymax = float(line[5])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[6])
        self.w = float(line[7])
        self.l = float(line[8])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[9])
        self.ty = float(line[10])
        self.tz = float(line[11])

        # global orientation [-pi, pi]
        self.rot_global = float(line[12])
        self.score=float(line[13])

def read_prediction(predictions_folder_path,fileid):

    TYPE_ID_CONVERSION = {
                        'Car': 0,
                        'Cyclist': 1,
                        'Pedestrian': 2,
                        'DontCare' : -1
                        }
    with open(os.path.join(predictions_folder_path,str(fileid).zfill(6)+'.txt'),'r') as f:
        reader=csv.reader(f)
        rows=list(reader)

    # print("predictions_folder: ",predictions_folder_path)
    # print("fileid: ",fileid)
    # print("rows: \n",rows)
    # for row in rows:
    #     print("type(row[0]): ",type(row[0]))
    #     print("row[0]: ",row[0])
        # print('row[0]: ',row[0])
        # print('row: ',row[0].split(' '))

    # predictions=[[float(row[0].split(' '))]]

    # smoke_predictions_read_from_file=[[float(row[0].split(' ')[i])  if i!=0 else TYPE_ID_CONVERSION[row[0].split(' ')[i] ] for i in [0]+list(range(3,len(row[0].split(' '))))] for row in rows]

    smoke_predictions_read_from_file=[[float(row[0].split(' ')[i])  if i!=0 else row[0].split(' ')[i]  for i in [0]+list(range(3,len(row[0].split(' '))))] for row in rows if len(row)!=0]
    #smoke_predictions_read_from_file=[[float(row[0].split(' ')[i])  if i!=0 else TYPE_ID_CONVERSION[row[0].split(' ')[i] ] for i in [0]+list(range(3,len(row[0].split(' '))))] for row in rows]

    return smoke_predictions_read_from_file



for i in range(4952):

    frame_predictions=read_prediction("C:\\Users\\hashot51\\Desktop\\TensorFlow-2.x-YOLOv3\\yolo_predictions_kitti_format",i)
    for prediction in frame_predictions:
        pred=detectionInfo(prediction)

        if pred.name=="donut":
            print("filename: ",i)


        predictions_filename=f'{pred.name}_predictions.json'
        bbox = str(pred.xmin) + " " + str(pred.ymin) + " " + str(pred.xmax) + " " +str(pred.ymax)
        pred_info={"confidence":pred.score, "fileid":str(i),"bbox":bbox}
        predictions_filepath=os.path.join("C:\\Users\\hashot51\\Desktop\\TensorFlow-2.x-YOLOv3\\converted_predictions",predictions_filename)
        #predictions_file=
        if os.path.exists(predictions_filepath):
            predictions_file=open(predictions_filepath,"r+")
            pred_list=json.load(predictions_file)
            #print("pred list: ",pred_list)
            pred_list.append(pred_info)
            predictions_file.seek(0)
            json.dump(pred_list,predictions_file)
        else:
            with open(predictions_filepath,"w") as predictions_file:
                json.dump([pred_info],predictions_file)



