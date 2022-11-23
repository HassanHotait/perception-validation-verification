
import cv2
# # R=TP/(TP+FN)
# # TP: True Positive Detection | Algorithm correctly detects classifies the object | From Ground Truth
# # FN: False Negative Detection | Algorithm does not detect the object | Detections from model that are not found in ground truth

# # P=TP/(TP+FP)
# # TP: True Positive Detection | Algorithm correctly detects and classifies the object | From Ground Truth
# # FP: False Positive Detection | Algorithm incorrectly detects and incorrectly classifies it


# #               FRAME=
# # Predictions= 

def smoke_get_n_classes(predictions_list):
    n_cars=0
    n_bikes=0
    n_pedestrians=0

    for i in range(len(predictions_list)):
        if predictions_list[i][0]==0.0:
            n_cars+=1
        elif predictions_list[i][0]==1.0:
            n_bikes+=1
        elif predictions_list[i][0]==2.0:
            n_pedestrians+=1

    return (n_cars,n_pedestrians,n_bikes)

def yolo_get_n_classes(predictions_list):
    n_cars=0
    n_bikes=0
    n_pedestrians=0
    for i in range(len(predictions_list)):
        if predictions_list[i]=='car' or predictions_list[i]=='truck' or predictions_list[i]=='bus':
            n_cars+=1
        elif predictions_list[i]=='bicycle' or  predictions_list[i]=='motorbike' :
            n_bikes+=1
        elif predictions_list[i]=='person':
            n_pedestrians+=1
        else:
            pass

    n_detections=n_cars+n_bikes+n_pedestrians

    return (n_detections,n_cars,n_pedestrians,n_bikes)


# # def precision(predictions,ground_truth):
# #     pass

# Python program to check if rectangles overlap
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y



 
# Returns true if two rectangles(l1, r1)
# and (l2, r2) overlap
def rectanges_overlap(l1, r1, l2, r2):
    overlap_condition=None
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        print('False 1')
        overlap_condition=1

        return False,overlap_condition
     
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        print('False 2')
        overlap_condition=2
        return False,overlap_condition
 
    # If one rectangle is above other
    if r1.y < l2.y or r2.y < l1.y:
        # print('L1.Y: ',l1.y)
        # print('R2.Y: ',r2.y)

        # print('L2.Y: ',l2.y)
        # print('R1.Y: ',r1.y)
        overlap_condition=3

        print('False 3')
        return False,overlap_condition
 
    print('rectangles overlap')
    return True,overlap_condition

def boxes_in_list_do_not_intersect(l1,r1,l2,r2):
    overlap_condition=None
     
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        print('False 1')
        overlap_condition=1

        return False,overlap_condition
     
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        print('False 2')
        overlap_condition=2
        return False,overlap_condition
 
    # If one rectangle is above other
    if r1.y < l2.y or r2.y < l1.y:
        # print('L1.Y: ',l1.y)
        # print('R2.Y: ',r2.y)

        # print('L2.Y: ',l2.y)
        # print('R1.Y: ',r1.y)
        overlap_condition=3

        print('False 3')
        return False,overlap_condition
 
    print('rectangles pverlap')
    return True,overlap_condition


def get_IoU(box1,box2):

    # box (Top Left X,Top Left Y,Bottom Right X,Bottom Right Y)

    Pt1=box1[0]
    Pt2=box1[1]

    Pt3=box2[0]
    Pt4=box2[1]


    # l1=Point(x1,y1)
    # r1=Point(x2,y2)

    # l2=Point(x3,y3)
    # r2=Point(x4,y4)


    print('IoU input in function: ',(Pt1,Pt2,Pt3,Pt4))
    boxs_overlap,overlap_condition=rectanges_overlap(Pt1,Pt2,Pt3,Pt4)
    print('IoU boxs overlap in function: ',boxs_overlap)
    if boxs_overlap==True:


        x_inter1=max(Pt1.x,Pt3.x)
        y_inter1=max(Pt1.y,Pt3.y)

        x_inter2=min(Pt2.x,Pt4.x)
        y_inter2=min(Pt2.y,Pt4.y)

        print('(Y2,Y4): ',(Pt2.y,Pt4.y))

        print('Intersection Point 1: ',(x_inter1,y_inter1))
        print('Intersection Point 2: ',(x_inter2,y_inter2))

        width_inter=abs(x_inter2-x_inter1)
        height_inter=abs(y_inter2-y_inter1)

        area_inter=width_inter*height_inter

        print('Width Intersection: ',width_inter)
        print('Height Intersection: ',height_inter)

        

        width_box1=abs(Pt2.x-Pt1.x)
        height_box1=abs(Pt2.y-Pt1.y)

        print('Width Box 1: ',width_box1)
        print('Height Box 1: ',height_box1)

        width_box2=abs(Pt4.x-Pt3.x)
        height_box2=abs(Pt4.y-Pt3.y)

        area_box1=width_box1*height_box1
        area_box2=width_box2*height_box2

        print('Area Box 1: ',area_box1)
        print('Area Box 2: ',area_box2)


        area_union=area_box1+area_box2-area_inter

        print('Area Intersection: ',area_inter)
        print('Area Union: ',area_union)

        iou=area_inter/area_union

    else:
        print('Rectangles Do not intersect')
        iou=0
    
    return iou

def get_key(val):
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }
    for key, value in TYPE_ID_CONVERSION.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def yolo_get_key(val):
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Bicycle': 1,
    'Pedestrian': 2,
    }
    for key, value in TYPE_ID_CONVERSION.items():
        if val == value:
            return key
 
    return "key doesn't exist"


def get_evaluation_metrics(groundtruth,predictions):
    TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
    }

    TP=0
    for i in range(len(groundtruth)):
        true_pt1=Point(float(groundtruth[i][3]),float(groundtruth[i][2]))
        true_pt2=Point(float(groundtruth[i][5]),float(groundtruth[i][4]))
        true_class=groundtruth[i][1]

        for j in range(len(predictions)):

            pred_pt1=Point(predictions[j][2],predictions[j][3])
            pred_pt2=Point(predictions[j][4],predictions[j][5])
            pred_class_ID=predictions[j][0]
            pred_class_string=get_key(pred_class_ID,TYPE_ID_CONVERSION)

            print('Pred',(pred_pt1,pred_pt2))
            print('Truth', (true_pt1,true_pt2))
            IoU=get_IoU((pred_pt1,pred_pt2),(true_pt1,true_pt2))

            if true_class=='Car':

                if IoU>=0.7 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass
            
            elif true_class=='Cyclist':
                if IoU>=0.5 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass

            elif true_class=='Pedestrian':
                if IoU>=0.5 and pred_class_string==true_class:
                    print('True Positive Found')
                    TP=TP+1

                else:
                    pass

            else:
                pass

    return TP


def plot_groundtruth(img,groundtruth):
    thickness=2
    groundtruth_img=img
    for i in range(len(groundtruth)):

        if groundtruth[i][0]!='DontCare':
            color=(0,255,0)
            TL=(int(float(groundtruth[i][4])),int(float(groundtruth[i][5])))
            BR=(int(float(groundtruth[i][6])),int(float(groundtruth[i][7])))

            groundtruth_img=cv2.rectangle(groundtruth_img,TL,BR,color,thickness)
        else:
            pass

    
    return groundtruth_img

def plot_prediction(img,predictions):
    thickness=2
    prediction_img=img
    for i in range(len(predictions)):

        color=(0,0,255)
        TL=(int(float(predictions[i][2])),int(float(predictions[i][3])))
        BR=(int(float(predictions[i][4])),int(float(predictions[i][5])))

        prediction_img=cv2.rectangle(prediction_img,TL,BR,color,thickness)

    
    return prediction_img
# # img=cv2.imread('/home/hasan/perception-validation-verification/SMOKE/datasets/KITTI_MOD_fixed/training/images/2011_09_26_drive_0001_sync_0000000000.png')

# # #               Y1    X1     Y2     X2
# # # static Car 168.83 309.59 238.03 407.51
# # # static Car 180.42 407.07 220.74 468.79
# # # static Car 180.74 465.82 207.1 510.37
# # # static Car 175.42 498.0 202.98 533.96



# # # img=cv2.rectangle(img,(309,168),(407,238),(255,0,0),2)
# # # img=cv2.rectangle(img,(407,180),(468,220),(255,0,0),2)
# # # img=cv2.rectangle(img,(465,180),(510,207),(255,0,0),2)
# # # img=cv2.rectangle(img,(498,175),(534,202),(255,0,0),2)

# # print('IMG Shape: ',img.shape)


# # # img=cv2.rectangle(img,(0,0),(int(img.shape[1]/2),int(img.shape[0]/2)),(255,0,0),2)
# # # img=cv2.rectangle(img,(int(img.shape[1]/2),int(img.shape[0]/2)),(int(img.shape[1]),int(img.shape[0])),(255,0,0),2)

# # # iou_1=get_IoU((0,0,int(img.shape[1]/2),int(img.shape[0]/2)),(int(img.shape[1]/2),int(img.shape[0]/2),int(img.shape[1]),int(img.shape[0])))

# # # print('IoU 1: ',iou_1)

# # # img=cv2.rectangle(img,(0,0),(int(img.shape[1]/2),int(img.shape[0]/2)),(0,255,0),2)
# # # img=cv2.rectangle(img,(0,0),(int(img.shape[1]),int(img.shape[0])),(0,255,0),2)

# # # iou_2=get_IoU((0,0,int(img.shape[1]/2),int(img.shape[0]/2)),(0,0,int(img.shape[1]),int(img.shape[0])))

# # # print('IoU 2: ',iou_2)

# # # test 1
# # # pt1=Point(0,int(img.shape[0]/4))
# # # pt2=Point(int(img.shape[1]/2),int(3*img.shape[0]/4))

# # # # test 2
# # # pt1=Point(0,int(img.shape[0]/4))
# # # pt2=Point(int(img.shape[1]/2),int(3*img.shape[0]/4))

# # # test 2 on your paper

# # # pt1=Point(int(img.shape[1]*0/4),int(img.shape[0]/2))
# # # pt2=Point(int(img.shape[1]/2),int(img.shape[0]))

# # # Test 3 on your paper
# # pt1=Point(int(img.shape[1]*0/4),int(img.shape[0]*0/2))
# # pt2=Point(int(img.shape[1]/2),int(img.shape[0]))

# # print('Pt 1: ',(pt1.x,pt1.y))
# # print('Pt 2: ',(pt2.x,pt2.y))

# # img=cv2.rectangle(img,(pt1.x,pt1.y),(pt2.x,pt2.y),(0,255,0),2)

# # # test 1
# # # pt3=Point(int(img.shape[1]/4),0)
# # # pt4=Point(int(3*img.shape[1]/4),int(img.shape[0]/2))

# # # test 2
# # # pt3=Point(int(img.shape[1]/4)+50,150)
# # # pt4=Point(int(3*img.shape[1]/4),int(img.shape[0]/2))

# # # Test 2 on your paper
# # # pt3=Point(int(img.shape[1]/4),int(img.shape[0]/2))
# # # pt4=Point(int(img.shape[1]*3/4),int(img.shape[0]))

# # # Test 3 on your paper
# # pt3=Point(int(img.shape[1]/4),int(img.shape[0]/4))
# # pt4=Point(int(img.shape[1]*3/4),int(img.shape[0]*3/4))

# # print('Pt 3: ',(pt3.x,pt3.y))
# # print('Pt 4: ',(pt4.x,pt4.y))

# # img=cv2.rectangle(img,(pt3.x,pt3.y),(pt4.x,pt4.y),(0,255,0),2)

# # iou_2=get_IoU((pt3,pt4),(pt1,pt2))

# # print('IoU 2: ',iou_2)




# # cv2.imshow('Output',img)
# # cv2.waitKey(0)



