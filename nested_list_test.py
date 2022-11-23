def yolo_2_smoke_output_format(boxs,classes):

    output_per_frame=[]

    for label,box in zip(classes,boxs):
        if label=='car':
            class_id=0

        elif label=='bike':
            class_id=1

        elif label=='person':
            class_id=2
        
        else:
            class_id=-1
        output_per_frame.append([class_id,-9,box[0],box[1],box[2],box[3]])

    return output_per_frame

print(yolo_2_smoke_output_format([[1,2,3,4]],['car']))