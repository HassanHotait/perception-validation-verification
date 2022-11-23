


# # import csv
# # import numpy as np
# # import pandas as pd
# # from IPython.display import display

# # # with open('/home/hasan/perception-validation-verification/results/Streamkitti_training_set_smoke_metrics2022_11_17_10_16_29/plot/car_detection.txt','r') as f:
# # #     reader=csv.reader(f)
# # #     data=list(reader)


# # # data=[row[0].split(' ') for row in data]

# # # column1=[float(row[0]) for row in data]
# # # column2=[float(row[1]) for row in data]
# # # column3=[float(row[2]) for row in data]
# # # column4=[float(row[3]) for row in data]


# # # print(column1)
# # # print(len(column1[1:]))

# # # print('Column 1 Average: ',sum(column1[1:])/40)
# # # print('Column 2 Average: ',sum(column2[1:])/40)
# # # print('Column 3 Average: ',sum(column3[1:])/40)
# # # print('Column 4 Average: ',sum(column4[1:])/40)


# # column_headers=['Class/Difficulty','Easy','Moderate','Hard','Overall']
# # row_headers=['Car','Pedestrian','Overall']

# # data=np.zeros((3,4))

# # ARRAY=np.column_stack((row_headers,data))

# # df=pd.DataFrame(ARRAY)

# # blankIndex=['']*len(df)

# # df.index=blankIndex
# # df

# # #print(df)



# # ARRAY=np.row_stack((column_headers,ARRAY))
# # df=pd.DataFrame(ARRAY)

# # df.style.hide_index()




# # str1='AP: {} - Precision: {} - Recall: {}'.format(1,2,3)


# # cars=[str1]
# import numpy as np

# cars_easy_AP=0.4
# cars_moderate_AP=0.4
# cars_hard_AP=0.4




# easy_cars='AP: {} - Precision: {} - Recall: {}'.format(cars_easy_AP,0,0)
# moderate_cars='AP: {} - Precision: {} - Recall: {}'.format(cars_easy_AP,0,0)
# hard_cars='AP: {} - Precision: {} - Recall: {}'.format(cars_hard_AP,0,0)
# overall_cars='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)
# cars=[easy_cars,moderate_cars,hard_cars,overall_cars]
# easy_pedestrians='AP: {} - Precision: {} - Recall: {}'.format(cars_easy_AP,0,0)
# moderate_pedestrians='AP: {} - Precision: {} - Recall: {}'.format(cars_easy_AP,0,0)
# hard_pedestrians='AP: {} - Precision: {} - Recall: {}'.format(cars_hard_AP,0,0)
# overall_pedestrians='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)
# pedestrians=[easy_pedestrians,moderate_pedestrians,hard_pedestrians,overall_pedestrians]

# easy_classes='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)
# moderate_classes='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)
# hard_classes='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)
# overall_classes='AP: {} - Precision: {} - Recall: {}'.format(0,0,0)

# classes=[easy_classes,moderate_classes,hard_classes,overall_classes]


# column_headers=['Class/Difficulty','Easy','Moderate','Hard','Overall']
# row_headers=['Car','Pedestrian','Overall']


# ar1=np.array(cars)
# ar2=np.array(pedestrians)
# ar3=np.array(classes)

# data=np.row_stack((ar1,ar2,ar3))
# print(ar1)
# import numpy as np
# a=[1,2,3]
# b=[2,4,6]

# c=np.add(a,b)
# print(np.array(a)+np.array(b))

# print('{:.2f} is something'.format(3.5555555))

import keyboard
import keyboard
 
while True:
   
    print(keyboard.read_key())
    if keyboard.read_key() == "a":
        break