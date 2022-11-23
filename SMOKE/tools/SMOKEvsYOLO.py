import matplotlib.pyplot as plt
import numpy as np

with open('record_detections_SMOKE.txt','r') as file:
    smoke_file=file.read()


SMOKE_list=smoke_file.split()
SMOKE = [SMOKE_list[x:x+5] for x in range(0,int(len(SMOKE_list)),5)]

with open('record_detections_YOLO.txt','r') as file:
    yolo_file=file.read()


YOLO_list=yolo_file.split()
YOLO = [YOLO_list[x:x+5] for x in range(0,int(len(YOLO_list)),5)]

smoke_n_detections_yaxis=[float(list_elements[1]) for list_elements in SMOKE ]
yolo_n_detections_yaxis=[float(list_elements[1]) for list_elements in YOLO ]
print('n elements: ',len(SMOKE))




fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')

axs[0].bar(range(len(smoke_n_detections_yaxis)),smoke_n_detections_yaxis,label='SMOKE')
axs[0].set_xticks(range(0,len(smoke_n_detections_yaxis),2))
axs[0].set_ylabel('# SMOKE Detections')
axs[0].grid(True)

# axs[0].set_xticks
# axs[0].set_xticklabels(rotation = (45), fontsize = 10)
axs[1].bar(range(len(smoke_n_detections_yaxis)),yolo_n_detections_yaxis,label='YOLOV3')
axs[1].set_xticks(range(0,len(smoke_n_detections_yaxis),2))
axs[1].set_ylabel('# YOLOV3 Detections')
axs[1].grid(True)

error=np.array(smoke_n_detections_yaxis)-np.array(yolo_n_detections_yaxis)

axs[2].bar(range(len(smoke_n_detections_yaxis)),np.abs(error),label='ERROR')
axs[2].set_xticks(range(0,len(smoke_n_detections_yaxis),2))
axs[2].set_ylabel('# Detections Error')
axs[2].grid(True)


#plt.legend(loc='lower rightq')
# plt.ylim((0 ,10))
# plt.xlim((0,110))
plt.show()
 
    


