import numpy as np
import matplotlib.pyplot as plt

N = 3
expected_yolo_improvement = (56,36.23, 29.55)#, 35, 27)
yolo_v3_coco_trained_experimental_results = (32.4,12.96, 29.55)#, 20, 25)
yolo_v3_kitti__trained_paper_results=(56,36.23, 29.55)
smoke_v3_kitti_paper_results=(92.88,86.95,29.55)
smoke_v3_kitti_experimental_results=(99.44,96.15,93.58)

ind = np.arange(N)  # the x locations for the groups
width =0.2     # the width of the bars
offset=0.2
fig = plt.figure()
ax = fig.add_subplot(111)

# COCO Experimental Results Stacked with potential improvement
rects1 = ax.bar(ind+(offset), expected_yolo_improvement, width)#, color='royalblue')
rects2 = ax.bar(ind+(offset), yolo_v3_coco_trained_experimental_results, width,)# color='seagreen')

# COCO Experimental Results Stacked with potential improvement
rects3 = ax.bar(ind+(2*offset), expected_yolo_improvement, width)#, color='royalblue')
rects4 = ax.bar(ind+(3*offset), smoke_v3_kitti_paper_results, width)#, color='seagreen')
rects5 = ax.bar(ind+(4*offset), smoke_v3_kitti_experimental_results, width)#, color='seagreen')



# add some
ax.set_ylabel('AP')
ax.set_title('Average Precision on KITTI Dataset')
ax.set_xticks([0.5,1.5,2.5])
ax.set_xticklabels( ('Easy AP', 'Moderate AP', 'Hard AP') )

ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0],rects5[0]), ('YOLOv3 Expected Improvement','YOLOv3 trained on COCO Experimental Results','YOLOv3 trained on KITTI Paper Results','SMOKE Paper Results','SMOKE Experimental Results') ,loc=(0.7,0.99))
plt.grid(True)
plt.yticks(range(0,110,10))
#plt.legend(loc=(0.1,0.1))
plt.show()