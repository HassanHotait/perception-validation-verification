
import pandas as pd
import matplotlib.pyplot as plt


metrics_results={

              'Cars AP':[1,2,3,2],
              'Pedestrians AP':[4,5,6,2],
              'All Classes AP' : [7,8,9,2],


              'Cars Precision':[1,2,3,2],
              'Pedestrians Precision':[4,5,6,2],
              'All Classes Precision' : [7,8,9,2],

              'Cars Recall':[1,2,3,2],
              'Pedestrians Recall':[4,5,6,2],
              'All Classes Recall' : [7,8,9,2],

            }

metrics_df=pd.DataFrame(metrics_results)

metrics_df.iloc[:,0:3].plot(kind='bar',title="SMOKE AP Evaluation",figsize=(20,8),grid=True)

plt.xlabel("Difficulties")
plt.ylabel("Percentage %")
plt.yticks(range(0,110,5))
plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
plt.legend(loc=(-0.15,0.7))

plt.show()

metrics_df.iloc[:,3:6].plot(kind='bar',title="SMOKE Precision Evaluation",figsize=(20,8),grid=True)

plt.xlabel("Difficulties")
plt.ylabel("Percentage %")
plt.yticks(range(0,110,5))
plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
plt.legend(loc=(-0.15,0.7))

plt.show()


metrics_df.iloc[:,6:9].plot(kind='bar',title="SMOKE Recall Evaluation",figsize=(20,8),grid=True)

plt.xlabel("Difficulties")
plt.ylabel("Percentage %")
plt.yticks(range(0,110,5))
plt.xticks(ticks=[0,1,2,3],labels=['Easy','Moderate','Hard','Overall'])
plt.legend(loc=(-0.15,0.7))

plt.show()