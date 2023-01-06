import numpy as np
import matplotlib.pyplot as plt




x=[]#list(range(0,70))
# x=[np.random.randint(0,70) for i in list(range(70))].sort()
y=[]

print(np.random.randint(0,70))

for i in range(70):
    x.append(np.random.randint(0,70))

#print("X Range: ",x)

x.sort()
for i in x:
    if 0<=i<10:
        y.append(i)

    elif 10<=i<=20:
        y.append((2*i+y[-1]))
    elif 20<=i<=30:
        y.append((5*i+y[-1]))

    elif 30<=i<=40:
        y.append((-i+y[-1]))

    elif 40<=i<=50:
        y.append((i+y[-1]))

    elif 50<=i<=60:
        y.append((i*3+y[-1]))

    elif 60<=i<=70:
        y.append((i*3+y[-1]))
    else:
        pass


print("length x",len(x))
print("length y",len(y))
#plt.scatter(x,y)


for i in range(7):
    print("iteration: ",i)
    plt.plot(np.unique(x[(i*10):(i+1)*10]), np.poly1d(np.polyfit(x[(i*10):(i+1)*10], y[(i*10):(i+1)*10], 1))(np.unique(x[(i*10):(i+1)*10])))

#plt.plot(np.unique(x[(i*10):(i+1)*10]), np.poly1d(np.polyfit(x[(i*10):(i+1)*10], y[(i*10):(i+1)*10], 1))(np.unique(x[(i*10):(i+1)*10])))
# plt.plot(np.unique(x[0:10]), np.poly1d(np.polyfit(x[0:10], y[0:10], 1))(np.unique(x[0:10])))
# plt.plot(np.unique(x[10:20]), np.poly1d(np.polyfit(x[10:20], y[10:20], 1))(np.unique(x[10:20])))
# plt.plot(np.unique(x[20:30]), np.poly1d(np.polyfit(x[20:30], y[20:30], 1))(np.unique(x[20:30])))
# plt.plot(np.unique(x[30:40]), np.poly1d(np.polyfit(x[30:40], y[30:40], 1))(np.unique(x[30:40])))
# plt.plot(np.unique(x[40:50]), np.poly1d(np.polyfit(x[40:50], y[40:50], 1))(np.unique(x[40:50])))
# plt.plot(np.unique(x[50:60]), np.poly1d(np.polyfit(x[50:60], y[50:60], 1))(np.unique(x[50:60])))
# plt.plot(np.unique(x[60:70]), np.poly1d(np.polyfit(x[60:70], y[60:70], 1))(np.unique(x[60:70])))
plt.show()
