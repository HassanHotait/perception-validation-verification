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

coefs=[]

y_smoothed=[]
for i in range(7):
    a,b=np.polyfit(x[(i*10):(i+1)*10], y[(i*10):(i+1)*10], 1)
    y_smooth=a*(i*10)+(b)
    y_smoothed.append(y_smooth)


print("length of Y: ",len(y_smoothed))
print("Y values calculated with coeffs: ",y_smoothed)

fig,ax = plt.subplots(nrows=1,ncols=1)
ax.plot(list(range(0,70,10)),y_smoothed)
plt.show()


