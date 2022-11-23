import numpy as np
import pandas as pd
import random

#ar=np.zeros((5,))

random.seed(4)
TP=[random.randint(1,3) for i in range(5)]
print('TP: ',TP)
FP=[random.randint(1,3) for i in range(5)]
print('FP: ',FP)

FN=[random.randint(1,3) for i in range(5)]
print('Fn: ',FN)

ΣTP=[sum(TP[0:i+1]) for i in range(len(TP))]

ΣFP=[sum(FP[0:i+1]) for i in range(len(FP))]

ΣFN=[sum(FN[0:i+1]) for i in range(len(FN))]

print('ΣTP: ',ΣTP)

ar=np.column_stack((TP,FP,FN,ΣTP,ΣFP,ΣFN))



data=pd.DataFrame(ar,columns=['TP','FP','FN','ΣTP','ΣFP','ΣFN'])#,'ΣTP','ΣFP','ΣFN'])

print(data)
