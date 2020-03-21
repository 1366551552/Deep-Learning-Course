# Deep-Learning-Course
import math 
import numpy as np
np.random.seed(612)
a = np.random.rand(1000)
k=1
x=input()
x=int(x)
print("序号",end="   ")
print("索引值",end="    ")
print("随机数")
for i in range (0,999,1):
    if(i%x==0):
        if(k<10):
            print(k,end="   ")
        else:
            print(k,end="  ")
        print(i,end="    ")
        print(a[i])
        k=k+1
