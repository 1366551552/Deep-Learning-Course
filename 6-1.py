# Deep-Learning-Course
import matplotlib.pyplot as pl
import numpy as np
pl.rcParams['font.sans-serif']='SimHei'
pl.rcParams['axes.unicode_minus']=False
x=np.array([137.97,104.5,100,124.32,79.2,99,124,114,106.69,138.05,53.75,46.91,68,63.02,81.26,86.21])
y=np.array([145,110,93,116,65.32,104,118,91,62,133,51,45,78.5,69.65,75.69,95.3])
pl.scatter(x,y,color="red")
pl.title("商品房销售记录",color="blue",fontsize=16)
pl.xlim(30,200)
pl.ylim(30,200)
pl.xlabel('面积(平方米)',fontsize=14)
pl.ylabel('价格(万元)',fontsize=14)
pl.show()
