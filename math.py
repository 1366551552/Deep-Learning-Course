print('a*x*x+b*x+c=0')
a= float (input('a='))
b= float (input('b='))
c= float (input('c='))
import math
if a==0:
    if b==0:
        if c==0:
            print("方程有任意解")
        else:
            print("方程无解")
    else:
       x=-c/b
       print("方程有解：x=%.2f" %x)

else:
    p=b*b-4*a*c
    if p<0:
      print('无解')
    elif p==0:
        x=-b/(a*2)
        print("方程有解：x1=x2=%.2f" %x)
    else:
        x1 = (-b+math.sqrt(p))/(2*a)
        x2 = (-b-math.sqrt(p))/(2*a)
        print("方程有两个解：x1=%.2f,x2=%.2f" %(x1,x2))
