#coding=gbk
'''
Created on 2017

@author: wss
'''
import numpy as np
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
x=np.linspace(1,10,10)
print(x)
fun=np.poly1d(np.array([1,0,0]))
y=fun(x)
plot(x,y,'--r')
plt.xlabel('x')
show()

fun1=np.poly1d(np.array([1,0]))
fun2=np.poly1d(np.array([1,0,0]))
x=np.linspace(1,100,10)
y1=fun1(x)
y2=fun2(x)

plt.subplot(211)
plt.title('haha')
plt.plot(x,y1)
plt.subplot(212)
plt.title('xixi')
plt.plot(x,y2,'--r')
show()

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot([0,1], [0,1])
ax.set_title("ax1")
show()

