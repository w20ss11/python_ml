#coding=gbk
'''
Created on 2017年4月14日

@author: wss
'''
import scipy.optimize
from numpy.ma.core import ones, dot
from src import kNN
def fun(k,x,y):
    print('haha')
    cost=(k*x-y)**2
    w_grad=1
    return [cost,w_grad]
def fun1():
    return 1

k0=1
x=1
y=1
max_iterations=10
#scipy.optimize.minimize(fun,k0,args=(x,y,),options = {'maxiter': max_iterations})
#scipy.optimize.leastsq(fun1,)

a=ones(2)
b=ones(2).reshape(2,1)
b[0,0]=2
b[1,0]=4
print(a,b)
print(a*b)
print(dot(a,b))

print('-------------------------')

import sys
sys.path.append('D:\eclipse_workspace\py_base\src') 
group,labels=kNN.createDataSet()
print('group:',group)
print('labels:',labels)
res=kNN.classify0([0,0], group, labels, 3)
print(res)