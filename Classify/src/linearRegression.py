#coding=gbk
'''
Created on 2017年4月29日

@author: wss
'''
from numpy.core.function_base import linspace
import numpy
from numpy.ma.core import arange, ones
import scipy.optimize
from numpy.ma.extras import row_stack, column_stack
import matplotlib.pyplot as plt

def fun(w,x):
    k0,k1,k2,k3=w
    x1,x2,x3=x
    y=k1*x1+k2*x2+k3*x3+k0
    return y
def error(w,Xi,Yi,iter):
    #print('iter:',iter)
    return fun(w,Xi)-Yi

x1=linspace(6,20,10,dtype=int)
x2=linspace(0,10,10,dtype=int)
x3=linspace(20,31,10,dtype=int)
x=numpy.array([x1,x2,x3])
print('x:',x)
w=numpy.array([15,5,8,10],dtype=float)
y=fun(w,x)
print('y:',y,type(y))

#--------leastsq---------------
#res=scipy.optimize.leastsq(error, [1,1,1,1], args=(x,y,300))
#print('res:',res[0]) #res: [ 15.   5.   8.  10.]

#---------mine-------------------
w=[1,1,1,1]
alpha=0.003
z=fun(w,x)
print('sum((z-y)',sum((z-y)))
for i in arange(300):
    z=fun(w,x)
    print('error:',error(w, x, y, i))
    print('iter:',i)
    w[0]=w[0]-alpha*sum(error(w, x, y, i))/10
    w[1]=w[1]-alpha*sum(error(w, x, y, i)*x1)/10
    w[2]=w[2]-alpha*sum(error(w, x, y, i)*x2)/10
    w[3]=w[3]-alpha*sum(error(w, x, y, i)*x3)/10
    print('w:',w)
