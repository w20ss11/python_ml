#coding=gbk
'''
Created on 2017年5月1日

@author: wss
'''
from numpy.core.function_base import linspace
import numpy
from numpy.ma.core import arange, ones
import scipy.optimize
import matplotlib.pyplot as plt

def fun(w,x):
    k,b=w
    return k*x+b
def error(w,x,y,iter):
    print('iter:',iter)
    return fun(w,x)-y

#x=linspace(1,50,20,dtype=int)
#print('x:',x)
#w=[5,13]
#y=fun(w, x)
#print('y:',y)
Xi=numpy.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
Yi=numpy.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])

w=numpy.array([100,2],dtype='float')
#------------leastsq------------------------
"""
res=scipy.optimize.leastsq(error,w,args=(Xi,Yi,100))
print('res:',res[0])

plt.scatter(Xi,Yi,color='red')
x=linspace(0,10,100,dtype=int)
w=res[0]
plt.plot(x,fun(w,x),'--')
plt.show()""" #res: [ 0.61349535  1.79409255]
#-------------mine-------------------------

alpha=0.03
for i in arange(200):
    z=fun(w,Xi)
    print('error(w, Xi, Yi, i):',error(w, Xi, Yi, i))
    w[0]=w[0]-alpha*sum(error(w, Xi, Yi, i)*Xi)/7
    w[1]=w[1]-alpha*sum(error(w, Xi, Yi, i))/7
    print('w iter:',w)
print('w:',w)
plt.scatter(Xi,Yi,color='red')
x=linspace(0,10,100,dtype=int)
plt.plot(x,fun(w,x),'--')
plt.show()
#log: 0.03 w: [ 0.59562777  1.90639941]
#-----------------------------------------