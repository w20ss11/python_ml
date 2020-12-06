#coding=gbk
'''
Created on 2017年4月8日

@author: wss
'''
import numpy
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
from numpy import *
import sys
from numpy.lib.function_base import average
from numpy.ma.core import maximum
b=arange(24).reshape(2,3,4)
print(b)
c=b.ravel()
print(c)
print('------------------')
d=b.flatten()
print(d)
print('------------------')
#b.shape=(2,12)
#print(b)
print('------------------')
print(b.transpose())
print('-------组合-----------')
m=arange(9).reshape(3,3)
print(m)
n=m*2
print(n)

print('-------分割-----------')
a=arange(9).reshape(3,3)
print(a)
print(hsplit(a, 3))
print(a)
print(vsplit(a, 3))

print()
c=arange(27).reshape(3,3,3)
print(c)
print(dsplit(c, 3))

print('-------转换-----------')
c=arange(27).reshape(3,3,3)
a=[1,2,3]
#print(a.astpye(double))

print('-------txt读写-----------')
e=eye(2)
print(e)
savetxt("eye.txt",e)

print('-------csv-----------')
t,y=loadtxt('haha1.csv', delimiter=',', usecols=(6,7), unpack=True)
print(t)
print(y)
c=loadtxt('haha1.csv', delimiter=',', usecols=(6,7), unpack=False)


print('-------均值 最大小值 排序-----------')
print(c)
print(average(t))
print(mean(t))
print(max(t))
print(min(t))

c=arange(5)
c[2]=77
c[3]=88
print(c)
print(median(c))
print(msort(c))
print('-------方差-----------')
print(var(c))
print('c:',c)
print('len:',len(c))
print('c.size:',c.size)
aver=mean((c-mean(c))**2)
print(aver)

print('-------diff std-----------')
print('c     :',c)
print('diff  :',diff(c))
print('c[:-1]:',c[:-1])
print(std(c))
r=arange(10)
r[0]=1
print(log(r))

print('-------where take-----------')
u=arange(10)
u[7]=77
print(u)
print(numpy.where(u>6))
index=numpy.where(u>6)
print(index)
print(u.take(index))
print(take(u,index))

print('-------sqrt take-----------')
a=arange(4)
print(a)
a[1]=121
a[2]=256
a[3]=17**2
print(sqrt(a))
print(a.take(3))


print('-------argmax-----------')
print(c)
print(argmax(c))


print('-------apply_along_axis-----------')
arr=arange(12).reshape(3,4)
print('arr:')
print(arr)
def add(arr):
    arr[1][1]=999
    return 999
res=apply_along_axis(min, 1, arr)
print(res)
   

print('-------sys.argv maximum-----------') 
a=sys.argv[0]
print(a)
arr.reshape(2,6)
a=arange(9).reshape(3,3)
b=eye(3)
print(a)
print(b)
maximum_res=maximum(a,b)
print('maximum_res:')
print(maximum_res)


print('-------convolve exp linspace fill-----------') 
a=ones((3,3), int32)
print(a)
b=ones((3,3),int32)
#h1=convolve(a,b,'full')
print()
y=convolve([1, 2, 3], [1,1])
print(y)
print(exp(arange(5)))

print(linspace(1,10,4))

a=arange(4)
b=ones((4),int32)
print('a:',a,'b:',b)
a.fill(999)
print(a)


print('-------bmat-----------') 
a=eye((2))
b=2*a
print(a)
print(b)
print(bmat("a b;a b"))
a.flat=999
print(a)

z=arange(4).reshape(2,2)
print('z:',z)
def myfun(a):
    res=zeros_like(a)
    res.flat=888
    return res
myfun_var=frompyfunc(myfun,1,1)
print('myfun res:',myfun_var(z))
#print(myfun_var.reduce(z))

a=9
b=8
t = linspace(-pi, pi, 201)
x = sin(a * t + pi/2)
y = sin(b * t)
plot(x,y)
#show()


a=ones((2,2),int32)
b=ones((2,2),int32)
print(a*b)
print(mat(a)*mat(b))
