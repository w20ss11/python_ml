#coding=gbk
'''
Created on 2017年6月10日

@author: wss
'''
import numpy as np
import matplotlib.pyplot as plt
x = np.random.rand(8)
print(x)
y = np.fft.fft(x)
print(y)
xx = np.fft.ifft(y)
print(xx)

print("-------------sin-------------------")
x = np.linspace(0,2*np.pi,9,endpoint=False)
print(x)
y = np.fft.fft(np.sin(x))
print(y)

print('-------------------------------')
x = np.linspace(-np.pi,np.pi,100)
y = np.cos(x)
plt.plot(x,y)

xx = np.linspace(-np.pi,np.pi,100)
yy = np.cos(xx)+np.cos(3*xx)
plt.plot(xx,yy)
plt.show()


