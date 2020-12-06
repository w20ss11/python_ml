#coding=gbk
'''
Created on 2017年5月23日

@author: wss
'''
import numpy as np

def getData():
    f = open("D:\\eclipse_workspace\\py_base\\src\\data\\testSet.txt")
    x0 = []
    x1 = []
    x2 = []
    y = []
    for line in f:
        strs = line.split(" ")
        x0.append(1.0)
        x1.append(float(strs[0]))
        x2.append(float(strs[1]))
        y.append(float(strs[2]))
    x0 = np.array(x0).reshape([100,1])
    x1 = np.array(x1).reshape([100,1])
    x2 = np.array(x2).reshape([100,1])
    y = np.array(y).reshape([100,1])
    w = np.ones([3,1])
    w = w.astype(np.float32)
    return x0,x1,x2,w,y

def sigmoid(x):
    return 1.0/(1 + np.exp(-x)) #负号

x0,x1,x2,w,y = getData()
x = np.hstack((x0,x1,x2))

print(w)
for i in np.arange(1000):
    y_ = sigmoid(np.matmul(x,w))
    alpha = 0.01
    #w[1] = w[1] - alpha*np.sum((y_-y)*x1)
    #w[2] = w[2] - alpha*np.sum((y_-y)*x2)
    w = w + alpha*x.transpose()*(y_-y)
print(w)