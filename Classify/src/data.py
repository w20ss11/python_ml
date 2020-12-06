#coning='gbk'

import sys

import numpy
from numpy.ma.core import arange, dot, empty, zeros
from numpy.core.numeric import ones
from numpy.core.fromnumeric import transpose
from numpy.lib.shape_base import column_stack
from numpy.ma.extras import row_stack
import sys
fr=open('D:\eclipse_workspace\Classify\data\data.txt')
macList=['c0:38:96:25:5b:c3','e0:05:c5:ba:80:40','b0:d5:9d:46:a3:9b','42:a5:89:51:c7:dd']
X=empty((4,60),numpy.int8)
for line in fr:
    parts=line.split(',')
    try:
        poi=macList.index(parts[2])
        print('poi',poi)
        if poi!='-1':
            print('try parts[2]:',parts[2])
            lie=int(parts[-1].strip())-1
            X[poi,lie] = parts[1]
    except :
        pass
        #print('haha',parts[2])
    else:
        print('no error')
print("final:",type(list),type(1),type(macList))
print(X)
w=ones((4,1))
b=1
print(transpose(w))
z=dot(transpose(w),X)+1
y1=zeros((30,1))
y2=ones((30,1))
y=row_stack((y1,y2))
print(y)
#plt.plot(list)
#plt.show()
