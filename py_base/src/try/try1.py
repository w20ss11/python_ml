#coding=gbk
'''
Created on 2017年5月6日

@author: wss 
tail(a,(2,1))          argsort 
sorted {dict.items,key=operator.itemgetter(0)}
'''
from numpy.lib.shape_base import tile
from numpy.ma.core import sum, array, arange
import operator
def knn(group,labels,test,k):
    tests=tile(test,(group.shape[0],1))
    diff=group-tests
    diff_pow=diff*diff
    diff_sum=sum(diff_pow,axis=1)
    distance=diff_sum**0.5
    argsort=distance.argsort();
    print('distance:',distance)
    print('argsort:',argsort)
    classcount={}
    for i in arange(k):
        label=labels[argsort[i]]
        print('label:'+label)
        classcount[label]=classcount.get(label,0)+1
    print(classcount)
    sortClasscount=sorted(classcount.items(),key=operator.itemgetter(1))
    print(sortClasscount)
    print('res:',sortClasscount[0][0])
    
group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']
test=[0,0]
k=3
print(group.shape[0])
knn(group, labels, test, k)