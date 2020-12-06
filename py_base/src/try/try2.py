#coding=gbk
'''
Created on 2017��5��6��

@author: wss 
len(dataset)        set(list) 
temp.pop(feat_id)   copy.deepcopy(dataset[m]) 
not
'''
from numpy.ma.core import arange
from math import log
import operator
import copy
def calcEntropy(dataset):
    countclass={}
    for i in arange(len(dataset)):
        countclass[dataset[i][-1]]=countclass.get(dataset[i][-1],0)+1
    entropy=0
    sum=len(dataset)  # @ReservedAssignment
    for value in countclass.values():
        temp=value/sum
        entropy+=-temp*log(temp)
    return entropy

def chooseMaxGain(dataset,labels):
    entropy=calcEntropy(dataset)
    feature_num=len(dataset[0])-1#һ������������
    gainCount={}#��¼ÿ������������
    for i in arange(feature_num):
        list=[]  # @ReservedAssignment
        for j in arange(len(dataset)):
            list.append(dataset[j][i])
        list=set(list)#��i�������ж�����ֵ      @ReservedAssignment
        diff=0.0
        for value in list:#����ÿ��ֵ��subdataset
            subdataset=[]
            for m in arange(len(dataset)):
                if value==dataset[m][i]:
                    subdataset.append(dataset[m])
            subEntroy=calcEntropy(subdataset)#��i�������е�j��ֵ�ֳ��������ݼ�����
            diff+=len(subdataset)/len(dataset)*subEntroy#���غͶ�Ӧ�����˻������������ı�����
            
        gainCount[i]=entropy-diff
    sortedGainCount=sorted(gainCount.items(),key=operator.itemgetter(1))
    print(sortedGainCount)
    #argsortGainCount=gainCount.argsort()
    print(labels[sortedGainCount[0][0]])
    return sortedGainCount[0][0]
    
def judgelabelSame(dataset):
    bool=False  # @ReservedAssignment
    values=set()
    for data in dataset:
        values.add(data[-1])
    if len(values)==1:
        bool=True  # @ReservedAssignment
        res=dataset[0][-1]
        print(dataset)
        print("       ȫ���������:"+res)
    return bool

def splitdataset(dataset,labels,feat_id):
    list=[]  # @ReservedAssignment
    for i in arange(len(dataset)):
        list.append(dataset[i][feat_id])
    list=set(list)#��i�������ж�����ֵ      @ReservedAssignment
    subdatasets=[]
    
    for value in list:#����ÿ��ֵ��subdataset
        subdataset=[]
        for m in arange(len(dataset)):
            if value==dataset[m][feat_id]:
                temp=copy.deepcopy(dataset[m])
                temp.pop(feat_id)
                #dataset[m].pop(feat_id)
                subdataset.append(temp)
        subdatasets.append(subdataset)
    labels.pop(feat_id)
    return subdatasets,labels

dataset=[[1,1,'yes'],
         [1,1,'yes'],
         [1,0,'no'],
         [0,1,'no'],
         [0,1,'no']]
labels=['no surfacing','flippers']
print(calcEntropy(dataset))
chooseMaxGain(dataset, labels)

dataset1=[[1,1,'no'],
         [1,1,'no'],
         [1,0,'no'],
         [0,1,'no'],
         [0,1,'no']]
print(judgelabelSame(dataset1))
print(splitdataset(dataset,labels, 0))

print('-------main-----------')
def main(dataset,labels):
    bool=judgelabelSame(dataset)  # @ReservedAssignment
    if bool :
        print(dataset[0][-1])
    else:
        feat=chooseMaxGain(dataset, labels)
        subdatasets,sublabels=splitdataset(dataset, feat)
        for subdataset in subdatasets:
            for sublabel in sublabels:
                 main(subdataset, sublabel)
print('---------------------------')
dataset=[[1,1,'yes'],
         [1,1,'yes'],
         [1,0,'no'],
         [0,1,'no'],
         [0,1,'no']]
labels=['no surfacing','flippers']

id=chooseMaxGain(dataset, labels)
subdatasets,sublabels=splitdataset(dataset, labels, id)
print('subdatasets:',subdatasets)
print('sublabels��',sublabels)
for i in arange(len(subdatasets)):
    if not judgelabelSame(subdatasets[i]):
        id=chooseMaxGain(subdatasets[i], sublabels)
        subdatasets,sublabels=splitdataset(dataset, labels, id)
        for j in arange(len(subdatasets)):
            if not judgelabelSame(subdatasets[i]):
                print(subdatasets,sublabels)
        