#coding=gbk
'''
Created on 2017年5月3日

@author: wss
'''
f=open('D:\eclipse_workspace\Classify\data\data.txt')
for l in f:
    print(l)
    strs=l.split(',')
    print(strs[2])