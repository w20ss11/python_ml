#coding=gbk
'''
Created on 2017年3月31日

@author: wss
'''
#name=input("what's your name?")
#print('hello',name)
from pip._vendor.requests.packages.urllib3.connectionpool import xrange

x='hello'
y='world'
print(x+y)
print(x,y)
#print(x y) error
print('-----part1-------------\n')

#name=input("what's your name?")
#print('hello'+name+'!')

print('1haha \nhehe')
print(r'2haha \ hehe')
print('-----part2-------------\n')

print(10*'hengheng')
liebiao=['haha','heihei']+10*['xixi']
print(liebiao[5])
print('-----part3-------------\n')

name='wss'
print('s' in name)
print('-----part3-------------\n')

num=[12,3,21,234,1225,38,76,0,3437]
x=len(num)
ma=max(num)
mi=min(num)
print('len:'+str(x)) #str(x)是数字 'x'是x
print('max:'+str(ma)) 
print('min:'+str(mi)) 
print('-----part4-------------\n')

name="abcdefghijklmn"
print(list(name))
print('-----part5-------------\n')

for i in xrange(10):
    print(i)
print('-----part6-------------\n')