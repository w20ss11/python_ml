#coding=gbk
'''
Created on 2017年4月1日

@author: wss
'''
from pip._vendor.distlib.compat import raw_input

#列表
liebiao=['wss','0927','haha',]
liebiao.append('append')
print('append:  ',liebiao)

liebiao.extend(['extend1','extend2'])
print('extend:  ',liebiao)

liebiao.insert(0, 'insert')
print('insert:  ',liebiao)

liebiao.remove('wss')
print('remove:  ',liebiao)

index=liebiao.index('0927', )
print(index)
print('-----part1-------------\n')

def hello(name):
    print('hello,'+name+'!')
name=raw_input("what's your name?")
hello(name)
    
    
    