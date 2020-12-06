'''
Created on

@author: wss
'''
import time

def getTime():
    return time.strftime("%Y%m%d%H%M%S",time.localtime())
def getData(filePath,apName):
    f = open(filePath,'r',encoding='utf-8')
    f_new = open('D://text//'+apName+'_'+getTime()+'.txt','a')
    for line in f:
        strs = line.split(' ')
        for str in strs:
            if apName == str.split(':')[0]:
                print(str.split(':')[-1])
                f_new.write(str.split(':')[-1]+'\n')
            else:
                pass
        #print(line)
getData('D://text//data2.txt', 'HUAWEI-KEPM3D')