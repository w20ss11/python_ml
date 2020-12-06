#coding = gbk
'''

@author: wss
'''
import os
from numpy.ma.core import arange
def rename(path):
    n = 1;
    for file in os.listdir(path):
        print(file)
        newname = ''+str(n)+'.png'
        nam = os.path.join(path,file)
        if 'ref' in nam :
            print(nam)
        else :
            os.rename(os.path.join(path,file),os.path.join(path,newname))
        print(file,'ok')
        n = n+1
for j in arange(7):
    j=j+3
    path = 'Z:\HDD\dianzikeda_data\shot\class0'+str(j)
    rename(path)
    
#rename("Z:\HDD\dianzikeda_data\shot\class10")

"""rename("X:\\s_class1\\P5")
rename("X:\\s_class2\\P5")
rename("X:\\s_class3\\P5")
rename("X:\\s_class4\\P5")
rename("X:\\s_class5\\P5")"""


"""for i in arange(7):
    i=i+1
    for j in arange(3):
        j=j+1
        path='Y:\\class'+str(j)+'\\P'+str(i)
        rename(path)"""
    
