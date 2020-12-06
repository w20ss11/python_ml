#coding = gbk
'''

@author: wss
'''
import os
from numpy.ma.core import arange
import shutil

def copyfile(file,path):
    if os.path.isfile(path):
        os.system('rm -fr '+file)
    else:
        shutil.copy(file, path)
        print("copy %s to %s successful"   % (file,path))
        
def rename(path):
    for i in range(10):
        path_class = path+"class"+str(i+1).zfill(2)
        print('path_class:',path_class)
           
        n_train = 1 
        n_test = 1
        n = 1
        for file in os.listdir(path_class):
            nam_source = os.path.join(path_class,file)
            #print(nam_source)
            if n<=70:
                path_train = "Z:\\HDD\\dataset_dzkd_radar\\trains\\"
                copyfile(nam_source, path_train)
                
                nam_new = 'class'+str(i+1).zfill(2)+'_'+str(n_train).zfill(3)+'.png'
                nam_copy = os.path.join(path_train,file)
                nam_new = os.path.join(path_train,nam_new)
                print('nam_copy:',nam_copy)
                print('nam_new:',nam_new)
                os.rename(nam_copy,nam_new)
                n_train = n_train+1
#             elif n<=100:
#                 path_test = "Z:\\HDD\\data_dzkd_png\\test\\"
#                 copyfile(nam_source, path_test)
#                 
#                 nam_new = 'class'+str(i+1).zfill(2)+'_'+str(n_test).zfill(3)+'.png'
#                 nam_copy = os.path.join(path_test,file)
#                 nam_new = os.path.join(path_test,nam_new)
#                 print('nam_copy:',nam_copy)
#                 print('nam_new:',nam_new)
#                 os.rename(nam_copy,nam_new)
#                 n_test = n_test+1
            n = n+1
     
     
path = 'Z:\\HDD\\dataset_dzkd_radar\\shot\\'
rename(path)
#copyfile("d:\\haha.txt", "f:\\")
