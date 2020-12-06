#conding="gbk"
'''
Created on 

@author: wss
'''
# -*- coding: utf-8 -*- 

from svm import *
from svmutil import *


"""y, x = [1,-1], [{1:1, 2:1}, {1:-1,2:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
model = svm_train(prob, param)
#svm_save_model("model111", model)
yt = [1]
xt = [{1:1, 2:1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)
print('----------------------')"""

y, x = svm_read_problem('D:\\eclipse_workspace\\py_base\\src\\regression_demo\\train.txt')
#yt, xt = svm_read_problem('D:\eclipse_workspace\py_base\src\svm_demo\test1.txt')
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
model = svm_train(prob, param)
svm_save_model("model_linear", model)

print('------------------------------')
y, x = svm_read_problem('D:\\eclipse_workspace\\py_base\\src\\regression_demo\\train.txt')
#yt, xt = svm_read_problem('D:\eclipse_workspace\py_base\src\svm_demo\test1.txt')
prob  = svm_problem(y, x)
param = svm_parameter('-t 2 -c 4 -b 1')
model = svm_train(prob, param)
print('model:',model)
svm_save_model("model_rbf", model)

print('------------------------------')
y, x = svm_read_problem('D:\\eclipse_workspace\\py_base\\src\\regression_demo\\train.txt')
#yt, xt = svm_read_problem('D:\eclipse_workspace\py_base\src\svm_demo\test1.txt')
prob  = svm_problem(y, x)
param = svm_parameter('-t 3 -c 4 -b 1')
model = svm_train(prob, param)
print('model:',model)
svm_save_model("model_Sigmoid", model)
