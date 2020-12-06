#coding = utf-8
import numpy as np
from numpy.ma.core import arange, mean, var
from numpy.lib.shape_base import tile
from numpy import random
from math import sqrt

def line2array(line_list,list_label):
    per_array = np.zeros([3,6])
    for index in arange(6):#index : 0~5
        for i in arange(3):#i : 0~3
            if line_list[index]==list_label[i][index]:
                per_array[i][index]+=1
    return per_array

list_label=[["浅白","蜷缩","浊响","清晰","凹陷","硬滑"],
            ["青绿","稍蜷","沉闷","稍糊","稍凹","软粘"],
            ["乌黑","硬挺","清脆","模糊","平坦",""]]

f = open("watermelon3.0.txt")
res0=np.zeros([3,6])#前六列 存放为"是"的结果矩阵和
res1=np.zeros([3,6])#前六列 存放为"否"的结果矩阵和
val0=np.zeros([2,9])#后两列 否： 第一行为密度 第二行为含糖率
val1=np.zeros([2,8])#后两列 是： 第一行为密度 第二行为含糖率

n = 0
for line in f.readlines():
    strs=line.split(" ")
    if "是" in strs[-1]:
        res1+=np.array(line2array(strs, list_label))
        val1[0][n]=strs[-3]
        val1[1][n]=strs[-2]
    else:
        res0+=np.array(line2array(strs, list_label))
        val0[0][n-8]=strs[-3]
        val0[1][n-8]=strs[-2]
    n=n+1

print("res1:")
print(res1)
print("res0:")
print(res0)
  
add_1 = np.hstack(((np.ones([3,5])*3),np.ones([3,1])*2))#拉普拉斯正则化
add_2 = np.ones([3,6])

diff_0 = (res0+add_2)/(tile(res0.sum(axis=0),(3,1))+add_1)
diff_1 = (res1+add_2)/(tile(res1.sum(axis=0),(3,1))+add_1)
print("diff_0:")
print(diff_0)
print("diff_1:")
print(diff_1)
#--------------------↑前六列训练数据----------------------
list_test = ["青绿","蜷缩","浊响","清晰","凹陷","硬滑"]
test_1 = line2array(list_test, list_label)
print("test:")
print(test_1)
#--------------------↑前六列测试数据----------------------
p_0_pre = diff_0*test_1
print("前六列为'否'各概率:")
print(p_0_pre)
print("前六列为'是'各概率:")
p_1_pre = diff_1*test_1
print(p_1_pre)
#--------------------↑前六列概率计算----------------------
print("val0： （存放'否'类中后两列密度和含糖率的数据）")
print(val0)
print("val1： （存放'是'类中后两列密度和含糖率的数据）")
print(val1)
mean0 = np.array([mean(val0[0]),mean(val0[1])])#两个数 分别为0(否)类 密度和含糖率的均值
mean1 = np.array([mean(val1[0]),mean(val1[1])])#两个数 分别为1(是)类 密度和含糖率的均值
var0 = np.array([var(val0[0]),var(val0[1])])#两个数 分别为0(否)类 密度和含糖率的方差
var1 = np.array([var(val1[0]),var(val1[1])])#两个数 分别为1(是)类 密度和含糖率的方差
#--------------------↑后两列均值方差计算----------------------
def gaussian(mean,var,x):
    res = 1/sqrt(2*3.14*var)*np.exp(-(mean-x)**2/2*var)
    return res
p_0_pro = gaussian(mean0[0], var0[0], 0.697)+gaussian(mean0[1], var0[1], 0.460)
print(p_0_pro)
p_1_pro = gaussian(mean1[0], var1[0], 0.697)+gaussian(mean1[1], var1[1], 0.460)
print(p_1_pro)
#--------------------↑后两列概率计算----------------------
p_0 = np.sum(p_0_pre)+p_0_pro
p_1 = np.sum(p_1_pre)+p_1_pro
print("p_0:",p_0,"   p_1:",p_1)