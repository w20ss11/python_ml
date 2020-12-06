#coding=gbk
'''
Created on 2017��4��14��

@author: wss
'''
"""ʹ��leastsq()�Դ����������Ҳ����ݽ�����ϡ�������õ��Ĳ�����Ȼ��ʵ�ʵĲ����п�����ȫ��ͬ�������������Һ������������ԣ�ʵ������ϵĽ����ʵ�ʵĺ�����һ�µġ�"""
import numpy as np
from scipy.optimize import leastsq

def func(x, p): 
    """����������õĺ���: A*sin(2*pi*k*x + theta)"""
    A, k, theta = p
    return A*np.sin(2*np.pi*k*x+theta)

def residuals(p, y, x): 
    """ʵ������x, y����Ϻ���֮��ĲpΪ�����Ҫ�ҵ���ϵ��"""
    return y - func(x, p)

x = np.linspace(-2*np.pi, 0, 100)
A, k, theta = 10, 0.34, np.pi/6 # ��ʵ���ݵĺ�������
y0 = func(x, [A, k, theta]) # ��ʵ����
# ��������֮���ʵ������
y1 = y0 + 2 * np.random.randn(len(x)) 

p0 = [7, 0.2, 0] # ��һ�β²�ĺ�����ϲ���

# ����leastsq�����������, residualsΪ�������ĺ���
# p0Ϊ��ϲ����ĳ�ʼֵ,# argsΪ��Ҫ��ϵ�ʵ������
plsq = leastsq(residuals, p0, args=(y1, x))
# ���˳�ʼֵ֮�⣬��������args����������ָ��residuals��ʹ�õ�������������ֱ�����ʱֱ��ʹ����X,Y��ȫ�ֱ�����,ͬ��Ҳ����һ��Ԫ�飬��һ��Ԫ��Ϊ��Ϻ�Ĳ������飻
# ���ｫ (y1, x)���ݸ�args������Leastsq()�Ὣ����������Ĳ������ݸ�residuals()�����residuals()������������p�����Һ����Ĳ�����y��x�Ǳ�ʾʵ�����ݵ����顣

print(u"��ʵ����:", [A, k, theta])
print(u"��ϲ���", plsq[0]) # ʵ��������Ϻ�Ĳ���

import pylab as pl
pl.plot(x, y0, label=u"��ʵ����")
pl.plot(x, y1, label=u"��������ʵ������")
pl.plot(x, func(x, plsq[0]), label=u"�������")
pl.legend()
pl.show()
