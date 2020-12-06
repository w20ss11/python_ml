#coding=gbk
'''
Created on 2017��6��1��

@author: wss
'''
# -*- coding: utf-8 -*-
import scipy.signal as signal
import numpy as np
import pylab as pl

# ĳ�������˲����Ĳ���
a = np.array([1.0, -1.947463016918843, 0.9555873701383931])
b = np.array([0.9833716591860479, -1.947463016918843, 0.9722157109523452])

# 44.1kHz�� 1���Ƶ��ɨ�貨
t = np.arange(0, 0.5, 1/44100.0)
x= signal.chirp(t, f0=10, t1 = 0.5, f1=1000.0)

# ֱ��һ�μ����˲��������
y = signal.lfilter(b, a, x)

# �������źŷ�Ϊ50������һ��
x2 = x.reshape((-1,50))

# �˲����ĳ�ʼ״̬Ϊ0�� �������˲���ϵ������-1
z = np.zeros(max(len(a),len(b))-1, dtype=np.float)
y2 = [] # ����������б�

for tx in x2:
    # ��ÿ���źŽ����˲����������˲�����״̬z
    ty, z = signal.lfilter(b, a, tx, zi=z)
    # �������ӵ�����б���
    y2.append(ty)
    
# �����y2ת��Ϊһά����
y2 = np.array(y2)
y2 = y2.reshape((-1,))

# ���y��y2֮������
print(np.sum((y-y2)**2))

# ��ͼ
pl.plot(t, y, t, y2)
pl.show()