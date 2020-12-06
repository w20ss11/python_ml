#coding=gbk
'''
Created on 2017��6��2��

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
y = signal.lfilter(b, a, x)
ns = range(10, 1100, 100)
err = []

for n in ns:
    # ����������Ӧ
    impulse = np.zeros(n, dtype=np.float)
    impulse[0] = 1
    h = signal.lfilter(b, a, impulse)
    
    # ֱ��FIR�˲��������
    y2 = signal.lfilter(h, 1, x)
   
    # ���y��y2֮������
    err.append(np.sum((y-y2)**2))

# ��ͼ
pl.figure(figsize=(8,4))
pl.semilogy(ns , err, "-o")
pl.xlabel(u"������Ӧ����")
pl.ylabel(u"FIRģ��IIR�����")
pl.show()