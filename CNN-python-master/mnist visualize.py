# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle


def create_dir():
    train_path = 'training_data'
    vaild_path = 'vaildation_data'
    test_path = 'testing_data'

    for i in range(10):
        stri = str(i)
        if (os.path.exists(train_path + "\\" + stri) == False):
            os.makedirs(train_path + "\\" + stri)
        if (os.path.exists(vaild_path + "\\" + stri) == False):
            os.makedirs(vaild_path + "\\" + stri)
        if (os.path.exists(test_path + "\\" + stri) == False):
            os.makedirs(test_path + "\\" + stri)
    if (os.path.exists(train_path + "\\all") == False):
        os.makedirs(train_path + "\\all")
    if (os.path.exists(vaild_path + "\\all") == False):
        os.makedirs(vaild_path + "\\all")
    if (os.path.exists(test_path + "\\all") == False):
        os.makedirs(test_path + "\\all")


def decompression():
    f = open('data/mnist.pkl', 'rb')
    training_data, validation_data, testing_data = pickle.load(f, encoding='bytes')

    train_data = training_data[0]
    train_label = training_data[1]
    L = len(train_data)
    statistics = [0 for i in range(10)]
    print(L)
    for i in range(L):
        pic = train_data[i].reshape((28, 28))
        pic = pic * 255
        label = train_label[i]
        cv2.imwrite("training_data\\" + str(label) + "\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        cv2.imwrite("training_data\\all\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        statistics[int(label)] += 1
        if (i % 1000 == 0):
            print(str(L) + ":————>>>", str(i))

    valid_data = validation_data[0]
    valid_label = validation_data[1]
    L = len(valid_data)
    statistics = [0 for i in range(10)]
    print(L)
    for i in range(L):
        pic = valid_data[i].reshape((28, 28))
        pic = pic * 255
        label = valid_label[i]
        cv2.imwrite("vaildation_data\\" + str(label) + "\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        cv2.imwrite("vaildation_data\\all\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        statistics[int(label)] += 1
        if (i % 1000 == 0):
            print(str(L) + ":————>>>", str(i))

    test_data = testing_data[0]
    test_label = testing_data[1]
    L = len(test_data)
    statistics = [0 for i in range(10)]
    print(L)
    for i in range(L):
        pic = test_data[i].reshape((28, 28))
        pic = pic * 255
        label = test_label[i]
        cv2.imwrite("testing_data\\" + str(label) + "\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        cv2.imwrite("testing_data\\all\\" + str(label) + "_" + str(statistics[int(label)]) + ".png", pic)
        statistics[int(label)] += 1
        if (i % 1000 == 0):
            print(str(L) + ":————>>>", str(i))


create_dir()
decompression()
