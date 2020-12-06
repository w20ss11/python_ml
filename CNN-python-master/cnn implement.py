# author : mag lyx
# reference:
# mnist dataset download link : http://download.csdn.net/download/u010159842/9733877
import random
import numpy as np
import pickle
from time import clock

learn_rate = 0.0005  # learn rate
ni = 720  # dimension of FCNN input layer
nh = 100  # node count of FCNN hidden layer
no = 10  # dimension of FCNN output layer

CK = no  # number of different classes

wih = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=(ni, nh))  # weight matrix of input layer hidden layer
who = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=(nh, no))  # weight matrix of hidden layer tooutput layer
bih = np.random.random((1, nh))  # bias matrix of input layer hidden layer
bho = np.random.random((1, no))  # bias matrix of hidden layer tooutput layer

cmat_s = [28, 28]
kernel_c = 5
kernel_s = [kernel_c, 5, 5]
cout_s = [kernel_c, 24, 24]
pool_s = 2
pout_s = [kernel_c, 12, 12]

conv_kernel = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=kernel_s)
conv_bias = np.random.normal(loc=0, scale=1.0, size=(kernel_c,))


# matrix rotation 180
# def MatRot180(k):
#     kh, kw = np.shape(k)
#     r = [[k[kw - 1 - i,kh - 1 - j] for j in range(kw)] for i in range(kw)]
#     return np.array(r)

def MatRot180(k):
    k = np.rot90(k)
    return np.rot90(k)


# Convolution operation
# m:matrix
# conv_kernel : convoluteion kernel
def Convolution(m, k):
    mw, mh = m.shape
    cw, ch = k.shape
    newmat = [[np.sum(m[i:i + cw, j:j + ch] * k) for j in range(mw - cw + 1)] for i in range(mh - ch + 1)]
    return np.array(newmat)


# pooling (maxpooling)
def DownSampling(m, scale=2):
    w, h = m.shape
    rw, rh = int((w + 1) / 2), int((h + 1) / 2)
    r = [[np.max(m[i * scale:i * scale + scale, j * scale:j * scale + scale]) for j in range(rw)] for i in range(rh)]
    return np.array(r)


def UpSampling(m, scale=2):
    b = np.ones((scale, scale))
    return np.kron(m, b)


def Mat2Vec(m):
    m = m.reshape(1, (m.size))
    return m


def Vec2Mat(m):
    m = m.reshape(pout_s)
    return m


# m:matrix
# conv_kernel : convoluteion kernel
def RunConvLay(m):
    cout = [Convolution(m, MatRot180(k)) for k in conv_kernel]
    return np.array(cout)


def ReLU(m):
    m = np.maximum(m, 0.0)
    return m


def DReLU(m):
    m[m > 0.0] = 1
    return m


# FCNN
def RunFcLay(x):
    xij = x.dot(wih) + bih
    hout = np.maximum(0, xij)  # ReLU active function
    xjk = hout.dot(who) + bho
    exp_xjk = np.exp(xjk)  # softmax active function [partion]
    oout = exp_xjk / np.sum(exp_xjk, axis=1)  # softmax [partion]
    return hout, oout


# Run CNN
def RunCNN(m):
    cout = RunConvLay(m)  # convolution layer
    pout = np.array([DownSampling(cout[ind], pool_s) for ind in range(kernel_c)])  # pooling layer
    bout = np.array([pout[ind] + conv_bias[ind] for ind in range(kernel_c)])  # add bias
    rout = np.array([ReLU(bout[ind]) for ind in range(kernel_c)])  # ReLU
    iout = Mat2Vec(rout)  # matrix to vector
    hout, oout = RunFcLay(iout)  # FCNN
    return m, cout, pout, bout, rout, iout, hout, oout


# run CNN model to judge label
def UseCNNJudgeType(m):
    _, _, _, _, _, _, _, oout = RunCNN(m)
    return oout.argmax()


# update weight by BPNN algorithm
# cout : convolution layer output
# pout : pooling layer output
# bout : after add bias
# rout : ReLU function output
# iout : matrix to vector / FCNN input
# hout : FCNN hidden layer output
# oout : FCNN output layer output
# tarout : target output
def BackProp(m, cout, pout, bout, rout, iout, hout, oout, tarout):
    global wih, bih, who, bho, conv_bias, conv_kernel
    odelta = oout
    odelta[0, tarout] -= 1  # 1,10

    dwho = hout.T.dot(odelta)
    dbho = np.sum(odelta, axis=0, keepdims=True)

    dReLU = DReLU(hout)  # 1,100
    hdelta = odelta.dot(who.T) * dReLU  # 1,100
    dbih = np.sum(hdelta, axis=0)
    dwih = iout.T.dot(hdelta)

    vdelta = hdelta.dot(wih.T)  # 1,720

    wih -= learn_rate * dwih
    bih -= learn_rate * dbih
    who -= learn_rate * dwho
    bho -= learn_rate * dbho

    rdelta = Vec2Mat(vdelta)  # 5,12,12

    dReLU = np.array([DReLU(item) for item in rout])
    bdelta = rdelta * dReLU

    bdelta = bdelta.sum(axis=(1, 2))  # 5,
    cdelta = np.array([UpSampling(item) for item in rdelta])
    cdelta = cdelta * cout  # 5,24,24

    dconvbias = bdelta
    dconvkernel = np.array([MatRot180(Convolution(m, kernel)) for kernel in cdelta])  # 5,5,5

    conv_bias -= learn_rate * dconvbias
    conv_kernel -= learn_rate * dconvkernel


# get c numbers of [s,e] interval
def RandomList(s, e, c):
    list = range(s, e)
    slice = random.sample(list, c)
    return slice


# training CNN
# batchsize : SGD batch size
# times : iteration times
def Training(train_data, train_label, batchsize=500, times=10):
    for i in range(times):
        rl = RandomList(0, 50000, batchsize)
        for j in range(batchsize):
            ind = rl[j]
            xdata = train_data[ind]
            ylabel = train_label[ind]
            xdata = xdata.reshape(cmat_s)
            mat, cout, pout, bout, rout, iout, hout, oout = RunCNN(xdata)
            BackProp(mat, cout, pout, bout, rout, iout, hout, oout, ylabel)
            if (j % 100 == 0):
                print('training count: ' + str(times) + '/' + str(i) + '       training: ' + str(batchsize) + '/ ' + str(j))


# test CNN model
def Testing(testing_data, testing_label, test_count=1000):
    error_c = 0
    for i in range(test_count):
        xdata = testing_data[i]
        xdata = xdata.reshape(cmat_s)
        modeloutput = UseCNNJudgeType(xdata)
        if (modeloutput != testing_label[i]):
            error_c += 1
        if (i % 100 == 0):
            print('testing:  ', test_count, '/', i)
    print('error rate:', error_c / test_count, '\n')


# write weight matrix to file
# path : file path
def WriteWeight(path):
    file = open(path, 'w')
    wigstring = ""
    np.set_printoptions(threshold=np.NaN, precision=10, suppress=True)

    wigstring += 'conv_kernel\n'
    wigstring += str(conv_kernel)
    wigstring += '\n\n\n'

    wigstring += 'conv_bias\n'
    wigstring += str(conv_bias)
    wigstring += '\n\n\n'

    wigstring += 'wih\n'
    wigstring += str(wih)
    wigstring += '\n\n\n'

    wigstring += 'bih\n'
    wigstring += str(bih)
    wigstring += '\n\n\n'

    wigstring += 'who\n'
    wigstring += str(who)
    wigstring += '\n\n\n'

    wigstring += 'bho\n'
    wigstring += str(bho)
    wigstring += '\n\n\n'

    file.write(wigstring)
    file.close()


# file 'mnist.pkl'  download link : http://download.csdn.net/download/u010159842/9733877
f = open('data/mnist.pkl', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
train_data = training_data[0]  # training data
train_label = training_data[1]  # training label

testing_data = test_data[0]  # test data
testing_label = test_data[1]  # test label

start = clock()  # get current time

for i in range(10):
    Training(train_data, train_label)
    Testing(testing_data, testing_label)

print("test all: ")
Testing(testing_data, testing_label, 10000)

finish = clock()  # get current time

print('\n\n')
print('Elapsed Time: ', finish - start, 's')

WriteWeight('weight.txt')
