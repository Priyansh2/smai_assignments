from ffn import ffn
import numpy as np
import csv
import math
def readDataBCD(filename):
    modifiedData = []
    with open(filename,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if '?' not in row:
                modifiedData.append(row)
    modifiedData.pop()
    npData = np.array(modifiedData)
    npData = npData.astype(np.float)
    return npData

def changeClassLabel(npData):
    dim = npData.shape[1]
    for i in range(npData.shape[0]):
        npData[i][dim-1] -= 1
    return npData
def remOtClasses(npData,noofCl):
    modifiedData = []
    dim = npData.shape[1]
    for i in range(npData.shape[0]):
        if npData[i][dim-1]<noofCl:
            modifiedData.append(npData[i].tolist())
    npData = np.array(modifiedData)
    return npData

def proLabvec(npData,noofCl):
    modifiedLabel = []
    dim = npData.shape[1]
    for i in range(npData.shape[0]):
        x = [0 for j in range(noofCl)]
        x[int(npData[i][dim-1])]=1.0
        modifiedLabel.append(x)
    npLabel = np.array(modifiedLabel)
    return npLabel

def crossValidate(data,label,k,epochs,noofCl,noofHi):
    seS = math.floor(data.shape[0]/float(k))
    # print seS
    # print data.shape[0]
    actualAcurracy = 0.0
    for x in range(k): # part which is used as training set
        listd = []
        tlistd = []
        listt = []
        tlistt = []
        for i in range(data.shape[0]):
            if min(k-1,math.floor(i/seS))==x:
                listt.append(data[i])
                tlistt.append(label[i])
            else:
                listd.append(data[i])
                tlistd.append(label[i])
        listt = np.array(listt)
        tlistt = np.array(tlistt)
        listd = np.array(listd)
        tlistd = np.array(tlistd)
        nnet = ffn(data.shape[1],noofHi,noofCl,'sigmoid',0.01,listd,tlistd,listt,tlistt)
        nnet.addNW()
        nnet.etrain(epochs)
        actualAcurracy+=nnet.test()
    return actualAcurracy/k

def removeLabel(x):
    lasti = x.shape[1]-1
    xR = np.delete(x,lasti,1)
    xR = xR.astype(np.float)
    return xR
def pdigit():
    data = readDataBCD('./data/pendigits.tra')
    data = remOtClasses(data,4)
    label = proLabvec(data,4)
    data = removeLabel(data)
    tdata = readDataBCD('./data/pendigits.tes')
    tdata = remOtClasses(tdata,4)
    tlabel = proLabvec(tdata,4)
    tdata = removeLabel(tdata)
    # print label
    # print tlabel
    nnet = ffn(data.shape[1],30,4,'tanh',0.01,data,label,tdata,tlabel)
    nnet.addNW()
    # print data[0],label[0]
    # nnet.forwardPass(data[0],label[0])
    nnet.etrain(30)
    print nnet.test()
    # print nnet.wI
    # print tdata[0]
    # print tlabel[0]
    # print tdata[2]
    # print tlabel[2]
    # # print data[0],label[0]
    # # print label
    # nnet.forwardPass(tdata[0],tlabel[0])
    # print nnet.O[1].yVal
    # nnet.forwardPass(tdata[2],tlabel[2])
    # print nnet.O[1].yVal
def dermatology():
    data = readDataBCD('./data/dermatology.data')
    data = changeClassLabel(data)
    data = remOtClasses(data,3)
    label = proLabvec(data,3)
    data = removeLabel(data)
    # tdata = readDataBCD('./data/dermatology.data')
    # tdata = changeClassLabel(tdata)
    # tdata = remOtClasses(tdata,3)
    # tlabel = proLabvec(tdata,3)
    # tdata = removeLabel(tdata)
    # print data,label
    # print tdata,tlabel
    # nnet = ffn(data.shape[1],30,3,'sigmoid',0.01,data,label,tdata,tlabel)
    # nnet.addNW()
    # nnet.etrain(5)
    # print nnet.test()
    # print nnet.wI
    # print tdata[0]
    # print tlabel[0]
    # print tdata[2]
    # print tlabel[2]
    # # print data[0],label[0]
    # # print label
    # nnet.forwardPass(tdata[0],tlabel[0])
    # print nnet.O[1].yVal
    # nnet.forwardPass(tdata[2],tlabel[2])
    # print nnet.O[1].yVal
    print crossValidate(data,label,5,10,3,30)
if __name__ == '__main__':
    # pdigit()
    dermatology()
