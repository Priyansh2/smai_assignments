import numpy as np
import acFun
import node
import math

class ffn(object):
    def __init__(self,noofIn,noofHi,noofOu,acf,lR,data,label,testData,testLabel):
        self.noofIn = noofIn
        self.noofHi = noofHi
        self.noofOu = noofOu
        self.acf = acf
        self.data = data
        self.label = label
        self.testData = testData
        self.testLabel = testLabel
        self.wI = []
        self.wH = []
        self.I = []
        self.H = []
        self.O = []
        self.lR = lR
    def addNW(self):
        for i in range(self.noofIn):
            tmp = node.inputNode()
            self.I.append(tmp)
        for i in range(self.noofHi):
            tmp = node.hiddenNode(self.acf)
            self.H.append(tmp)
        for i in range(self.noofOu):
            tmp = node.outputNode(self.acf)
            self.O.append(tmp)
        self.wI = np.random.normal(0,1,(self.noofIn,self.noofHi))
        self.wH = np.random.normal(0,1,(self.noofHi,self.noofOu))
    def reset(self):
        for i in range(self.noofIn):
            self.I[i].inVal=0.0
        for i in range(self.noofHi):
            self.H[i].inVal=0.0
            self.H[i].delv=0.0
        for i in range(self.noofOu):
            self.O[i].inVal=0.0
            self.O[i].delv=0.0
    def forwardPass(self,datai,labeli):
        self.reset()
        for i in range(self.noofIn):
            self.I[i].setinVal(datai[i])
            self.I[i].getY()
        for i in range(self.noofIn):
            for j in range(self.noofHi):
                vta = self.wI[i][j]*self.I[i].yVal
                self.H[j].addinVal(vta)
        for j in range(self.noofHi):
            self.H[j].getY()
        for i in range(self.noofHi):
            for j in range(self.noofOu):
                vta = self.wH[i][j]*self.H[i].yVal
                self.O[j].addinVal(vta)
        iVec = []
        for j in range(self.noofOu):
            # self.O[j].getY()
            iVec.append(self.O[j].inVal)
            self.O[j].acVal = labeli[j]
        oVec = acFun.softmax(iVec)
        for j in range(self.noofOu):
            self.O[j].yVal =oVec[j]

    def backwardPass(self,labeli):
        for i in range(self.noofOu):
            self.O[i].acVal = labeli[i]
            self.O[i].delv = (self.O[i].acVal-self.O[i].yVal)
        for i in range(self.noofHi):
            pfac = self.H[i].dac(self.H[i].inVal)
            for j in range(self.noofOu):
                self.H[i].delv += pfac*self.wH[i][j]*self.O[j].delv

        for i in range(self.noofIn):
            for j in range(self.noofHi):
                self.wI[i][j] += self.lR*self.H[j].delv*self.I[i].yVal
        for i in range(self.noofHi):
            for j in range(self.noofOu):
                self.wH[i][j] += self.lR*self.O[j].delv*self.H[i].yVal

    def totErr(self):
        ret = 0.0
        for i in range(self.noofOu):
            ret += self.O[i].acVal*math.log(self.O[i].yVal)
        return -1.0*ret

    def etrain(self,epochs):
        lend = self.data.shape[0]
        for i in range(epochs):
            for j in range(lend):
                sn = j
                self.forwardPass(self.data[sn],self.label[sn])
                self.backwardPass(self.label[sn])

    def errorTrain(self):
        print "I will do you later"
    def predictedClass(self,idata,iLabel):
        self.forwardPass(idata,iLabel)
        maxi = -100.0
        classi = None
        for i in range(self.noofOu):
            # print self.O[i].yVal,
            if self.O[i].yVal>maxi:
                maxi = self.O[i].yVal
                classi = i
        return classi
    def getClassfl(self,iLabel):
        ret = -1.0
        for i in range(self.noofOu):
            if iLabel[i]==1.0:
                ret = i
        return ret
    def test(self):
        corClin = 0.0
        for i in range(self.testData.shape[0]):
            # print self.testData[i]
            pCla = self.predictedClass(self.testData[i],self.testLabel[i])
            aCla = self.getClassfl(self.testLabel[i])
            print pCla,aCla
            if pCla==aCla:
                corClin += 1.0
        return corClin/float(self.testData.shape[0])

if __name__ == '__main__':
    print  "awesome"
    tt = ffn(2,2,3,'sigmoid',0.01,np.array([3.0,4.0]),np.array([0.0,1.0,0.0]),np.array([3.0,4.0]),np.array([1.0,0.0,0.0]))
    tt.addNW()
    print tt.wI
    print tt.wH
    tt.forwardPass(tt.data,tt.label)
    print tt.O[0].yVal
    print tt.O[1].yVal
    print tt.O[2].yVal
    for i in range(500):
        tt.forwardPass(tt.data,tt.label)
        # print tt.totErr()
        tt.backwardPass(tt.label)
    print tt.O[0].yVal
    print tt.O[1].yVal

    print tt.wI
    print tt.wH
