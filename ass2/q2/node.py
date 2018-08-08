import numpy as np
from acFun import sigmoid,dsigmoid,tanh,dtanh
import math

class inputNode(object):
    def __init__(self):
        self.inVal = 0.0
        self.yVal = 0.0
    def setinVal(self,val):
        self.inVal = val
    def getY(self):
        self.yVal = self.inVal

class hiddenNode(object):
    def __init__(self,acf):
        self.inVal = 0.0
        self.yVal = 0.0
        self.acf = acf
        self.delv = 0.0
    def setinVal(self,val):
        self.inVal = val
    def addinVal(self,addVal):
        self.inVal += addVal
    def getY(self):
        if self.acf == 'sigmoid':
            self.yVal=sigmoid(self.inVal)
        else:
            self.yVal=tanh(self.inVal)
    def dac(self,val):
        if self.acf=='sigmoid':
            return dsigmoid(val)
        else:
            return dtanh(val)
class outputNode(object):
    def __init__(self,acf):
        self.inVal = 0.0
        self.yVal = 0.0
        self.acf = acf
        self.delv = 0.0
        self.acVal = 0.0
    def setinVal(self,val):
        self.inVal = val
    def addinVal(self,addVal):
        self.inVal += addVal
    def getY(self):
        if self.acf == 'sigmoid':
            self.yVal=sigmoid(self.inVal)
        else:
            self.yVal=tanh(self.inVal)
    def dac(self,val):
        if self.acf=='sigmoid':
            return dsigmoid(val)
        else:
            return dtanh(val)
    def getError(self):
        return -1*self.acVal*math.log(self.yVal)
