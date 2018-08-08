import os
import sys
import numpy as np
import math
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
def sign(a):
    if a>0:
        return 1
    else:
        return -1

def predict_y(w,b,c,x,k):
    summ=0
    for i in range(k):
        temp = np.dot(w[i],x)+b[i]
        summ+=c[i]*sign(temp)
    if sign(summ)>0:
        y=1
    else:
        y=-1
    return y

def voted_perceptron(x,y,epochs,testdata):
    #print x
    #print y
    #print testdata
    (m,k)=x.shape
    #print m
    w=[]
    b=[]
    c=[]
    n=0
    w.append(np.array([0]*k,dtype=float))
    b.append(float(0))
    c.append(1)
    for i in range(0,epochs):
        for j in range(0,m):
            #print x[j].shape
            dot = np.dot(w[n],x[j])
            #print dot
            temp = y[j][0]*(dot+b[n])
            if temp<=float(0):
                n+=1
                w.append(w[n-1]+(y[j][0]*x[j]))
                b.append(b[n-1]+y[j][0])
                c.append(1)
            else:
                c[n]+=1

    w =np.array(w,dtype=float)
    b = np.array(b,dtype=float)
    c=np.array(c)
    x_testvector = np.array([0]*k,dtype=float)
    [testdatarows,testdatacols] = testdata.shape
    y_count=0
    for i in range(0,testdatarows):
        for j in range(0,k):
            x_testvector[j]=testdata[i][j]
        ycap = predict_y(w,b,c,x_testvector,n)
        if sign(ycap)==sign(testdata[i][k]):
            y_count+=1

    avg_voted_y_count = float(y_count)/testdatarows
    return avg_voted_y_count

'''testdata = [[ 5    , 1 ,    1  ,   1   ,  2   ,  1    , 3   ,  1   ,  1 ,    2],
     [5   ,  4   ,  4    , 5  ,   7   , 10  ,   3   ,  2 ,    1  ,   2],
     [3 ,    1 ,    1   ,  1   ,  2    , 2 ,    3 ,    1,     1   ,  2 ]]
x=[ [5 ,    1     ,3,     1    , 2,     1    , 2  ,   1     ,1],
     [6 ,   10     ,2,     8    ,10,     2    , 7  ,   8    ,10],
     [1  ,   3     ,3 ,    2     ,2 ,    1     ,7   ,  2    , 1],
     [9   ,  4    , 5  ,  10    , 6  ,  10    , 4    , 8    , 1],
    [10    , 6   ,  4   ,  1   ,  3   ,  4   ,  3     ,2   ,  3],
     [1    , 1  ,   2    , 1  ,   2    , 2  ,   4    , 2  ,   1],
     [1    , 1 ,    4    , 1 ,    2     ,1 ,    2    , 1 ,    1],
     [5     ,3,     1    , 2,     2     ,1,     2    , 1,     1
]]
y=[ [2],
    [4],
    [2],
    [4],
    [4],
    [2],
    [2],
    [2]]
ans = voted_perceptron(np.array(x),np.array(y),10,np.array(testdata))
print ans'''
