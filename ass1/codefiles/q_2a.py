#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)

def create_dataset():
    training_data = []
    training_data.append(np.array([3,3,1]))
    training_data.append(np.array([3,0,1]))
    training_data.append(np.array([2,1,1]))
    training_data.append(np.array([0,2,1]))
    training_data.append(np.array([-1,1,-1]))
    training_data.append(np.array([0,0,-1]))
    training_data.append(np.array([-1,-1,-1]))
    training_data.append(np.array([1,0,-1]))
    return training_data

#-----------------------Least square approach------------------------
def appending_one_at_end(data_matrix,classes,n):
    l = len(data_matrix)
    class1 = classes[0]
    class2= classes[1]
    for i in range(0,l):
        n=n-1
        k=len(data_matrix[i])
        temp=[]
        temp.append(data_matrix[i][0])
        for j in range(k-1,1,-1):
            temp.append(data_matrix[i][k-j])
        temp=np.array(temp)
        if n<0:
            data_matrix[i]=np.hstack((np.hstack((temp,1)),class2))
        else:
            data_matrix[i]=np.hstack((np.hstack((temp,1)),class1))
    return data_matrix

def least_sq_approach(data_matrix,classes,n):
    tr_data = appending_one_at_end(data_matrix,classes,n)
    tr_data=np.array(tr_data)
    (rows,cols) = tr_data.shape
    transposed_traindata = tr_data.T
    y_transposed = transposed_traindata[cols-1]
    x_transposed = transposed_traindata[0]
    for i in range(1,cols-1):
        temp_x_transposed = transposed_traindata[i]
        x_transposed = np.vstack((x_transposed,temp_x_transposed))
    x = x_transposed.T
    y = y_transposed[:][np.newaxis].T
    w = np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))
    return w

#-----------------------fisher code below------------------------
def fisher_LDA(training_data):
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    count_class1 = 0
    count_class2 = 0
    for data in training_data:
        if data[2] == 1:
            m1 += data[0:2]
            count_class1 += 1
        else:
            m2 += data[0:2]
            count_class2 += 1
    m1 = m1 / count_class1
    m2 = m2 / count_class2
    Sw = np.zeros((2,2))
    for data in training_data:
        if data[2] == 1:
            a = data[0:2]-m1
            aT = a[:,np.newaxis]
            Sw += np.multiply(a,aT)
        else:
            a = data[0:2]-m2
            aT = a[:,np.newaxis]
            Sw += np.multiply(a,aT)
    Wmatrix = np.dot(np.linalg.inv(Sw),(m1-m2))
    return Wmatrix,m1,m2



def main():
    training_data = create_dataset()
    classes=[]
    classes.append(1)
    classes.append(-1)
    l = len(training_data)
    w = least_sq_approach(training_data,classes,l/2)
    w = np.array([w[0][0],w[1][0],w[2][0]])
    w_vec = np.array([w[0],w[1]])
    #w_vec = w_vec/np.linalg.norm(w_vec) ###unit vector
    #bias = w[2]
    print "Least square approach w: ",w

    training_data = create_dataset()
    Wmatrix,m1,m2 = fisher_LDA(training_data)
    s=float(0)
    for data in training_data:
        s+=np.dot(Wmatrix,data[0:2])
    s=float(s/8)
    bias=s
    print "Fisher w and bias: ",Wmatrix,bias
    m =m1+m2
    #print m
    threshold = np.dot(Wmatrix,m)/2
    #print bias
    #Wmatrix = Wmatrix/np.linalg.norm(Wmatrix) ##unti vector
    #Wmatrix = np.array([Wmatrix[0],Wmatrix[1],bias])
    #Wmatrix= np.insert(Wmatrix,2,bias)

    ###projecting points linearly over fisher line
    projected_points=[]
    #tmp = np.array([float(0),Wmatrix[2]])
    for data in training_data:

        temp = np.dot(Wmatrix,data[0:2])
        #temp = np.dot(Wmatrix,np.hstack((data[0:2],1)))
        temp =  temp / np.dot(Wmatrix,Wmatrix)
        projected_points.append(Wmatrix* temp)

    #print projected_points
    new_points = []
    for i in range(len(training_data)):
        temp = projected_points[i][:]
        temp = np.hstack((temp,1))
        if training_data[i][2]==-1:
            temp = temp * -1
        new_points.append(temp)
    #print new_points
    #applying perceptron to the new points
    w_points = np.zeros(3)
    flag = True
    #print "lol"
    while(flag):
        flag = False
        for i in range(len(new_points)):
                result = np.dot(w_points,new_points[i])
                if result<=threshold:
                    w_points = w_points + new_points[i]
                    flag = True
    print "Perceptron w over FLD: ",w_points
    min_x1=-1
    max_x1=3
    min_x1+=-1
    max_x1+=1
    #####equation of fisher's line (y=mx+c where c is bais which we get from least square method)
    xcor=[min_x1,max_x1]
    #slope = Wmatrix[1]/Wmatrix[0]
    #constant = bias
    ycor=[(Wmatrix[1]*min_x1)/Wmatrix[0],(Wmatrix[1]*max_x1)/Wmatrix[0]]
    #ycor = [(slope*min_x1)+constant,(slope*max_x1)+constant]
    #print ycor
    x_cor=[] ### -2 to +4
    fisher_y_cor=[]
    leastsq_y_cor=[]
    for i in range(min_x1,max_x1+1):
        x_cor.append(i)
    for i in range(min_x1,max_x1+1):
        #temp=(-bias-(Wmatrix[0]*i))/Wmatrix[1]
        temp = (-w_points[2]-(w_points[0]*i))/w_points[1]
        fisher_y_cor.append(temp)
    for i in range(min_x1,max_x1+1):
        temp = (-w[2]-(w[0]*i))/w[1]
        leastsq_y_cor.append(temp)
    plt.plot(x_cor,leastsq_y_cor,label="Classifeir using Least Sq. approach")
    plt.plot(x_cor,fisher_y_cor,label="Classifier using FLD")
    plt.plot(xcor,ycor, label = "Fisher Projection Surface")
    #####plotting original and projected points
    i=0
    for data in training_data:
        if data[2] == 1:
            plt.plot(projected_points[i][0],projected_points[i][1], marker='x',fillstyle='none', markersize=6, color="red")
            plt.plot(data[0],data[1], marker='o', markersize=3, color="red")
            pyplot.plot([data[0], projected_points[i][0]], [data[1], projected_points[i][1]],'y--')
        else:
            plt.plot(data[0],data[1], marker='o', markersize=3, color="blue")
            plt.plot(projected_points[i][0],projected_points[i][1], marker='x',fillstyle='none', markersize=6, color="blue")
            pyplot.plot([data[0], projected_points[i][0]], [data[1], projected_points[i][1]],'y--')
        i+=1
    plt.legend(loc='best')
    plt.show()

main()
