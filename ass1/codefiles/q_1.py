#!/usr/bin/env python
import os
import sys
import math
import numpy as np
from voted_perceptron import voted_perceptron
from vanilla_perceptron import vanilla_perceptron
import matplotlib.pyplot as plt
from matplotlib import interactive
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
script_path =os.path.dirname(os.path.realpath(__file__))
dataset_path=script_path+"/datasets/"
def working_with_perceptron(data,dataset,shuffle):
    accuracy=[]
    data=np.array(data,dtype=float)
    if shuffle==1:
        np.random.shuffle(data)
    (rows,cols) = data.shape
    counter=0
    folds=10
    testrows = rows/folds
    epochs_all=[10,15,20,25,30,35,40,45,50]
    epochs_all=np.array(epochs_all)
    arr_final_avg_voted=[0]*len(epochs_all)
    arr_final_avg_vanilla=[0]*len(epochs_all)
    for i in range(0,len(epochs_all)):
        epochs = epochs_all[i]
        final_avg_voted_y_count=0
        final_avg_y_count_vanilla=0
        ptr=0
        alpha=ptr
        beta=ptr+testrows
        testdata=data[alpha]
        for j in range(alpha+1,beta):
            temp_testdata=data[j]
            testdata=np.vstack((testdata,temp_testdata))
        ptr+=testrows
        alpha=ptr
        beta=rows
        traindata=data[alpha]
        for j in range(alpha+1,beta):
            temp_traindata=data[j]
            traindata=np.vstack((traindata,temp_traindata))
        (row_te,col_te) = testdata.shape
        (row_tr,col_tr) = traindata.shape
        transposed_traindata =  traindata.T
        y_transposed = transposed_traindata[col_tr-1]
        alpha = 0
        beta = col_tr-1
        x_transposed = transposed_traindata[0]
        for j in range(alpha+1,beta):
            temp_x_transposed = transposed_traindata[j]
            x_transposed = np.vstack((x_transposed,temp_x_transposed))
        x = x_transposed.T
        y = y_transposed[:,np.newaxis]
        final_avg_voted_y_count = voted_perceptron(x,y,epochs,testdata)
        final_avg_y_count_vanilla = vanilla_perceptron(x,y,epochs,testdata)
        for k in range(2,folds+1):
            alpha=ptr
            beta=ptr+testrows
            testdata=data[alpha]
            for j in range(alpha+1,beta):
                temp_testdata=data[j]
                testdata=np.vstack((testdata,temp_testdata))
            #print testdata
            ptr+=testrows
            alpha=0
            beta=(k-1)*testrows
            traindata1=data[alpha]
            for j in range(alpha+1,beta):
                temp_traindata1=data[j]
                traindata1=np.vstack((traindata1,temp_traindata1))

            alpha=ptr
            beta=rows
            traindata2=data[alpha]
            for j in range(alpha+1,beta):
                temp_traindata2=data[j]
                traindata2=np.vstack((traindata2,temp_traindata2))
            #print traindata1.shape
            #print traindata2.shape
            traindata = np.vstack((traindata1,traindata2))
            #print traindata.shape
            transposed_traindata = traindata.T
            y_transposed = transposed_traindata[traindata.shape[1]-1]
            alpha = 0
            beta = traindata.shape[1]-1
            x_transposed = transposed_traindata[0]
            for j in range(alpha+1,beta-alpha):
                temp_x_transposed = transposed_traindata[j]
                x_transposed = np.vstack((x_transposed,temp_x_transposed))
            x = x_transposed.T
            y = y_transposed[:,np.newaxis]
            final_avg_voted_y_count+= voted_perceptron(x,y,epochs,testdata)
            final_avg_y_count_vanilla+= vanilla_perceptron(x,y,epochs,testdata)

        final_avg_voted_y_count=float(final_avg_voted_y_count)/folds
        arr_final_avg_voted[counter]=final_avg_voted_y_count
        final_avg_y_count_vanilla=float(final_avg_y_count_vanilla)/folds
        arr_final_avg_vanilla[counter]=final_avg_y_count_vanilla
        counter+=1
    accuracy.append(arr_final_avg_voted)
    accuracy.append(arr_final_avg_vanilla)
    return accuracy


def load_dataset(shuffle,dataset=None):
    if dataset:
        ###load that particular dataset
        data=[]
        with open(dataset_path+dataset,'r') as f:
            for row in f.readlines():
                feature_vec=[]
                temp = row.split(",")
                for i in range(0,len(temp)):
                    feature_vec.append(int(temp[i]))
                data.append(feature_vec)
        return working_with_perceptron(data,dataset,shuffle)
    else:
        ###loading all dataset present in script dir
        accuracy_matrix=[]
        filenames=[]
        for dataset in  os.listdir(dataset_path):
            filenames.append(dataset)
            ##for each dataset do the following
        filenames = sorted(filenames)
        for dataset in filenames:
            data=[]
            #dataset="ionosphere.csv"
            #dataset="breast_cancer.csv"
            with open(dataset_path+dataset,'r') as f:
                for row in f.readlines():
                    feature_vec=[]
                    temp = row.split(",")
                    if dataset=="breast_cancer.csv":
                        for i in range(1,len(temp)-1):
                            feature_vec.append(float(temp[i]))
                        if int(temp[len(temp)-1])==2:
                            feature_vec.append(1)
                        else:
                            feature_vec.append(-1)
                        data.append(feature_vec)
                    elif dataset=="ionosphere.csv":
                        for i in range(0,len(temp)-1):
                            feature_vec.append(float(temp[i]))
                        item =temp[len(temp)-1].split("\n")[0]
                        if item=='g':
                            feature_vec.append(1)
                        else:
                            feature_vec.append(-1)
                        data.append(feature_vec)
            f.close()
            accuracy = working_with_perceptron(data,dataset,shuffle)
            accuracy_matrix.append(accuracy)
        return accuracy_matrix

if len(sys.argv)>1:
    is_shuffle=int(sys.argv[1])
    if is_shuffle!=1:
        print "Shuffle can be 0 or 1\n"

else:
    is_shuffle=0
accuracy_matrix = load_dataset(is_shuffle)
print "For breast_cancer.csv:- \n"
print "voted_perceptron:- "
print accuracy_matrix[0][0]
print "vanilla_perceptron:- "
print accuracy_matrix[0][1]
print "\n"

print "For ionosphere.csv:- \n"
print "voted_perceptron:- "
print accuracy_matrix[1][0]
print "vanilla_perceptron:- "
print accuracy_matrix[1][1]

epochs_all=[10,15,20,25,30,35,40,45,50]

plt.figure(1)
#plt.ylim([0.9,1])
plt.xlabel('No: of epochs')
plt.ylabel('Average accurcy over 10-folds')
plt.title('Plot for Breast-Cancer dataset')
plt.plot(epochs_all,accuracy_matrix[0][0],label='voted')
plt.plot(epochs_all,accuracy_matrix[0][1],label='vanilla')
plt.legend(loc='upper right')
interactive(True)
plt.show()

plt.figure(2)
plt.xlabel('No: of epochs')
plt.ylabel('Average accurcy over 10-folds')
plt.title('Plot for Ionosphere dataset')
plt.plot(epochs_all,accuracy_matrix[1][0],label='voted')
plt.plot(epochs_all,accuracy_matrix[1][1],label='vanilla')
plt.legend(loc='upper right')
interactive(False)
plt.show()
