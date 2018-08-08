#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import io
import codecs
import unicodedata
import glob
import operator
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from q3_functions import generating_data,get_reduced_matrix
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
reload(sys)
sys.setdefaultencoding('utf8')
delim = "/","~",'"',"'",":","`","!","…","?","|","-","(",")","()",";","<",">","<>",",",".","[","]","[]","–","—"
delim = list(delim)
script_path =os.path.dirname(os.path.realpath(__file__))
#data_path=script_path+"/q3data/"
#train_data_path=data_path+"train/"
#test_data_path=data_path+"test/"
#no_of_classes=5
#train_data_paths=[]
#test_data_paths=[]
#for i in range(no_of_classes):
    #train_data_paths.append(train_data_path+str(i)+"/")
    #test_data_paths.append(test_data_path+str(i)+"/")
train_data_path=sys.argv[1]
if train_data_path.split("/")[-1]!='':
    train_data_path+="/"
test_data_path=sys.argv[2]
if test_data_path.split("/")[-1]!='':
    test_data_path+="/"
no_of_classes=len(next(os.walk(train_data_path))[1])
train_data_paths=[train_data_path+str(cl)+"/" for cl in next(os.walk(train_data_path))[1]]
test_data_paths=[test_data_path+str(cl)+"/" for cl in next(os.walk(test_data_path))[1]]
train_data_paths=sorted(train_data_paths)
test_data_paths=sorted(test_data_paths)


train_tfidf_matrix,ytrain,vocabulary = generating_data(train_data_paths,None)
test_tfidf_matrix,ytest = generating_data(test_data_paths,vocabulary)
xtrain,train_V = get_reduced_matrix(train_tfidf_matrix)
xtest = test_tfidf_matrix.dot(train_V)

def data_matrix_backup(data_matrix,filename):
    fd = codecs.open(script_path+filename,'w','utf-8')
    for doc_vec in data_matrix:
        string=''
        for val in doc_vec:
            string+=str(val)+" "
        string+="\n"
        fd.write(string)
    fd.close()

def loading_data_matrix(filename):
    data_matrix=[]
    with codecs.open(script_path+filename,'r','utf-8')  as fl:
        for line in fl.readlines():
            temp=line.split()
            row_vec=[]
            for val in temp:
                if val!="\n":
                    row_vec.append(float(val))
            data_matrix.append(row_vec)
    fl.close()
    return np.array(data_matrix)

xtrain_file="/xtrain_bak.txt"
ytrain_file="/ytrain_bak.txt"
xtest_file="/xtest_bak.txt"
ytest_file="/ytest_bak.txt"
'''data_matrix_backup(xtrain,xtrain_file)
data_matrix_backup(ytrain,ytrain_file)
data_matrix_backup(xtest,xtest_file)
data_matrix_backup(ytest,ytest_file)'''
'''xtrain = loading_data_matrix(xtrain_file)
ytrain = loading_data_matrix(ytrain_file)
xtest = loading_data_matrix(xtest_file)
ytest = loading_data_matrix(ytest_file)'''
#print xtrain.shape
#print xtest.shape
#print ytrain.shape
#print ytest.shape

def predict_y(w,c,x,k,classes):
    argmax=0
    predicted_class=classes[0]
    for cl in classes:
        summ=0
        for i in range(k[cl]):
            temp = np.dot(w[cl][i],x)
            summ+=c[cl][i]*temp

        F_cl = summ
        if F_cl>=argmax:
            argmax=F_cl
            predicted_class=cl
    return predicted_class


def multiclass_votedperceptron(xtrain,ytrain,xtest,ytest,no_of_classes):
    (m,k) = xtrain.shape
    epochs=100
    classes=[str(i) for i in range(no_of_classes)]
    w = {cl:[np.array([0 for i in range(k+1)],dtype=float)] for cl in classes}
    c = {cl:[1] for cl in classes}
    n = {cl:0 for cl in classes}
    for i in range(epochs):
        for j in range(m):
            xtrain_vec = np.hstack((xtrain[j],1))
            actual_label = str(int(ytrain[j][0]))
            argmax=0
            predicted_class=classes[0]
            for cl in classes:
                F_cl = np.dot(w[cl][n[cl]],xtrain_vec)
                if F_cl>=argmax:
                    argmax=F_cl
                    predicted_class=cl
            if predicted_class!=actual_label:
                n[actual_label]+=1
                w[actual_label].append(w[actual_label][n[actual_label]-1]+xtrain_vec)
                n[predicted_class]+=1
                w[predicted_class].append(w[predicted_class][n[predicted_class]-1]-xtrain_vec)
                c[actual_label].append(1)
                c[predicted_class].append(1)
            else:
                c[actual_label][n[actual_label]]+=1
    count=0
    (test_rows,test_cols)=xtest.shape
    w = {cl:np.array(w[cl]) for cl in classes}
    c = {cl:np.array(c[cl]) for cl in classes}
    for i in range(test_rows):
        xtest_vec=np.hstack((xtest[i],1))
        actual_label=str(int(ytest[i][0]))
        predicted_class = predict_y(w,c,xtest_vec,n,classes)
        if predicted_class==actual_label:
            count+=1
    return float(count)/test_rows

def predict(x,classes,w):
    argmax=0
    predicted_label=classes[0]
    for cl in classes:
        F_cl = np.dot(x,w[cl])
        if F_cl>=argmax:
            argmax=F_cl
            predicted_label=cl
    return predicted_label


def multiclass_perceptron(xtrain,xtest,ytrain,ytest,no_of_classes):
    (m,k)=xtrain.shape
    epochs=100
    classes=[str(i) for i in range(no_of_classes)]
    w = {cl:np.array([0 for i in range(k+1)],dtype=float) for cl in classes}
    for i in range(epochs):
        for j in range(m):
            xtrain_vec = np.hstack((xtrain[j],1))
            actual_label =str(int(ytrain[j][0]))
            argmax=0
            predicted_label=classes[0]
            for cl in classes:
                F_cl = np.dot(xtrain_vec,w[cl])
                if F_cl>=argmax:
                    argmax=F_cl
                    predicted_label=cl
            if not(actual_label==predicted_label):
                w[actual_label]+=xtrain_vec
                w[predicted_label]-=xtrain_vec

    (m1,k1) = xtest.shape
    count=0
    for i in range(m1):
        xtest_vec = np.hstack((xtest[i],1))
        actual_label=str(int(ytest[i][0]))
        predicted_label = predict(xtest_vec,classes,w)
        if actual_label==predicted_label:
            count+=1
    return float(count)/m1
#accuracy = multiclass_votedperceptron(xtrain,xtest,ytrain,ytest,no_of_classes)
accuracy = multiclass_perceptron(xtrain,xtest,ytrain,ytest,no_of_classes)
print "Accuracy: ",accuracy

