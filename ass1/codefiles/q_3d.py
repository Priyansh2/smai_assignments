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

def reduced_matrix(tfidf_matrix):
    U,s,V_transposed = np.linalg.svd(tfidf_matrix,full_matrices=False)
    S = np.diag(s)
    threshold=[0.95,0.9,0.85,0.8,0.75,0.7]
    sum_singular = float(np.trace(S))
    singuar_values = np.diagonal(S)
    r=[]
    for th in threshold:
        s=float(0)
        cnt=0
        for i in range(0,len(singuar_values)):
            s+=singuar_values[i]
            if s>=(th*sum_singular):
                cnt+1
                break
            else:
                cnt+=1
        reduced_dimension=cnt
        r.append(reduced_dimension)
    return r


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

def predict(x,classes,w):
    argmax=0
    predicted_label=classes[0]
    for cl in classes:
        F_cl = np.dot(x,w[cl])
        if F_cl>=argmax:
            argmax=F_cl
            predicted_label=cl
    return predicted_label

(m,k)=train_tfidf_matrix.shape
'''r=[]
r.append(train_tfidf_matrix.shape[0]/2)
r.append(test_tfidf_matrix.shape[0]/2)
r.append(train_tfidf_matrix.shape[0]-1)
r.append(test_tfidf_matrix.shape[0]-1)
U,S,V_T,r_dim = reduced_matrix(train_tfidf_matrix)
r.append(r_dim)
r.append(r_dim/2)'''
#r = sorted(r)
r = reduced_matrix(train_tfidf_matrix)
threshold=[0.95,0.9,0.85,0.8,0.75,0.7]
i=0
for reduced_dimension in r:
    xtrain = U[:,0:reduced_dimension].dot(S[0:reduced_dimension,0:reduced_dimension])
    train_V = V_T[0:reduced_dimension,:].transpose()
    xtest = test_tfidf_matrix.dot(train_V)
    accuracy = multiclass_perceptron(xtrain,xtest,ytrain,ytest,no_of_classes)
    print "threshold%,r_dim, Accuracy: ",threshold[i],reduced_dimension,accuracy
    i+=1