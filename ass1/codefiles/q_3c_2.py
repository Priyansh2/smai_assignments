#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import io
import operator
import codecs
import unicodedata
import glob
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



def cosine_similarity(xtrain,ytrain,xtest,ytest): ##here xtest and ytest are 2d arrays
    top_k=10
    (r_xtrain,c_xtrain) = xtrain.shape
    (r_xtest,c_xtest) = xtest.shape
    j=0
    cnt=0
    for test_doc in xtest:
        cosine_values=[]
        query_vec = test_doc
        for i in range(0,r_xtrain):
            cosine_similarity = float(np.dot(query_vec,xtrain[i]))/(float(np.linalg.norm(query_vec))*float(np.linalg.norm(xtrain[i])))
            cosine_values.append([cosine_similarity,int(ytrain[i][0])])

        cosine_values = sorted(cosine_values,key=lambda z: (z[0],z[1]),reverse=True)
        temp=[]
        for i in range(0,top_k):
            temp.append(cosine_values[i][1])

        label_count_dict =Counter(temp)
        sorted_label_counts = sorted(label_count_dict.items(), key=operator.itemgetter(1),reverse=True)
        predicted_label = int(sorted_label_counts[0][0])
        #if r_xtest==1:
            #return predicted_label
        #else:
        if predicted_label==int(ytest[j][0]):
            cnt+=1
        j+=1

    accuracy = float(cnt)/r_xtest
    return accuracy

train_tfidf_matrix,ytrain,vocabulary = generating_data(train_data_paths,None)
test_tfidf_matrix,ytest = generating_data(test_data_paths,vocabulary)
xtrain,train_V = get_reduced_matrix(train_tfidf_matrix)
xtest = test_tfidf_matrix.dot(train_V)
accuracy = cosine_similarity(xtrain,ytrain,xtest,ytest)
print "Accuracy: ",accuracy

