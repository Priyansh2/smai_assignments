#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import io
import codecs
import operator
import unicodedata
import glob
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from q3_functions import generating_data,get_reduced_matrix,corpus_formation,stopwords_removal
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
reload(sys)
sys.setdefaultencoding('utf8')
delim = "/","~",'"',"'",":","`","!","…","?","|","-","(",")","()",";","<",">","<>",",",".","[","]","[]","–","—"
delim = list(delim)
script_path =os.path.dirname(os.path.realpath(__file__))
query_doc_path=sys.argv[2]
query_doc_label=int(sys.argv[3])
#data_path=script_path+"/q3data/"
#train_data_path=data_path+"train/"
train_data_path=sys.argv[1]
if train_data_path.split("/")[-1]!='':
    train_data_path+="/"
no_of_classes=len(next(os.walk(train_data_path))[1])
train_data_paths=[train_data_path+str(cl)+"/" for cl in next(os.walk(train_data_path))[1]]
train_data_paths=sorted(train_data_paths)

train_tfidf_matrix,ytrain,vocabulary = generating_data(train_data_paths,None)
xtrain,train_V = get_reduced_matrix(train_tfidf_matrix)
query_label=[]
query_label.append(query_doc_label)

train_text = [d[1] for d in stopwords_removal(corpus_formation(train_data_paths))]

def construct_query_vector(query_doc_path,vocabulary,train_text):
    textLinesFromFile=io.open(query_doc_path,encoding='utf-8').read()
    for i in range(0,len(delim)):
        textLinesFromFile = textLinesFromFile.replace(delim[i]," ")
    with io.open(query_doc_path,'w',encoding='utf-8') as fl:
        fl.write(textLinesFromFile)
    fl.close()
    with codecs.open(query_doc_path,'r','utf-8') as fl:
        string=''
        for line in fl:
            temp=line.split()
            for word in temp:
                string+=word+" "
    fl.close()
    query_text=[]
    query_text.append(string)
    stop = set(stopwords.words('english'))
    query_text[0]=" ".join(str(x) for x in [j for j in query_text[0].lower().split() if j not in stop])
    train_text.append(query_text[0])
    feature_vec=[float(0)]*len(vocabulary)
    querydoc_words_freq=Counter(query_text[0].split())
    i=0
    for word in vocabulary:
        if word in querydoc_words_freq:
            tf = querydoc_words_freq[word]
            cnt=0
            for doc in train_text:
                found=0
                for token in doc.split():
                    if token==word:
                        found=1
                        break
                if found==1:
                    cnt+=1
            idf=math.log((len(train_text)+1)/(cnt+1))+float(1)
            feature_vec[i]=float(tf*idf)
        i+=1

    feature_vec=np.array(feature_vec,dtype=float)
    l2_norm=np.linalg.norm(feature_vec)
    if l2_norm>float(0):
        feature_vec = feature_vec/l2_norm
    else:
        print "rare case"
    return feature_vec

query_vec = construct_query_vector(query_doc_path,vocabulary,train_text)
query_vec=query_vec[:,np.newaxis].T
reduced_query_vec = query_vec.dot(train_V)
query_label=np.array(query_label)
query_label=query_label[:,np.newaxis]



def cosine_similarity(xtrain,ytrain,xtest): ##here xtest and ytest are 2d arrays
    top_k=10
    (r_xtrain,c_xtrain) = xtrain.shape
    (r_xtest,c_xtest) = xtest.shape
    j=0
    cnt=0
    cosine_values=[]
    query_vec = xtest[0]
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
    return predicted_label

predicted_label = cosine_similarity(xtrain,ytrain,reduced_query_vec)
print "ypred :",predicted_label
print "yactual :",query_label[0][0]