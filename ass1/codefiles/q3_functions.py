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
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
reload(sys)
sys.setdefaultencoding('utf8')
delim = "/","~",'"',"'",":","`","!","…","?","|","-","(",")","()",";","<",">","<>",",",".","[","]","[]","–","—"
delim = list(delim)
'''script_path =os.path.dirname(os.path.realpath(__file__))
data_path=script_path+"/q3data/"
train_data_path=data_path+"train/"
test_data_path=data_path+"test/"
no_of_classes=5
train_data_paths=[]
test_data_paths=[]
for i in range(no_of_classes):
    train_data_paths.append(train_data_path+str(i)+"/")
    test_data_paths.append(test_data_path+str(i)+"/")
'''

def removing_punctuations_marks(data_paths):
    for files_path in data_paths:
        for filename in os.listdir(files_path):
            if os.stat(files_path+filename).st_size!=0:
                infilename=os.path.join(files_path,filename)
                textLinesFromFile=io.open(infilename,encoding='utf-8').read()
                for i in range(0,len(delim)):
                    textLinesFromFile = textLinesFromFile.replace(delim[i]," ")
                with io.open(infilename,'w',encoding='utf-8') as fl:
                    fl.write(textLinesFromFile)
                fl.close()
            else:
                print "Not a file"
                continue

def corpus_formation(data_paths):
    corpus=[]
    for data_path in data_paths:
        category = int(data_path.split("/")[8])
        filenames=[]
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames=sorted(filenames)
        for filename in filenames:
            with codecs.open(data_path+filename,'r','utf-8') as fl:
                string=''
                for line in fl:
                    temp=line.split()
                    for word in temp:
                        string+=word+" "
            fl.close()
            temp=[]
            temp.append(category)
            temp.append(string)
            corpus.append(temp)
    return corpus

def stopwords_removal(corpus):
    stop = set(stopwords.words('english'))
    for i in range(0,len(corpus)):
        corpus[i][1]=" ".join(str(x) for x in [j for j in corpus[i][1].lower().split() if j not in stop])
    return corpus


def create_vocabulary(corpus_data):
    vocabulary=[]
    for data in corpus_data:
        temp=data.split()
        for word in temp:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

def freq_cnts_of_corpus_words(corpus_data):
    corpus_words_freq_cnts=[]
    for doc in corpus_data:
        temp = doc.split()
        corpus_words_freq_cnts.append(Counter(temp))
    return corpus_words_freq_cnts

def tf_score(word,document_no,corpus_data,corpus_words_freq_cnts):
    my_word=word
    doc_words_freq_cnts = corpus_words_freq_cnts[document_no]
    total = len(corpus_data[document_no])
    if my_word in doc_words_freq_cnts:
        wordcount=int(doc_words_freq_cnts[my_word])
    tf = float(wordcount)/total
    return tf

def word_docid_mapping(vocabulary,corpus_data):
    word_docid_mapping={}
    for word in vocabulary:
        corpus_id=0
        docs_id=[]
        for doc in corpus_data:
            found=0
            temp = doc.split()
            for token in temp:
                if word==token:
                    found=1
                    break
            if found==1:
                docs_id.append(corpus_id)
            corpus_id+=1
        word_docid_mapping[word]=[docs_id,float(float(1)+math.log((len(corpus_data)+1)/(1+len(docs_id))))]
    return word_docid_mapping

def construct_tfidf_matrix(corpus_data,vocabulary,word_docid_mapping,corpus_words_freq_cnts):
    word_id_mapping={}
    cnt=0
    for word in vocabulary:
        word_id_mapping[word]=cnt
        cnt+=1
    feature_matrix=[]
    for i in range(0,len(corpus_data)):
        score_vect=[float(0)]*cnt
        for word in vocabulary:
            if i in word_docid_mapping[word][0]:
                score=float(tf_score(word,i,corpus_data,corpus_words_freq_cnts)*word_docid_mapping[word][1])
                score_vect[word_id_mapping[word]]=score

        score_vect=np.array(score_vect)
        l2_norm = np.linalg.norm(score_vect)
        if l2_norm>float(0):##if norm is zero than ignor that document
            score_vect = score_vect/l2_norm
            feature_matrix.append(score_vect)

    return feature_matrix



def tfidf(vocabulary,corpus_data,corpus_words_freq_cnts):

    word_document_id_mapping=word_docid_mapping(vocabulary,corpus_data)
    tfidf_matrix = construct_tfidf_matrix(corpus_data,vocabulary,word_document_id_mapping,corpus_words_freq_cnts)
    return np.array(tfidf_matrix,dtype=float)



def generating_data(data_paths,vocabulary=None):
    flag=0
    removing_punctuations_marks(data_paths)
    corpus = corpus_formation(data_paths)
    corpus = stopwords_removal(corpus)
    text = [d[1] for d in corpus]
    if vocabulary==None:
        flag=1
        vocabulary = create_vocabulary(text)
    label = [d[0] for d in corpus]
    label = np.array(label)[:,np.newaxis]
    corpus_words_freq_cnts=freq_cnts_of_corpus_words(text)
    tfidf_matrix = tfidf(vocabulary,text,corpus_words_freq_cnts)
    tfidf_matrix-= np.mean(tfidf_matrix,axis=0)
    if flag==1:
        return tfidf_matrix,label,vocabulary
    else:
        return tfidf_matrix,label

#train_tfidf_matrix,ytrain,vocabulary = generating_data(train_data_paths,None)
#test_tfidf_matrix,ytest = generating_data(test_data_paths,vocabulary)


def get_reduced_matrix(tfidf_matrix):
    U,s,V_transposed = np.linalg.svd(tfidf_matrix,full_matrices=False)
    S = np.diag(s)
    threshold=float(0.9)
    sum_singular = float(np.trace(S))
    singuar_values = np.diagonal(S)
    s=float(0)
    cnt=0
    for i in range(0,len(singuar_values)):
        s+=singuar_values[i]
        if s>=(threshold*sum_singular):
            cnt+1
            break
        else:
            cnt+=1
    reduced_dimension=cnt
    B = U[:,0:reduced_dimension].dot(S[0:reduced_dimension,0:reduced_dimension])
    V = V_transposed[0:reduced_dimension,:].transpose()
    return B,V


#xtrain,train_V = get_reduced_matrix(train_tfidf_matrix)
#ytrain = test_tfidf_matrix.dot(train_V)
