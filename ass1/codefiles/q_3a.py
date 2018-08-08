#!/usr/bin/env python
import os
import sys
import numpy as np
import random
train_data_path=sys.argv[1]
dataset_name=train_data_path.split("/")[6]
script_path =os.path.dirname(os.path.realpath(__file__))
if train_data_path.split("/")[-1]!='':
    train_data_path+="/"
no_of_classes=len(next(os.walk(train_data_path))[1])
train_data_paths=[train_data_path+str(cl)+"/" for cl in next(os.walk(train_data_path))[1]]
train_data_paths=sorted(train_data_paths)
for class_path in train_data_paths:
    num_docs=len(next(os.walk(class_path))[2])
    num_testdocs=num_docs/5
    test_docs=[]
    c=0
    while 1:
        test_doc = random.choice(os.listdir(class_path))
        if test_doc not in test_docs:
            test_docs.append(test_doc)
            c+=1
        if c==num_testdocs:
            break

    destination_path=script_path+"/"+dataset_name+"/test/"+class_path.split("/")[8]+"/"
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for doc in test_docs:
        os.rename(class_path+doc, destination_path+doc)
