# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:37:28 2020

@author: Randy MSI
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#load data
train_features = pd.read_csv(open('data/train_features_wb.csv',encoding="utf-8"))
train_labels = pd.read_csv(open('data/train_labels.tsv',encoding="utf-8"), sep='\t')

y_train = []
for e in train_labels.values:    
    y_train.append(e[1])
#print(y_train)


X_train = []
for e in train_features.values:    
    X_train.append(e[1:])
#print(np.array(X_train).shape)



valid_features = pd.read_csv(open('data/valid_features_wb.csv',encoding="utf-8"))
valid_labels = pd.read_csv(open('data/valid_labels.tsv',encoding="utf-8"), sep='\t')

y_test = []
for e in valid_labels.values:    
    y_test.append(e[1])
#print(y_test)

X_test = []
for e in valid_features.values:
    X_test.append(e[1:])
#print(np.array(X_test).shape)










"""
#for MultinomianlNB transform data from 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
"""



from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
import random as rd
#NB Classifier    #gnb  or  mnb

mnb = MultinomialNB(alpha=0.7)

    
tmp = list(zip(X_train, y_train))
rd.shuffle(tmp)
X_train, y_train = zip(*tmp)


#train model and predict
y_pred = mnb.fit(X_train, y_train).predict(X_test) #change gnb or mnb

#compute accuracy
a = mnb.score(X_test, y_test)
print('mNB_Accuracy=',a)
    
  










