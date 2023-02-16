# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:19:04 2020

@author: Randy MSI
"""





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
import numpy as np

#load data

features = pd.read_csv(open('data/testwrite.csv',encoding="utf-8"))
labels = pd.read_csv(open('data/train_labels.tsv',encoding="utf-8"), sep='\t')

#print(features.shape)
#print(labels.shape)



#print(features['avf98'].head())


labelsLi = []
for e in labels.values:    
    labelsLi.append(e[1])
#print(set(labelsLi))

'''
a = pd.get_dummies(labelsLi)
print(a)
'''
featuresLi = []
for e in features.values:    
    featuresLi.append(e[1:])
#print(X_train)
  









from sklearn.model_selection import train_test_split #1st

from sklearn.preprocessing import StandardScaler #2nd LR had to standardize value first
sc = StandardScaler()

from sklearn.linear_model import LogisticRegression #3rd

from sklearn.metrics import accuracy_score #4th
acc_list = []
"""
for i in range(20):

    X_train, X_test, y_train, y_test = train_test_split(featuresLi, labelsLi, test_size=0.3)    
    
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)    

    lr = LogisticRegression(C=0.001)  #defaul regularization = 1

    y_pred = lr.fit(X_train_std, y_train).predict(X_test)

    #compute accuracy
    a = accuracy_score(y_test, y_pred)
    acc_list.append(a)
    print('LR_Accuracy=',a)

from statistics import mean
print('\naverageAcc= ',mean(acc_list))

"""
X_train, X_test, y_train, y_test = train_test_split(featuresLi, labelsLi, test_size=0.3)    
    
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)    

lr = LogisticRegression(C=0.001)  #defaul regularization = 1

y_pred = lr.fit(X_train_std, y_train).predict(X_test)
for e in y_pred:
    print(e,end=' ')


