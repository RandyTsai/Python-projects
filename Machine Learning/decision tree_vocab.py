# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:46:56 2020

@author: Randy MSI
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
import numpy as np

#load data

features = pd.read_csv(open('data/testwrite2.csv',encoding="utf-8"))
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
  



from sklearn.model_selection import train_test_split   

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
"""
for i in range(20):
    
    X_train, X_test, y_train, y_test = train_test_split(featuresLi, labelsLi, test_size=0.3)
    
    tree = DecisionTreeClassifier(criterion = 'entropy')
    y_pred = tree.fit(X_train, y_train).predict(X_test)
    
    #compute accuracy
    a = accuracy_score(y_test, y_pred)
    print('DT-Accuracy=',a)
'====================================================='
print('\n\n')
"""
from sklearn.ensemble import RandomForestClassifier

list_A = []
for i in range(20):
    
    X_train, X_test, y_train, y_test = train_test_split(featuresLi, labelsLi, test_size=0.3)

    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, n_jobs=5)

    y2_pred = forest.fit(X_train, y_train).predict(X_test)

    b = accuracy_score(y_test, y2_pred)
    list_A.append(b)
    print('RF-Accuracy=',b)

from statistics import mean
print('\naverageAcc= ',mean(list_A))

