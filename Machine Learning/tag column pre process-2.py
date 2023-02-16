# -*- coding: utf-8 -*-
"""
Created on Thu May  7 03:57:17 2020

@author: Randy MSI
"""

"""this file preprocess data
deal with ONLY 'tag' column
"""


import pandas as pd
import numpy as np

#load data
features = pd.read_csv(open('data/train_features.tsv',encoding="utf-8"), sep='\t')
labels = pd.read_csv(open('data/train_labels.tsv',encoding="utf-8"), sep='\t')
#print(features.values.shape)

valid_features = pd.read_csv(open('data/valid_features.tsv',encoding="utf-8"), sep='\t')

test_features = pd.read_csv(open('data/NEW_test_features.tsv',encoding="utf-8"), sep='\t') 

mix_tags = pd.read_csv(open('data/combineT&Vtags.csv',encoding="utf-8"))







'''======================================================================================'''
"""features process, one-hot encoding of words bag"""

#build dictionay of each word, hence key_list=unique vocabulary value, value_list= row value(of each tag) of each movie
#注意  就是這裡跟版本1不同了 這次是等於要train跟vlid的tag一起統整所有tag做column 
    #但卻只有統計train feature裡的tag的資料  才能用valid測



def word_bag_producer(li):    
    
    voca = []
    for w in mix_tags.values:    
        temp = w[0].strip().split(',')
        for j in temp:
            voca.append(j)    
    #print(len(set(voca))) 
    
    
    
    count = {} #統一製造同樣數量的column空的dict 
    for i in set(voca): 
        count[i] = 0
    #print('original dictionary: ',count,'\n')
    #print(count.values()) #initial value in each column of a movie
    #print(count.keys()) #calumn(feature) names
    
    word_bag = []
    for i in range(len(li)):
        copy_dict = dict(count) #very important copy without reference
        for j in li[i]:        
            copy_dict[j] = copy_dict.get(j,0)+1
            #print(copy_dict.values())
        word_bag.append( list(copy_dict.values()) )

    #print('original word bags=:\n',pd.DataFrame(word_bag),'\n')
    return(word_bag)

'''======================================================================================'''







train_tags = [] #extract only tag feature of each movie
for e in features.values:  #plus head() to show top 5 instances
    temp = e[4].strip().split(',')            
    train_tags.append(temp)
#print(train_tags)
wb1 = word_bag_producer(train_tags)
#print(pd.DataFrame(wb1))
#pd.DataFrame(wb1).to_csv(r'data/train_features_wb.csv', index = True, header=True)


valid_tags = []
for e in valid_features.values:
    temp = e[4].strip().split(',')            
    valid_tags.append(temp)
wb2 = word_bag_producer(valid_tags)
#print(pd.DataFrame(wb2))
#pd.DataFrame(wb2).to_csv(r'data/valid_features_wb.csv', index = True, header=True)


test_tags = []
for e in test_features.values:
    temp = e[4].strip().split(',')            
    test_tags.append(temp)
wb3 = word_bag_producer(test_tags)
#print(pd.DataFrame(wb2))
#pd.DataFrame(wb3).to_csv(r'data/test_features_wb.csv', index = True, header=True)







