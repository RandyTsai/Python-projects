# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:26:42 2020

@author: Randy Tsai 1071545

"""

"""the main difference with final_3 is the #2split part and final executing"""


"""1: load data==================================================================================================="""

def load_data(filename):
    import pandas as pd       
    import numpy as np
    
    data = pd.read_csv(filename,sep=',') 
    data = data.astype(str)  #tranfer all the item into string in case following operation error
    instances = data.values  
    
    
    
    features = []    
    for i in range(len(instances)):         
        tmp_list = []
        for j in instances[i][:-1]:            
            tmp_list.append(j)
        features.append(tmp_list)
    #print(features)
    
    
    classes = []
    for i in range(len(instances)):              
        classes.append(instances[i][-1])             
    #print(classes)
    
    
    uniq_attrs = []
    attributes = np.array(features).T #transform matrix ex: size3x6 to size6x3    
    for e in attributes:
        #print(list(set(e)))
        uniq_attrs.append(list(set(e)))
    #print(uniq_attrs)
    
    
    uniq_class = list(set(classes)) 
    #print(uniq_class)    
    
    
    return instances, features, classes, uniq_attrs, uniq_class


    

"""2: split data==================================================================================================="""

def split_data(instances, num_of_partition, extract_idx):  
    
    
    instances = instances.tolist() #**important turn nparray to python array         
    test_size = len(instances) // num_of_partition  #num_of_portion    
    e = extract_idx      
    
    temp_list = []
    temp_list = instances[:]  #copy a new list without referenc
    #print(temp_list)
       
    test_data = temp_list[ e*test_size : e*test_size+test_size ]
    del temp_list[ e*test_size : e*test_size+test_size ] #delete test data, left train data
    train_data = temp_list
    #print('test_data= ',test_data)   
    #print('train_data= ',train_data,'\n')       
    
    
    def split_feature_class(source_li):
        features = []
        classes = []
        for i in range(len(source_li)):            
            features.append(source_li[i][:-1])
            classes.append(source_li[i][-1])
        #print('features= ',features)
        #print('classes= ',classes)
        return features, classes
    
    
    X_train, y_train = split_feature_class(train_data)
    X_test, y_test = split_feature_class(test_data)    
    
    
    return X_train, X_test, y_train, y_test   



"""3: string feature to numeric feature============================================================================="""

import numpy as np
def turn_sting_into_number_oneD(listA):   
   
    unique_data, numeric_data = np.unique(listA, return_inverse=True)   
    return unique_data, numeric_data 



def turn_sting_into_number_muiltiD(listA):
    
    listA = listA.T  #transform Matrix first, ex dimention3*6 to 6*3
    
    final_return_list = np.array([[ 9 for i in range( len(listA[0]) ) ]]) #inicialize the final list size
    
    for i in listA:        
        #print('origin feature= ',i)
        unique_data, numeric_data = turn_sting_into_number_oneD(i)        
        numeric_data = np.array([numeric_data]) #capsulate in one more dimension
        #print('numeric={}, unique={}'.format(numeric_data, unique_data))
        final_return_list = np.append(final_return_list, numeric_data, axis=0)
        final_return_list = np.delete(final_return_list,0,0)
    final_return_list = final_return_list.T   #transform Matrix back
    return final_return_list
 





"""4  train model, Naive Bayes Algorithm============================================================================"""
def train(X_train, y_train, uniq_attrs, uniq_class):   
    
    import numpy as np
    data = np.array(X_train)         
    #print(data.shape)
    trf_data = data.T    
    #print(trf_data)
    
    
    initial_cls_dict = {}
    for e in uniq_class:
        initial_cls_dict[e] = 0
    #print(initial_cls_dict)
        
    #count how many classes and number of individual class in train data=======
    uniq_cls_count = dict(initial_cls_dict) #copy without reference
    for e in y_train:
        uniq_cls_count[e] = uniq_cls_count.get(e,0)+1 
    #print(uniq_cls_count,'\n')      
        
         
    #get result_list & organize correct attr-class-pair data tree============== 
        
    result_list = []    
    for i in range(len(uniq_attrs)):
        #print(uniq_attrs[i])
        sub_list = []
        sub_dict = {}
        for j in range(len(uniq_attrs[i])):
            copy_dict = dict(initial_cls_dict) #***copy without reference
            sub_dict[uniq_attrs[i][j]] = sub_dict.get(uniq_attrs[i][j] , copy_dict )
        #print(sub_dict)
        sub_list.append(sub_dict)
        #print(sub_list)
        result_list.append(sub_list)
    #print(result_list)
    #try print(result_list[0][0]['F'])
    #try print(result_list[1][0]['M'])
    #try print(result_list[1][0]['R'])
     
    
        
    #calculate prior probabilities=============================================
        
    prior_prob = {}
    for k,v in uniq_cls_count.items():#incase key error we should use global 'uniq_class' generate from the beginning instead of class merely in train data      
        prior_prob[k] = round( v/sum( uniq_cls_count.values() ), 3)
    #print(prior_prob)
    

    
    #calculate class conditional probabilities=================================
    
    def compare(lookup_idx, pass_attr, pass_class):
        
        for attr_name, v in result_list[lookup_idx][0].items():
            #print(attr_name,v) 
            #print('pass_attr=',pass_attr)
            if pass_attr == attr_name:                               
                result_list[lookup_idx][0][attr_name][pass_class] += 1        
                
    #print(result_list[1][0]['R']['D'])
    #test compare(0,'F','D')
    #test compare(0,'M', 'C')
    #print(result_list)    
        
    
    for i in range(len(X_train)):        
        for j in range(len(X_train[i])):  
            
            '''j is the column_idx in xtrain, also the lookup_idx of where we found 
            a specific branch in result_list[]'''            
            pair_attr = X_train[i][j] #attr of X_train in row[i]column[j]
            pair_label = y_train[i]   #class of y_train data in column[i]
            #print('pass_attr=', pair_attr)
            #print('pass_label=', pair_label)
            
            compare(j, pair_attr, pair_label)
            #use these 3value to find corrosponding location at result_list for quantity counting
        
    
         
    #compute probability of each attr:class P(X|Y) of all
    for i in range(len(result_list)):
        #print (result_list[i][0])
        for key in result_list[i][0]:
            for cls_name in uniq_class:
               
                #print(key, cls_name)
                #try:
                probability = result_list[i][0][key][cls_name] / uniq_cls_count[cls_name]                                                                                
                #except:
                    #print('something wrong')
                
                #smoothing
                if result_list[i][0][key][cls_name] ==0:
                    result_list[i][0][key][cls_name] = round( 1/sum(uniq_cls_count.values())*10, 3)
                else:
                    result_list[i][0][key][cls_name] = round(probability, 3)
    
        
           
    return result_list, prior_prob
          



"""5 predict ======================================================================================================"""

def predict(X_test, uniq_class, NB_cls_prob):
    import numpy as np
    from functools import reduce
    
    instnces = X_test    
    y_predict=[]         

    for i in range(len(X_test)):
        clsProb_ByLabel = []
        
        #extract every class probability with the feature
        for cls_name in uniq_class:
            lookup_numList = []        
            for j in range(len(X_test[i])):        
                attr=X_test[i][j]
                looup_num = NB_cls_prob[j][0][attr][cls_name]
                lookup_numList.append(looup_num)
                
            cls_con_probaility = reduce(lambda x,y:x*y, lookup_numList ) * prior_prob[cls_name]            
            clsProb_ByLabel.append(cls_con_probaility)    
        
        idx = np.argmax(clsProb_ByLabel) #argmax P(X|Y), which class has max probability      
        y_predict.append(uniq_class[idx])    
      
    
    return y_predict

      



"""6 evaluate ======================================================================================================"""

def evaluate(y_predict, y_test):

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
        
    
    #Accuracy
    accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy= ', accuracy)
    
    
    #trun string to number
    str_test, num_test = turn_sting_into_number_oneD(y_test)
    #print(str_test, num_test)
    str_predict, num_predict = turn_sting_into_number_oneD(y_predict)
    #print(str_predict, num_predict)
    
    #Precision
    precision = precision_score(num_test, num_predict, average='macro')
    print('precition= ', precision)
    
    #Recall
    recall = recall_score(num_test, num_predict, average='macro')
    print('recall= ', recall)
    
    #F1_scaore
    f1_score = f1_score(num_test, num_predict, average='macro')
    print('F1_score= ', f1_score,'\n')
    
    
    return accuracy
    



    
    
    









"""Execute ================================================================"""

import random as rd

instances, features, classes, uniq_attrs, uniq_class = load_data('student.csv') #features, classes unuse
num_of_partition = 3
acc_list = []
#rd.shuffle(instances) #shuffle before run!!!!!!after shuffle!!Accuracy up to0.48!!
for extract_idx in range(num_of_partition):
    
    #split data(instance, proportion_of_test_data)
    X_train, X_test, y_train, y_test = split_data(instances,num_of_partition, extract_idx) 


    #X_train, y_train
    NB_cls_prob, prior_prob = train(X_train, y_train, uniq_attrs, uniq_class)


    #predict result
    y_predict = predict(X_test, y_test, NB_cls_prob)


    #evaluate accuracy
    acc_score = evaluate(y_predict, y_test)
    acc_list.append(acc_score)
    
import statistics as st
print(acc_list ,'and average= ',st.mean(acc_list))

 



 




