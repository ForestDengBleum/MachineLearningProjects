# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:21:49 2016

@author: forest.deng
"""

import Algrithm_CVJDSimilarity as ac
import numpy as lib_np
import pandas as lib_pd
import math as lib_math

# tfidf implementation instead of using CounterVetorize
# changed tfidf idf part for our train is based on JD and only the occurance
# should be positive related with the similarity
# only consider counts>1

def get_trained_tfidfmatrics(inputListCollection, countLimit = 0):
    """
    """
    allWords=[]
    InputDataFrameList = []
    idf = []
    totalLen = len(inputListCollection)

    for i in range(totalLen):
        dic = ac.dict_counter(inputListCollection[i])
        allWords.extend(dic.keys())
    allWords = list(set(allWords))
    allWordsLen = len(allWords)

    idfMatrix = lib_pd.DataFrame(allWords, columns=['word'])
    for i in range(totalLen):
        idfMatrix['cv'+str(i)] = lib_np.zeros(allWordsLen,float)
    idfMatrix['counts'] = lib_np.zeros(allWordsLen,float)
    
    for i in range(totalLen):
        InputDataFrameList.append(ac.get_dataframewithper
                                (inputListCollection[i]))            

    for word in allWords:
        idf_wordscount = []
        tf = []
        cnt = 0.0
        for i in range(totalLen):
            opDF = InputDataFrameList[i]
            if len(opDF[opDF.word==word]) == 0:
                idf_wordscount.append(0)
                tf.append(0)
            else:
                idf_wordscount.append(1)
                tf.append(float(opDF[opDF.word==word].percentage))
                cnt += float(opDF.counts)
#        idf.append(lib_np.log(float(totalLen)/sum(idf_wordscount))+0.5)
        idf.append(sum(idf_wordscount)/float(totalLen))
        tf_idf = [e*float(idf[-1]) for e in tf]
        index = list(idfMatrix[idfMatrix.word==word].index)[0]
        for i in range(totalLen):
            idfMatrix.set_value(index,'cv'+str(i),tf_idf[i])        
        idfMatrix.set_value(index,'counts', cnt)
    if countLimit != 0:
        return idfMatrix[idfMatrix.counts > countLimit].sort(['word'], \
                        ascending = 0)
    else:
        return idfMatrix.sort(['word'], ascending = 0)

# customize the tfidf in order to use more features. i.e. the matching degree
# btween JD and CVs
    
def get_trained_centroid_tfidfmatrics(inputListCollection, countLimit = 0):
    """
    """
    allWords=[]
    DataFrameList = []
    totalLen = len(inputListCollection)

    for i in range(totalLen):
        dic = ac.dict_counter(inputListCollection[i])
        allWords.extend(dic.keys())
    allWords = list(set(allWords))
    allWordsLen = len(allWords)

    idfMatrix = lib_pd.DataFrame(allWords, columns=['word'])
    idfMatrix['frequency'] = lib_np.zeros(allWordsLen,float)
#    idfMatrix['density'] = lib_np.zeros(allWordsLen,float)
    idfMatrix['occurrence'] = lib_np.zeros(allWordsLen,float)

    for i in range(totalLen):
        DataFrameList.append(ac.get_dataframewithper(inputListCollection[i]))            

    for word in allWords:
        frequency = 0.0
        occurence = 0.0
        for i in range(totalLen):
            opDF = DataFrameList[i]
            if len(opDF[opDF.word==word]) != 0:
                frequency += float(opDF[opDF.word==word].counts)
                occurence += 1.0
        occurence = occurence/totalLen
        index = list(idfMatrix[idfMatrix.word==word].index)[0]
        idfMatrix.set_value(index,'frequency',frequency)
        idfMatrix.set_value(index,'occurrence',occurence)
    # remove the words having counts equal to 1
    if countLimit !=0:
        idfMatrix = idfMatrix[idfMatrix.frequency > countLimit]                
    idfMatrix['density'] = idfMatrix.frequency/sum(idfMatrix.frequency)
    idfMatrix['weights'] = idfMatrix.occurrence*idfMatrix.density    
            
    return idfMatrix.sort(['word'], ascending = 0)        

# bayes probability
# two training sets: passed and failed
# for a new cv calculate the probability of passed and failed and 
# decide its categry by the bigger one

def calculate_bayesprobability(train_matrix, test_list,
                               priority, c = 1e-6 ):
    """
    """
    prob = 1.0
    test_list = list(set(test_list))
    for word in test_list:
        if len(train_matrix[train_matrix.word == word]) == 0:
            prob *= c
        else:
            prob *= float(train_matrix.weights)
    return prob * priority

# use Naive bayes to get a list of probability 

def calculate_collection_bayesprobability(train_matrix, test_listCollection,
                               priority, c = 1e-6 ):
    """
    """
    probList = []
    for test_list in test_listCollection:
        test_list = list(set(test_list))        
        prob = 1.0
        for word in test_list:
            if len(train_matrix[train_matrix.word == word]) == 0:
                prob *= c
            else:
                prob *= float(train_matrix.weights)
        probList.append(prob*priority)        
    return probList


# try to get td df feature of trained docs for logic regress

def get_trained_tddf_similarity(inputListCollection):
    """
    """
    tddf_matrix = get_trained_tfidfmatrics(inputListCollection)
    centroid_matrix = get_trained_centroid_tfidfmatrics(inputListCollection)
    distance = []
    columnsList = list(tddf_matrix.columns)
    columnsList.remove('word')
    columnsList.remove('counts')
    centroid_denominator = lib_math.sqrt(sum(centroid_matrix['weights']*
                            centroid_matrix['weights']))
    for col in columnsList:
        numerator = sum(tddf_matrix[col]*centroid_matrix['weights'])
        denominator = lib_math.sqrt(sum(tddf_matrix[col]*
                            tddf_matrix[col]))*centroid_denominator         
        distance.append(numerator/denominator)
    return distance, centroid_matrix

def get_trained_bayesprobability(inputListCollection):
    """
    """
    centroid_matrix = get_trained_centroid_tfidfmatrics(inputListCollection)
    inputLen = len(inputListCollection)
    
    weightList = []    
    
    for i in range(inputLen):
        words = set(inputListCollection[i])
        weight = 1.0
        for word in words:
            if len(centroid_matrix[centroid_matrix.word == word])>0:
                weight *= float(centroid_matrix[centroid_matrix.word 
                            == word]['weights']*100)                    
        weightList.append(weight)
    return weightList                    

# calculate test similarity to trained centroid

def get_test_tddf_similarity(train_centroid_matrix, inputListCollection, 
                             c=1e-6):
    """
    """
    inputlistLen = len(inputListCollection)
    similarity = []
    train_denominator = lib_math.sqrt(sum(train_centroid_matrix.weights*
                        train_centroid_matrix.weights))
    for i in range(inputlistLen):
        test_df = ac.get_dataframewithper(inputListCollection[i])
        weights = []
        for word in test_df.word:
            if len(train_centroid_matrix[train_centroid_matrix.word 
                    == word]) == 0:
                weights.append(c*test_df[test_df.word == word]
                                    ['percentage'])                        
            else:
                weights.append(float(train_centroid_matrix
                                    [train_centroid_matrix.word
                                    == word]['occurrence']) * 
                float(test_df[test_df.word == word]['percentage']))
        test_df['weights'] = weights
        numerator = 0.0        
        for word in test_df.word:
            if len(train_centroid_matrix[train_centroid_matrix.word 
                    == word]) == 0:
                numerator += (float(train_centroid_matrix[train_centroid_matrix.word
                            == word]['weights'])*
                            float(test_df[test_df.word == word]['weights']))
        test_denominator = lib_math.sqrt(sum(test_df.weights*test_df.weights))
        similarity.append(numerator/(train_denominator*test_denominator))
    return similarity                      
  
    