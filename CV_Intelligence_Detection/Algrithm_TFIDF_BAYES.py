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

def get_trained_tfidfmatrics(inputListCollection):
    """
    """
    allWords=[]
    DataFrameList = []
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

    for i in range(totalLen):
        DataFrameList.append(ac.get_dataframewithper(inputListCollection[i]))            

    for word in allWords:
        idf_wordscount = []
        tf = []
        for i in range(totalLen):
            opDF = DataFrameList[i]
            if len(opDF[opDF.word==word]) == 0:
                idf_wordscount.append(0)
                tf.append(0)
            else:
                idf_wordscount.append(1)
                tf.append(float(opDF[opDF.word==word].percentage))
#        idf.append(lib_np.log(float(totalLen)/sum(idf_wordscount))+0.5)
        idf.append(sum(idf_wordscount)/float(totalLen))
        tf_idf = [e*float(idf[-1]) for e in tf]
        index = list(idfMatrix[idfMatrix.word==word].index)[0]
        for i in range(totalLen):
            idfMatrix.set_value(index,'cv'+str(i),tf_idf[i])        
        
    return idfMatrix.sort(['word'], ascending = 0)        

# customize the tfidf in order to use more features. i.e. the matching degree
# btween JD and CVs
    
def get_trained_centroid_tfidfmatrics(inputListCollection):
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
    for word in test_list:
        if len(train_matrix[train_matrix.word == word]) == 0:
            prob *= c
        else:
            prob *= float(train_matrix.weights)
    return prob * priority

# try to get td df feature of trained docs for logic regress

def get_trained_tddf_feature(inputListCollection):
    """
    """
    tddf_matrix = get_trained_tfidfmatrics(inputListCollection)
    centroid_matrix = get_trained_centroid_tfidfmatrics(inputListCollection)
    distance = []
    columnsList = list(tddf_matrix.columns)
    columnsList.remove('word')
    centroid_denominator = lib_math.sqrt(sum(centroid_matrix['weights']*
                            centroid_matrix['weights']))
    for col in columnsList:
        numerator = sum(tddf_matrix[col]*centroid_matrix['weights'])
        denominator = lib_math.sqrt(sum(tddf_matrix[col]*
                            tddf_matrix[col]))*centroid_denominator         
        distance.append(numerator/denominator)
    return distance


def get_test_tddf_feature(train_set, inputListCollection, c=1e-6):
    """
    """
    inputlistLen = len(inputListCollection)
    for i in range(inputlistLen):
        dic = ac.dict_counter(inputListCollection[i])
        


#                        
#    centroid_denominator =     
#    for col in columnsList:
#        temp_numerator = 0.0
#        temp_denominator = 0.0        
#        for word in centroid_matrix.word:
#            if tddf_matrix[tddf_matrix.word == word][col] > 0.0:
#                temp_numerator += float(tddf_matrix[tddf_matrix.word 
#                                  == word][col]) * float(centroid_matrix[
#                                  centroid_matrix.word == word]['weights'])
#                temp_denominator += float(tddf_matrix[tddf_matrix.word 
#                                == word][col]^2                   
#        numerator.append(temp_numerator)
#        temp_denominator = 
                                  
                  
    
    