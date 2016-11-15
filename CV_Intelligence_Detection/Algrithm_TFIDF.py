# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:21:49 2016

@author: forest.deng
"""

import Algrithm_CVJDSimilarity as ac
import numpy as lib_np
import pandas as lib_pd

def get_tfidfmatrics(inputListCollection):
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
        idf.append(lib_np.log(float(totalLen)/sum(idf_wordscount))+0.5)
        tf_idf = [e*float(idf[-1]) for e in tf]
        index = list(idfMatrix[idfMatrix.word==word].index)[0]
        for i in range(totalLen):
            idfMatrix.set_value(index,'cv'+str(i),tf_idf[i])        
        
    return idfMatrix        