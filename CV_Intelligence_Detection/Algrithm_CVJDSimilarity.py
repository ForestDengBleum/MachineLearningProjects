# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 08:44:16 2016

@author: forest.deng
"""
import pandas as lib_pd
import math as lib_math
#import numpy as lib_np
#from enchant.tokenize import get_tokenizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
# Get Tokenizer

def text_tokenize(docs):
    """
    """
#    tknz = get_tokenizer('en_US')
    cv = CountVectorizer()
    x1 = cv.fit_transform(docs).toarray().transpose()
    return lib_pd.DataFrame(x1,cv.get_feature_names())
    

def list_counter(inputList):
    """
    """
    return Counter(inputList)
    
def dict_counter(inputList):
    """
    """
    counts = list_counter(inputList)
    return dict(counts)
    
def get_dataframewithper(inputlist):
    """
    """
    counts = list_counter(inputlist)
    df = lib_pd.DataFrame(zip(dict(counts).keys(),dict(counts).values()))
    df.columns = ['word','counts']
    df = df.sort(['counts'], ascending=[0])
    
    for ind in df[df['counts']>12].index:
        df.set_value(ind, 'counts', 12)
    
#    df['percentage'] = lib_np.log(df.counts/sum(df.counts)*1000000)
    df['percentage'] = df.counts/sum(df.counts)    
    #print df
    return df
    
# get single cv similarity
    
def get_similarity(cvlist, jdlist):
    """
    """
    cvframe = get_dataframewithper(cvlist)
    jdframe = get_dataframewithper(jdlist)
    numerator = float(0)
    denominator = 0.0
    for cvrow in cvframe.iterrows():
        if len(jdframe[jdframe['word'] == cvrow[1][0]])==1:
            numerator += float(cvrow[1][2]) * \
                float(jdframe[jdframe['word'] == cvrow[1][0]]['percentage'])
    denominator = lib_math.sqrt(sum(cvframe['percentage']*\
                                cvframe['percentage'])) * \
                  lib_math.sqrt(sum(jdframe['percentage']*\
                                jdframe['percentage']))
    return (numerator/denominator)
    

# get similarity for series of cvs

def get_collection_similarity(cvlists, jdlist):
    """
    """
    similarity = []
    jdframe = get_dataframewithper(jdlist)
    for cvlist in cvlists:
        cvframe = get_dataframewithper(cvlist)
        numerator = float(0)
        denominator = 0.0
        for cvrow in cvframe.iterrows():
            if len(jdframe[jdframe['word'] == cvrow[1][0]])==1:
                numerator += float(cvrow[1][2]) * \
                    float(jdframe[jdframe['word'] == 
                            cvrow[1][0]]['percentage'])
        denominator = lib_math.sqrt(sum(cvframe['percentage']*\
                                    cvframe['percentage'])) * \
                      lib_math.sqrt(sum(jdframe['percentage']*\
                                    jdframe['percentage']))
        similarity.append(numerator/denominator)
    return similarity

    