# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:11:31 2016

@author: forest.deng
"""

import Utility_Input2Text as uit
import Utility_TextClearance as utc
import Algrithm_TFIDF_BAYES as at
import Algrithm_CVJDSimilarity as ac

# .Net engineer
cvpath_net_engineer = 'D:\\data\\.Net Engineer\\cv'
jdpath_net_engineer = 'D:\\data\\.Net Engineer\\jd'

# get data and pursed

train_cvlist = uit.get_listtext(cvpath_net_engineer)
train_jdlist = uit.get_listtext(jdpath_net_engineer)

pursed_jdlist = utc.getJDList(train_jdlist[0])
pursed_cvlist = utc.datapurse_collection_cv(train_cvlist, pursed_jdlist)

# get features
cv_jd_similarity = ac.get_collection_similarity(pursed_cvlist, 
                                                pursed_jdlist)
cv_centroid_similarity, train_centroid_matrix = \
                    at.get_trained_tddf_similarity(pursed_cvlist)

print ''