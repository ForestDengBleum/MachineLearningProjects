# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:11:31 2016

@author: forest.deng
"""

import Utility_Input2Text as uit
import Utility_TextClearance as utc
import Algrithm_TFIDF_BAYES as at
import Algrithm_CVJDSimilarity as ac

import numpy as lib_np
import matplotlib.pyplot as plt
import string as lib_str

#from sklearn.linear_model import LogisticRegression
from sklearn import svm




# .Net engineer
train_cvpath_netengineer = 'D:\\data\\.Net Engineer\\train\\cv'
train_jdpath_netengineer = 'D:\\data\\.Net Engineer\\train\\jd'
test_cvpath_netengineer = 'D:\\data\\.Net Engineer\\test'


# get data and pursed



train_cvlist, cv_namelist = uit.get_listtext(train_cvpath_netengineer)
train_jdlist, jd_namelist = uit.get_listtext(train_jdpath_netengineer)

trained_Y = ([int(lib_str.split(lib_str.split(e,'.')[0],'_')[-1]) 
                for e in cv_namelist ])

pursed_jdlist = utc.getJDList(train_jdlist[0])
pursed_cvlist = utc.datapurse_collection_cv(train_cvlist, pursed_jdlist)

# get features
cv_jd_similarity = ac.get_collection_similarity(pursed_cvlist, 
                                                pursed_jdlist)
cv_centroid_similarity, train_centroid_matrix = \
                    at.get_trained_tddf_similarity(pursed_cvlist, trained_Y)
                        
trained_X = lib_np.array(zip(cv_jd_similarity,cv_centroid_similarity))

plt.scatter(trained_X[:,0],trained_X[:,1])

#plt.show()



# LogisticRegression using default parameters

svc = svm.SVC(kernel = 'linear')
model = svc.fit(trained_X, trained_Y)

# get test data
test_cvlist, cv_namelist = uit.get_listtext(test_cvpath_netengineer)
test_pursed_cvlist = utc.datapurse_collection_cv(test_cvlist, pursed_jdlist)
test_cv_jd_similarity = ac.get_collection_similarity(test_pursed_cvlist, 
                                                     pursed_jdlist )
test_cv_centroid_similarity = at.get_test_tddf_similarity(
                                train_centroid_matrix, test_pursed_cvlist)
test_X = lib_np.array(zip(test_cv_jd_similarity,test_cv_centroid_similarity))

print model.predict(test_X)







