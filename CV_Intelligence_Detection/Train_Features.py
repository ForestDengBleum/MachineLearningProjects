# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:11:31 2016

@author: forest.deng
"""

import Utility_Input2Text as uit
import Utility_TextClearance as utc
import Algorithm_TFIDF_BAYES as at
import Algorithm_CVJDSimilarity as ac

import numpy as lib_np
import string as lib_str
import cPickle as lib_cp

#from sklearn.linear_model import LogisticRegression
from sklearn import svm

persistence_folder = '\\persistence'

def get_train_data(train_cv_path, train_jd_path):

# get data and pursed
    
    train_cvlist, cv_namelist = uit.get_listtext(train_cv_path)
    train_jdlist, jd_namelist = uit.get_listtext(train_jd_path)
    
    trained_Y = ([int(lib_str.split(lib_str.split(e,'.')[0],'_')[-1]) 
                    for e in cv_namelist ])
    
    jd_titlelist = utc.getJDTitlelist(jd_namelist[0])
    pursed_jdlist = utc.getJDList(train_jdlist[0])
    pursed_cvlist = utc.datapurse_collection_cv(train_cvlist, pursed_jdlist)
    
    # get features
    cv_jd_similarity = ac.get_collection_similaritywithfactors(pursed_cvlist, 
                                                    pursed_jdlist,
                                                    jd_titlelist)
    cv_centroid_similarity, train_centroid_matrix = \
                        at.get_trained_tddf_similarity(pursed_cvlist, 
                                                       trained_Y)
                            
    trained_X = lib_np.array(zip(cv_jd_similarity,cv_centroid_similarity))
    
# data persistence
    train_X_file = open(train_cv_path + persistence_folder + 
                        '\\trained_X.txt', 'w')
    train_Y_file = open(train_cv_path + persistence_folder + 
                        '\\trained_Y.txt', 'w')
    pursed_jdlist_file = open(train_jd_path + persistence_folder +
                        '\\jd.txt', 'w')
    train_centroid_matrix_file = open(train_cv_path + persistence_folder + 
                                    '\\centroid.txt', 'w')
    lib_cp.dump(trained_X, train_X_file)
    lib_cp.dump(trained_Y, train_Y_file)            
    lib_cp.dump(pursed_jdlist, pursed_jdlist_file)
    lib_cp.dump(train_centroid_matrix, train_centroid_matrix_file)
    
    train_X_file.close()
    train_Y_file.close()
    pursed_jdlist_file.close()
    train_centroid_matrix_file.close()    
    
#plt.show()


def get_trained_model(trained_X, trained_Y):

    svc = svm.SVC(kernel = 'linear')
    model = svc.fit(trained_X, trained_Y)
    
    return model
    

def get_trained_model_wrap(train_cv_path, train_jd_path, 
                           trainedRequired = True):
    """
    """
    if trainedRequired:
        get_train_data(train_cv_path, train_jd_path)
    trained_X_file = open(train_cv_path + persistence_folder +
                        '\\trained_X.txt')
    trained_Y_file = open(train_cv_path + persistence_folder +
                        '\\trained_Y.txt')
    trained_X = lib_cp.load(trained_X_file)
    trained_Y = lib_cp.load(trained_Y_file)
    trained_X_file.close()
    trained_Y_file.close()
    svc = svm.SVC(kernel = 'linear', probability = True)
    model = svc.fit(trained_X, trained_Y)
    return model, trained_X, trained_Y   

    
def get_test_data(test_cv_path, train_jd_path, train_cv_path):
    """
    """
    test_cvlist, test_cv_namelist = uit.get_listtext(test_cv_path)
    pursed_jdlist_file = open(train_jd_path + persistence_folder + 
                            '\\jd.txt')
    train_centroid_matrix_file = open(train_cv_path + persistence_folder + 
                            '\\centroid.txt')
    train_centroid_matrix = lib_cp.load(train_centroid_matrix_file)
    pursed_jdlist = lib_cp.load(pursed_jdlist_file)
    pursed_jdlist_file.close()
    test_pursed_cvlist = utc.datapurse_collection_cv(test_cvlist, 
                                                     pursed_jdlist)
    test_cv_jd_similarity = ac.get_collection_similarity(
                                                     test_pursed_cvlist, 
                                                     pursed_jdlist )
    test_cv_centroid_similarity = at.get_test_tddf_similarity(
                                train_centroid_matrix, test_pursed_cvlist)
    test_X = lib_np.array(zip(test_cv_jd_similarity,
                              test_cv_centroid_similarity))

    return test_X, test_cv_namelist     
# get test data










