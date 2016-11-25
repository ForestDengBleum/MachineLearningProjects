# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:53:18 2016

@author: forest.deng
"""

import Algorithm_Distance_SVM_NL as ads


train_dir = r"./train_face/"

if __name__=='__main__':

# get model
    train_required = True

    (model, 
     mapping, 
     trained_X, 
     trained_Y
             ) = ads.get_trained_model_data_wrap_neuro(
                                                         train_dir,
                                                         train_required
                                                         )
    test_dir = 'D:\\pic\\test'

    (results, test_X, test_Y) = ads.get_test_result_neuro(
                                                        test_dir, 
                                                        model,
                                                        mapping)
    
    
    
    print ''

#SVC
#    model, trained_X, trained_Y = ads.get_trained_model_data_wrap(
#                                                        train_dir,
#                                                        train_required
#                                                            )
#    centroids = ads.get_category_centroid(trained_X, trained_Y)                                                        
#    test_dir = 'D:\\pic\\test'
#    
#    results, test_X, test_Y = ads.get_test_result(test_dir, model, centroids)
    
