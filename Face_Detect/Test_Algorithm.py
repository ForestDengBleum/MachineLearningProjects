# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:53:18 2016

@author: forest.deng
"""

import Algorithm_Distance_SVM_NL as ads
import Utility_FaceDetect as ufd

# the function will generate all captured faces in train temp folder
# you need to change the name of the face using 'first name' + ' ' + 'last name'
# first letter should be captital 

def generate_train_data(picDir):
    """
    """
    ufd.face_batch_saving(picDir)

train_dir = r"./train_face/"

if __name__=='__main__':

# get model
    train_required = True
    model, trained_X, trained_Y = ads.get_trained_model_data_wrap(
                                                        train_dir,
                                                        train_required
                                                            )
    test_dir = 'D:\\pics'
    
    results, test_X, test_Y = ads.get_test_result(test_dir, model)
    
