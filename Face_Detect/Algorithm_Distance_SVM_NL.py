# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:24:23 2016

@author: forest.deng
"""

import numpy as lib_np
import Utility_FaceDetect as ufd
import cv2 as lib_cv2
from sklearn.decomposition import PCA
from sklearn import svm


def get_euclideandistance(x, y):
    """
    """
    x_array = lib_np.array(x)
    y_array = lib_np.array(y)
    
    return lib_np.sqrt(sum((x_array - y_array)*(x_array - y_array)))
    

def get_trained_data(trainDir):
    """
    """
    newsize = (30, 30)
    imgs, nameList = ufd.face_detect_batch_returnimg(trainDir)
    trained_X = []   
    
    pca = PCA(n_components = 1)
    
    trained_Y = nameList
    for img in imgs:
        rimg = lib_cv2.resize(img, newsize)
        rimg = lib_cv2.cvtColor(rimg, lib_cv2.COLOR_BGR2GRAY)
        trained_X.append(pca.fit_transform(rimg).T.tolist())
        
    trained_X = [e[0] for e in trained_X]
    trained_Y = lib_np.array(trained_Y)
    
    svc = svm.SVC(kernel = 'poly', degree = 3)
    
    model = svc.fit(trained_X, trained_Y)

    print ' '        
    
    
get_trained_data(ufd.train_face_folder)            
    
        
        
        
    
    
    