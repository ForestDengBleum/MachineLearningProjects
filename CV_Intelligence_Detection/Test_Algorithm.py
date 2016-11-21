# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:41:54 2016

@author: forest.deng
"""
import Train_Features as tf
import matplotlib.pyplot as plt
import numpy as lib_np


if __name__=='__main__':

    cv_path_temp = 'D:\\data\\rolename\\train\\cv'
    jd_path_temp = 'D:\\data\\rolename\\train\\jd' 
    cv_test_path_temp = 'D:\\data\\rolename\\test\\cv'

    fig = 0

# Group 1
    
    
# Training data and get model

    trainedRequired_netengineer = False
    role_name = '.Net Engineer'
    train_cvpath_netengineer = cv_path_temp.replace('rolename', role_name)
    train_jdpath_netengineer = jd_path_temp.replace('rolename', role_name)
    
    model_netengineer, trained_X, trained_Y = tf.get_trained_model_wrap(
                                        train_cvpath_netengineer,
                                        train_jdpath_netengineer,
                                        trainedRequired_netengineer)    
    print 'Use SVC to train and get the model: %s' %model_netengineer                                    
#    print 'SVC Model - coef_: %s' %model_netengineer.coef_
#    print 'SVC Model - Intercept_: %s' %model_netengineer.intercept_
    plt.figure(fig)
    plt.title('Train/Test data distribution for Role: ' + role_name 
               + '\n' + 'Train Data: circle; '+ 'Test Date: triangle' )
    plt.xlabel('JD similarity')
    plt.ylabel('Centroid similarity')
# decision boundary
    x_min, x_max = trained_X[:, 0].min() - 0.03, trained_X[:, 1].max() + 0.03
    y_min, y_max = trained_X[:, 1].min() - 0.03, trained_X[:, 1].max() + 0.03
    xx, yy = lib_np.meshgrid(lib_np.arange(x_min, x_max, 0.01), 
                             lib_np.arange(y_min, y_max, 0.01))
    z_predict = model_netengineer.predict(zip(xx.ravel(), yy.ravel()))
    z_predict = z_predict.reshape(xx.shape)
    plt.contourf(xx, yy, z_predict, alpha = 0.3)
#    plt.xlim(0,1)
#    plt.ylim(0,1)
    marker ='o'
    color ='rb'
    for x in trained_X:
        plt.scatter(x[0],x[1], s = 40, marker = marker, c=color[
                                model_netengineer.predict(x)])

    
# get test data   
    test_cvpath_netengineer = cv_test_path_temp.replace('rolename', role_name)  
    test_X, test_X_namelist = tf.get_test_data(test_cvpath_netengineer,
                              train_jdpath_netengineer,
                              train_cvpath_netengineer)
# predict cv     
    predict_netengineer = model_netengineer.predict(test_X)
    print 'Predict results: %s' %  zip(test_X_namelist, 
                                       list(predict_netengineer))
    print 'Predict Probability: %s' % model_netengineer.predict_proba(test_X)

    tmarker = '^'
    size = 60    
    for x in test_X:    
        plt.scatter(x[0], x[1], marker = tmarker, c=color[
                            model_netengineer.predict(x)], s = size)
    
# Group 2    
    fig += 1        
    
