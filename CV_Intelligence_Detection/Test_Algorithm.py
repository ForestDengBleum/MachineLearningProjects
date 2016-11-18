# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:41:54 2016

@author: forest.deng
"""
import Train_Features as tf
import matplotlib.pyplot as plt


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
    print 'SVC Model - coef_: %s' %model_netengineer.coef_
    print 'SVC Model - Intercept_: %s' %model_netengineer.intercept_
    plt.figure(fig)
    plt.title('Train/Test data distribution for Role: ' + role_name )
    plt.xlabel('JD similarity')
    plt.ylabel('Centroid similarity')
    marker ='os'
    color ='by'
    for x in trained_X:
        if model_netengineer.predict(x) == 1:
            plt.scatter(x[0],x[1], s = 40)
        else:
            plt.scatter(x[0],x[1], s = 40, marker ='s', c='g')

    
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
    
    plt.scatter(test_X[:,0], test_X[:,1], marker='^', c='g', s = 40)
    
    
# Group 2    
    fig += 1        
    
