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
import cPickle as lib_cp

resize_hight = 50
resize_width = 50
pca_components = 5
kernal_fun = 'rbf'
degree_kernal = 3
gamma = 'auto'
eudistance = 100
font = lib_cv2.FONT_HERSHEY_PLAIN
test_result_folder = r"./test_face/result/"
exts = ['.jpg', '.bmp', '.png', '.gif']


def get_euclideandistance(x, y):
    """
    """
    x_array = lib_np.array(x)
    y_array = lib_np.array(y)
    
    return lib_np.sqrt(sum((x_array[0] - y_array[0])*(x_array[0] - y_array[0])))
    

def get_trained_model_data(trainDir, persistence_folder = 'persistence'):
    """
    """
    imgs, nameList = ufd.img_batch_read(trainDir)
    
    trained_Y = nameList
    trained_X = get_pca_data(imgs)   
        
    trained_X = lib_np.array(trained_X)
    trained_Y = lib_np.array(trained_Y)
    
    svc = svm.SVC(kernel = kernal_fun, degree = degree_kernal)
    
    model = svc.fit(trained_X, trained_Y)

# data persistence

    train_X_file = open(trainDir + persistence_folder + 
                        '\\trained_X.txt', 'w')
    train_Y_file = open(trainDir + persistence_folder + 
                        '\\trained_Y.txt', 'w')
    lib_cp.dump(trained_X, train_X_file)
    lib_cp.dump(trained_Y, train_Y_file)            
    train_X_file.close()
    train_Y_file.close()

    return model, trained_X, trained_Y

def get_trained_model_data_wrap(trainDir, data_generated, 
                                persistence_folder = 'persistence'):
    """
    """
    if data_generated:
        return get_trained_model_data(trainDir, persistence_folder)
    else:
        train_X_file = open(trainDir + persistence_folder + 
                        '\\trained_X.txt')
        train_Y_file = open(trainDir + persistence_folder + 
                        '\\trained_Y.txt')
        trained_X = lib_cp.load(train_X_file)
        trained_Y = lib_cp.load(train_Y_file)
        train_X_file.close()
        train_Y_file.close()
                
        svc = svm.SVC(kernel = kernal_fun, gamma = gamma)
        
        model = svc.fit(trained_X, trained_Y)
        return model, trained_X, trained_Y            
        


def get_pca_data(imgs):
    """
    """
    reImg =[]
    newsize = (resize_hight, resize_width)
        
    pca = PCA(n_components = pca_components)
    
    for img in imgs:
        rimg = lib_cv2.resize(img, newsize)
        rimg = lib_cv2.cvtColor(rimg, lib_cv2.COLOR_BGR2GRAY)
        reImg.append(pca.fit_transform(rimg).ravel().tolist())

    return reImg

def get_pca_datum(img):
    """
    """
    img_c = []
    newsize = (resize_hight, resize_width)
    rimg = lib_cv2.resize(img, newsize)
    rimg = lib_cv2.cvtColor(rimg, lib_cv2.COLOR_BGR2GRAY)

    pca = PCA(n_components = pca_components)
    img_c.append(pca.fit_transform(rimg).ravel().tolist())     
    return img_c    
    
def get_test_result(testDir, model, resultDir = test_result_folder):
    """
    """
    fileList = list(ufd.list_allfiles(testDir))    
    imgs, nameList, facelist = ufd.face_detect_batch_returnimg(testDir)
    
    img_index = 0
    facelist_len = len(facelist)
    fn_list = []
    face_list = []
    result_list = []
    returnX = []
    returnY = []
    
    for i in range(facelist_len):
        faces_len = len(facelist[i])
        image = lib_cv2.imread(fileList[i])
        for j in range(faces_len):
            img = imgs[img_index]
            fn = nameList[i]
            face = facelist[i][j]
            test_X = lib_np.array(get_pca_datum(img))
            returnX.append(test_X)
            test_Y = model.predict(test_X)
            returnY.append(test_Y)    
# test result by euclidean distance
            reImg = get_train_img_by_name(test_Y[0])
            dis = get_euclideandistance(test_X, reImg)
            lib_cv2.rectangle(image, (face[0], face[1]), 
                (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)                        
            if dis < eudistance:
                lib_cv2.putText(image, test_Y[0], (face[0] -3 , face[1] - 3), 
                                font, 0.9,  (0, 0, 255), 1)
                result_list.append(list(test_Y)[0])
            else:
                lib_cv2.putText(image, 'Not Found', (face[0] - 3, face[1] - 3), 
                                font, 0.9, (100, 100, 255), 1)        
                result_list.append('Not Found')                
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
        lib_cv2.imwrite(resultDir + ufd.get_fileShortNamewithext(fileList[i])
                   , image)
    return zip(fn_list, face_list, result_list), returnX, returnY    
        
def get_train_img_by_name(sfn, folder = ufd.train_face_folder):
    """
    """
    for ext in exts:
        img = lib_cv2.imread(folder + sfn + ext)
        if img != None:    
            return get_pca_datum(img)
  
           
    
        
        
        
    
    
    