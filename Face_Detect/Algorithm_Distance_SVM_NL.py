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
import string as lib_str
import neurolab as lib_nl

# picture resize parameter 
resize_hight = 35
resize_width = 35

# pca component parameter to setup reduced dimensions
pca_components = 5

# svm kernel function selection
kernal_fun = 'rbf'
degree_kernal = 3
gamma_s = 8


# neuro parameter
layer_s1 = 20
layer_s2 = 5

# euclidean distance parameter
eudistance = 1500

# setup font style
font = lib_cv2.FONT_HERSHEY_PLAIN

# folders
test_result_folder = r"./test_face/result/"
trainedX_pers_name_svc = r'/trained_X_SVC.txt'
trainedY_pers_name_svc = r'/trained_Y_SVC.txt'
model_pers_name_svc = r'/model_SVC.txt'

trainedX_pers_name_neuro = r'/trained_X_neuro.txt'
trainedY_pers_name_neuro = r'/trained_Y_neuro.txt'
model_pers_name_neuro = r'/model_neuro.txt'

pers_folder = 'persistence'


# not found words
word_notfound = 'Not Found'

exts = ['.jpg', '.bmp', '.png', '.gif']


def get_euclideandistance(x, y):
    """
    calculate euclidean distance between two vectors
    """
    x_array = lib_np.array(x)
    y_array = lib_np.array(y)
    
    return lib_np.sqrt(sum((x_array - y_array)*(x_array - y_array)))
    

def get_trained_model_data(
                            trainDir, 
                            persistence_folder = pers_folder
                            ):
    """
    1. get svm.SVM trained model
    2. data persistence using cPickle
    """
    imgs, nameList = ufd.img_batch_read(trainDir)
    
    trained_Y = [lib_str.split(e, '__')[-2] for e in nameList]
    trained_X = get_pca_data(imgs)   
        
    trained_X = lib_np.array(trained_X)
    trained_Y = lib_np.array(trained_Y)
    
    svc = svm.SVC(
                    kernel = kernal_fun, 
                    probability = True, 
                    gamma = gamma_s
                    )
    
    model = svc.fit(trained_X, trained_Y)

# data persistence

    data_persistence_encode(
                            model, trained_X, 
                            trained_Y,
                            model_pers_name_svc, 
                            trainedX_pers_name_svc,
                            trainedY_pers_name_svc,
                            trainDir
                            )
    
    return model, trained_X, trained_Y


def get_trained_model_data_wrap(
                                trainDir, 
                                data_generated, 
                                persistence_folder = pers_folder
                                ):
    """
    provide uniform interface to call SVM.svc algorithm
    """
    if data_generated:
        return get_trained_model_data(
                                        trainDir, 
                                        persistence_folder
                                        )
    else:
        return data_persistence_decode(
                                        model_pers_name_svc,
                                        trainedX_pers_name_svc,
                                        trainedY_pers_name_svc,                                        
                                        trainDir, 
                                        persistence_folder
                                        )                

# for neuro, convert Y to requirement format

def get_trained_Y_neuromatrix(trained_Y):
    """
    """
    y_set_code = []
    y_set = set(trained_Y)
    y_len = len(y_set)
    y_conv = []
    for i in range(y_len):
        temp = lib_np.zeros((1,y_len), dtype = int)
        temp[0, -(i+1)] = 1
        y_set_code.append(list(temp[0]))    
    mapping = dict(zip(y_set, y_set_code))
    
    for y in trained_Y:
        y_conv.append(mapping.get(y))        
    
    return y_conv, mapping    

# for neuro, convert Y back to text    
def get_trained_Y_textvalue(
                            trained_Y, 
                            mapping
                            ):
    """
    """
    y_array = trained_Y[0]
    ymax_index = y_array.argmax(0)
    if y_array[ymax_index] < 0.7:
        return word_notfound, y_array[ymax_index] 
    for m in mapping.keys():
        mmax_index = lib_np.array(mapping.get(m)).argmax(0)
        if ymax_index == mmax_index:
            return m, y_array[ymax_index]

def data_persistence_encode(
                            model, 
                            trained_X, 
                            trained_Y,
                            model_file, 
                            trained_X_file, 
                            trained_Y_file,
                            trainDir, 
                            persistence_folder = pers_folder
                            ):
    """
    persist data
    """
    train_X_file = open(
                        trainDir + 
                        persistence_folder + 
                        trained_X_file, 
                        'w'
                        )
    train_Y_file = open(
                        trainDir + 
                        persistence_folder + 
                        trained_Y_file, 
                        'w'
                        )
    model_file = open(
                        trainDir + 
                        persistence_folder +
                        model_file, 
                        'w'
                        )                    
    lib_cp.dump(trained_X, train_X_file)
    lib_cp.dump(trained_Y, train_Y_file)
    lib_cp.dump(model, model_file)            
    train_X_file.close()
    train_Y_file.close()
    model_file.close()

def data_persistence_decode(
                            model_file, 
                            trained_X_file, 
                            trained_Y_file, 
                            trainDir, 
                            persistence_folder = pers_folder
                            ):
    """
    restore data
    """
    train_X_file = open(
                        trainDir + 
                        persistence_folder + 
                        trained_X_file
                        )
    train_Y_file = open(
                        trainDir + 
                        persistence_folder + 
                        trained_Y_file
                        )
    model_file = open(
                        trainDir + 
                        persistence_folder +
                        model_file
                        )                  
    trained_X = lib_cp.load(train_X_file)
    trained_Y = lib_cp.load(train_Y_file)
    model = lib_cp.load(model_file)
    train_X_file.close()
    train_Y_file.close()
    model_file.close()
    return model, trained_X, trained_Y
        
def get_trained_model_data_neuro(
                                    trainDir, 
                                    persistence_folder = pers_folder
                                    ):
    """
    """
    imgs, nameList = ufd.img_batch_read(trainDir)
    
    trained_Y = [lib_str.split(e, '__')[-2] for e in nameList]
    trained_Y, mapping = get_trained_Y_neuromatrix(trained_Y)
    trained_X = get_pca_data(imgs)   
        
    trained_X = lib_np.array(trained_X)
    trained_Y = lib_np.array(trained_Y)
    cat_no = len(mapping)
    netminmax = zip(lib_np.min(trained_X,0), lib_np.max(trained_X,0))
    netminmax = [list(e) for e in netminmax]
    
# create neuro network
#    net = lib_nl.net.newff(netminmax, [layer_s1, layer_s2, cat_no])
    net = lib_nl.net.newff(netminmax, [layer_s1, cat_no])
   
    err = net.train(trained_X, trained_Y)        

# data persistence

    data_persistence_encode(
                            net, trained_X, 
                            trained_Y,
                            model_pers_name_neuro, 
                            trainedX_pers_name_neuro,
                            trainedY_pers_name_neuro,
                            trainDir
                            )
                            
    print 'trained error: {0} '.format(err)
    
    return net, mapping, trained_X, trained_Y

def get_trained_model_data_wrap_neuro(trainDir, data_generated, 
                                persistence_folder = pers_folder):
    """
    """
    if data_generated:
        return get_trained_model_data_neuro(
                                            trainDir, 
                                            persistence_folder
                                            )
    else:
        return data_persistence_decode(
                                        model_pers_name_neuro,
                                        trainedX_pers_name_neuro,
                                        trainedY_pers_name_neuro,                                        
                                        trainDir, 
                                        persistence_folder
                                        )                



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

def pic_retangle(image, face):
    """
    """
    lib_cv2.rectangle(
                        image, 
                        (face[0], face[1]), 
                        (face[0] + face[2], face[1] + face[3]), 
                        (0, 255, 0), 
                        2
                        )

def pic_inputtext(
                    image, 
                    face, 
                    text, 
                    size, 
                    thick):
    """
    """
    lib_cv2.putText(
                    image, 
                    text, 
                    (face[0] -3 , face[1] - 3), 
                    font, 
                    size,  
                    (0, 0, 255), 
                    thick
                    )
    
    

def get_test_result(
                    testDir, 
                    model, 
                    centroids, 
                    resultDir = test_result_folder
                    ):
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
        test_X = None
        test_Y = None
        for j in range(faces_len):
            img = imgs[img_index]
            fn = nameList[i]
            face = facelist[i][j]
            test_X = lib_np.array(get_pca_datum(img))
            returnX.append(test_X)
            test_Y = model.predict(test_X)
            returnY.append(test_Y)    
# test result by euclidean distance
            reImg = centroids.get(test_Y[0])
            pic_retangle(image, face)
            if len(test_Y) != 0:                        
                dis = get_euclideandistance(test_X[0], reImg)
                if dis < eudistance:
                    pic_inputtext(image, face, test_Y[0], 0.9, 1)
                    result_list.append(list(test_Y)[0])
                else:
                    pic_inputtext(image, face, word_notfound, 0.9, 1)
                    result_list.append(word_notfound)                
            else:
                pic_inputtext(image, face, word_notfound, 0.9, 1)
                result_list.append(word_notfound)                
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
        lib_cv2.imwrite(
                        resultDir + 
                        ufd.get_fileShortNamewithext(fileList[i]), 
                        image
                        )
    return zip(fn_list, face_list, result_list), returnX, returnY    

def get_test_result_neuro(
                            testDir, 
                            model, 
                            mapping, 
                            resultDir = test_result_folder
                            ):
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
        test_X = None
        test_Y = None
        for j in range(faces_len):
            img = imgs[img_index]
            fn = nameList[i]
            face = facelist[i][j]
            test_X = lib_np.array(get_pca_datum(img))
            returnX.append(test_X)
            test_Y = model.sim(test_X)
            test_Y, similarity = get_trained_Y_textvalue(test_Y, mapping)
            returnY.append(test_Y)    
            lib_cv2.rectangle(
                                image, 
                                (face[0], face[1]), 
                                (face[0] + face[2], face[1] + face[3]), 
                                (0, 255, 0), 
                                2
                                )
            lib_cv2.putText(
                            image, 
                            test_Y, 
                            (face[0] -3 , face[1] - 3), 
                            font, 
                            0.9,  
                            (0, 0, 255), 
                            1
                            )
            result_list.append([test_Y,similarity])
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
        lib_cv2.imwrite(
                        resultDir + 
                        ufd.get_fileShortNamewithext(fileList[i]), 
                        image
                        )
    return zip(fn_list, face_list, result_list), returnX, returnY    


        
def get_train_img_by_name(
                            sfn, 
                            folder = ufd.train_face_folder
                            ):
    """
    """
    for ext in exts:
        img = lib_cv2.imread(folder + sfn + ext)
        if img != None:    
            return get_pca_datum(img)
  
           
def get_category_centroid(trained_X, trained_Y):
    """
    """
    centroids = []
    categories = set(trained_Y)
    for cat in categories:
        centroids.append(sum(trained_X[trained_Y == cat])/float(len
                                        (trained_X[trained_Y == cat])))    
    return dict(zip(categories, centroids))
        
        
        
    
    
    