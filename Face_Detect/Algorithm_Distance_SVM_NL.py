# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:24:23 2016

@author: forest.deng
"""

import numpy as lib_np
import Utility_FaceDetect as ufd
import cv2 as lib_cv2
#from sklearn.decomposition import PCA
from sklearn import svm
import cPickle as lib_cp
import string as lib_str
import neurolab as lib_nl

from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA


'''
'  Parameters - Common
'''
resize_hight = 30
resize_width = 30

pca_components = 8
degree_kernal = 3
gamma_s = 8

similarity_limit = 0.75

# euclidean distance parameter
eudistance = 1500

# setup font style
font = lib_cv2.FONT_HERSHEY_PLAIN

test_result_folder = r"./test_face/result/"
pers_folder = 'persistence'

word_notfound = 'Not Found'
exts = ['.jpg', '.bmp', '.png', '.gif']


'''
'  Parameters - SVM Algorithms
'''

kernal_fun = 'rbf'

trainedX_pers_name_svc = r'/trained_X_SVC.txt'
trainedY_pers_name_svc = r'/trained_Y_SVC.txt'
model_pers_name_svc = r'/model_SVC.txt'


'''
'  Parameters - Neuro Algorithms
'''
layer_s1 = 300
layer_s2 = 150
layer_s3 = 40
epoch_time = 1000

trainedX_pers_name_neuro = r'/trained_X_neuro.txt'
trainedY_pers_name_neuro = r'/trained_Y_neuro.txt'
model_pers_name_neuro = r'/model_neuro.txt'
mapping_pers_name_neuro = r'/mapping_neuro.txt'


'''
'  Parameter - OpenCV Algorithms
'''
resize_hight_cv = 100
resize_width_cv = 100

confidence_limit = 2000

trainedX_pers_name_cv = r'/trained_X_cv.txt'
trainedY_pers_name_cv = r'/trained_Y_cv.txt'
model_pers_name_cv = r'/model_cv.xml'
mapping_pers_name_cv = r'/mapping_cv.txt'



'''------------------------------------------------------------
'  SVM Algorithms
'  ------------------------------------------------------------
'''

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
#    trained_X = get_pca_data(imgs)   
    trained_X = get_pca_data(imgs)            
    trained_X = lib_np.array(trained_X)
    trained_Y = lib_np.array(trained_Y)
    
    param_grid = {'C': [5e3, 1e4, 5e4, 1e5],
         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1], }
 
    svc = GridSearchCV(svm.SVC(kernel='rbf', 
                               class_weight='auto',
                               probability = True), 
                               param_grid)
    #clf = clf.fit(X_train_pca, y_train)
    
    #svc = svm.SVC(
    #                kernel = kernal_fun, 
    #                probability = True, 
    #                gamma = gamma_s
    #                )
    
    model = svc.fit(trained_X, trained_Y)
    print model.best_estimator_

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
            similarity = model.predict_proba(test_X)[0]
#            index = dict(zip(model.best_estimator_.classes_, similarity[0]))
            results_by_probability = map(
                       lambda x: (x[0],x[1]), 
                       sorted(zip(model.best_estimator_.classes_, similarity), 
                       key=lambda x: x[1], reverse=True))
            #similarity = model.score(test_X, test_Y)
            #print similarity
            returnY.append(test_Y)    
            pic_retangle(image, face)
            if results_by_probability[0][1] >= 0.1:
                words = test_Y[0]
            else:
                words = word_notfound

            if image.shape[0] > 800:
                pic_inputtext(image, face, words, 1.5, 1)
            else:
                pic_inputtext(image, face, words, 0.9, 1)
            result_list.append((list(test_Y)[0], results_by_probability))
            #else:
            #    if image.shape[0] > 800:
            #        pic_inputtext(image, face, word_notfound, 3, 2)
            #    else:    
            #        pic_inputtext(image, face, word_notfound, 0.9, 1)
            #    result_list.append(word_notfound)                
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
        lib_cv2.imwrite(
                        resultDir + 
                        ufd.get_fileShortNamewithext(fileList[i]), 
                        image
                        )
    return zip(fn_list, face_list, result_list), returnX, returnY    


''' ----------------------------------------------------------
'   Neuro Algorithms
'   ----------------------------------------------------------
''' 

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
        mapping = data_persistence_decode_neuromapping(
                                                    mapping_pers_name_neuro,
                                                    trainDir                                                                                                            
                                                        )
        model, trained_X, trained_Y = data_persistence_decode(
                                                    model_pers_name_neuro,
                                                    trainedX_pers_name_neuro,
                                                    trainedY_pers_name_neuro,                                        
                                                    trainDir, 
                                                    persistence_folder
                                                            )                
        return model, mapping, trained_X, trained_Y                                                    


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
#    net = lib_nl.net.newc(netminmax, cat_no)
    net = lib_nl.net.newff(netminmax, [layer_s1, cat_no])
#    net = lib_nl.net.newelm(netminmax, [layer_s1, cat_no])
    net.trainf = lib_nl.train.train_gdx
#    net.trainf = lib_nl.train.train_bfgs
    
    err = net.train(trained_X, trained_Y, epochs = epoch_time, lr = 10)        
#    err = net.train(trained_Y)        


# data persistence

    data_persistence_encode(
                            net, trained_X, 
                            trained_Y,
                            model_pers_name_neuro, 
                            trainedX_pers_name_neuro,
                            trainedY_pers_name_neuro,
                            trainDir
                            )
    data_persistence_encode_neuromapping(
                                        mapping,
                                        mapping_pers_name_neuro,
                                        trainDir
                                        )
                            
    print 'trained error: {0} '.format(err)
    
    return net, mapping, trained_X, trained_Y

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
        for j in range(faces_len):
            img = imgs[img_index]
            fn = nameList[i]
            face = facelist[i][j]
            test_X = lib_np.array(get_pca_datum(img))
            returnX.append(test_X)
            test_Y = model.sim(test_X)
            test_Y, similarity = get_trained_Y_textvalue(test_Y, mapping)
            returnY.append(test_Y)    
            pic_retangle(image, face)

            if image.shape[0] > 800:
                pic_inputtext(image, face, test_Y, 2, 2)
            else:
                pic_inputtext(image, face, test_Y, 0.9, 1)
            result_list.append([test_Y,similarity])
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
            test_X = None
            test_Y = None
            img = None
        lib_cv2.imwrite(
                        resultDir + 
                        ufd.get_fileShortNamewithext(fileList[i]), 
                        image
                        )
    return zip(fn_list, face_list, result_list), returnX, returnY    



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
    if y_array[ymax_index] < similarity_limit:
        return word_notfound, y_array[ymax_index] 
    for m in mapping.keys():
        mmax_index = lib_np.array(mapping.get(m)).argmax(0)
        if ymax_index == mmax_index:
            return m, y_array[ymax_index]


def data_persistence_encode_neuromapping(
                                    mapping,
                                    mapping_pfile, 
                                    trainDir, 
                                    persistence_folder = pers_folder
                                    ):
    """
    persist data
    """
    mapping_file = open(
                        trainDir + 
                        persistence_folder + 
                        mapping_pfile, 
                        'w'
                        )
    lib_cp.dump(mapping, mapping_file)
    mapping_file.close()

def data_persistence_decode_neuromapping(
                                    mapping_pfile, 
                                    trainDir, 
                                    persistence_folder = pers_folder
                                    ):
    """
    restore data
    """
    mapping_file = open(
                        trainDir + 
                        persistence_folder + 
                        mapping_pfile
                        )
    mapping = lib_cp.load(mapping_file)
    mapping_file.close()
    return mapping



'''-------------------------------------------------------
'  OpenCV face reg Algorithms (eigon, fisher or LBPH)
'  -------------------------------------------------------
'''

def get_trained_model_data_wrap_cv(trainDir, data_generated, 
                                persistence_folder = pers_folder):
    """
    """
    if data_generated:
        return get_trained_model_data_cv(
                                            trainDir, 
                                            persistence_folder
                                            )
    else:
        mapping = data_persistence_decode_neuromapping(
                                                    mapping_pers_name_cv,
                                                    trainDir                                                                                                            
                                                        )
        model = lib_cv2.createEigenFaceRecognizer()
        model.load(trainDir + persistence_folder + model_pers_name_cv)
        model_none, trained_X, trained_Y = data_persistence_decode(
                                                    model_pers_name_cv,
                                                    trainedX_pers_name_cv,
                                                    trainedY_pers_name_cv,                                        
                                                    trainDir, 
                                                    persistence_folder
                                                            )                
        return model, mapping, trained_X, trained_Y                                                    

def get_trained_model_data_cv (    
                                    trainDir, 
                                    persistence_folder = pers_folder
                              ):
    """
    """
    imgs, nameList = ufd.img_batch_read(trainDir)
    
    trained_Y = [lib_str.split(e, '__')[-2] for e in nameList]
    trained_Y, mapping = get_trained_Y_cv(trained_Y)
    
    
    trained_X = get_img_data_cv(imgs)   
        
    trained_X = lib_np.array(trained_X)
    trained_Y = lib_np.array(trained_Y)
    
    model = lib_cv2.createEigenFaceRecognizer()
#    model = lib_cv2.createFisherFaceRecognizer()    
    model.train(trained_X, trained_Y)
    
    
    model.save(trainDir + persistence_folder + model_pers_name_cv)
    
    data_persistence_encode(
                            None, trained_X, 
                            trained_Y,
                            model_pers_name_cv, 
                            trainedX_pers_name_cv,
                            trainedY_pers_name_cv,
                            trainDir
                            )
    data_persistence_encode_neuromapping(
                                        mapping,
                                        mapping_pers_name_cv,
                                        trainDir
                                        )
    

    return model, mapping, trained_X, trained_Y




def get_test_result_cv(
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
        for j in range(faces_len):
            img = imgs[img_index]
            fn = nameList[i]
            face = facelist[i][j]
            test_X = get_img_datum_cv(img)
            returnX.append(test_X)
            test_Y, confidence = model.predict(test_X)
            if confidence <= confidence_limit:
                test_Y = get_trained_Y_cv_textvalue(test_Y, mapping)
            else:
                test_Y = word_notfound
            returnY.append(test_Y)    
            pic_retangle(image, face)
            if image.shape[0] > 800:
                pic_inputtext(image, face, test_Y, 1.3, 2)
            else:
                pic_inputtext(image, face, test_Y, 0.9, 1)
            result_list.append([test_Y,confidence])
            img_index += 1
            fn_list.append(fn)
            face_list.append(face)            
            test_X = None
            test_Y = None
            img = None
        lib_cv2.imwrite(
                        resultDir + 
                        ufd.get_fileShortNamewithext(fileList[i]), 
                        image
                        )
    return zip(fn_list, face_list, result_list), returnX, returnY    


def get_trained_Y_cv(trained_Y):
    """
    """
    y_set = set(trained_Y)
    y_len = len(y_set)
    y_set_code = range(y_len)

    y_conv = []
    mapping = dict(zip(y_set, y_set_code))
    
    for y in trained_Y:
        y_conv.append(mapping.get(y))        
    
    return y_conv, mapping    

def get_trained_Y_cv_textvalue(
                            trained_Y, 
                            mapping
                            ):
    """
    """
    mapping_1 = dict(zip(mapping.values(),mapping.keys()))
    
    return mapping_1.get(trained_Y)     
    

    y_array = trained_Y[0]
    ymax_index = y_array.argmax(0)
    if y_array[ymax_index] < similarity_limit:
        return word_notfound, y_array[ymax_index] 
    for m in mapping.keys():
        mmax_index = lib_np.array(mapping.get(m)).argmax(0)
        if ymax_index == mmax_index:
            return m, y_array[ymax_index]


def get_img_data_cv(
                    imgs, 
                    hight = resize_hight_cv, 
                    width = resize_hight_cv 
                    ):
    """
    """
    reImg =[]
    newsize = (hight, width)
        
    for img in imgs:
        rimg = lib_cv2.cvtColor(img, lib_cv2.COLOR_BGR2GRAY)
        rimg = lib_cv2.resize(rimg, newsize)
        reImg.append(lib_np.asarray(rimg, lib_np.uint8))        

    return reImg

def get_img_datum_cv(
                    img, 
                    hight = resize_hight_cv, 
                    width = resize_hight_cv 
                    ):
    """
    """
    newsize = (hight, width)
    rimg = lib_cv2.cvtColor(img, lib_cv2.COLOR_BGR2GRAY)
    rimg = lib_cv2.resize(rimg, newsize, interpolation = lib_cv2.INTER_LINEAR)
    
    return rimg


'''-------------------------------------------------------
' Common Functions
'  -------------------------------------------------------
'''

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
    if model != None:
        model_file = open(
                    trainDir + 
                    persistence_folder +
                    model_file, 
                    'w'
                    )                 
        lib_cp.dump(model, model_file)            
        model_file.close()
                            
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
   
    lib_cp.dump(trained_X, train_X_file)
    lib_cp.dump(trained_Y, train_Y_file)
    train_X_file.close()
    train_Y_file.close()


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
    trained_X = lib_cp.load(train_X_file)
    trained_Y = lib_cp.load(train_Y_file)
    train_X_file.close()
    train_Y_file.close()

    try:
        model_file = open(
                            trainDir + 
                            persistence_folder +
                            model_file
                            )                  
                            
        model = lib_cp.load(model_file)
        model_file.close()
    except:
        model_file.close()
        model = None

    return model, trained_X, trained_Y



def get_euclideandistance(x, y):
    """
    calculate euclidean distance between two vectors
    """
    x_array = lib_np.array(x)
    y_array = lib_np.array(y)
    
    return lib_np.sqrt(sum((x_array - y_array)*(x_array - y_array)))
    

def get_pca_data(
                    imgs, 
                    hight = resize_hight, 
                    width = resize_width 
                    ):
    """
    """
    reImg =[]
    newsize = (hight, width)
        
    pca = RandomizedPCA(n_components = pca_components, whiten = True)
    
    for img in imgs:
        rimg = lib_cv2.cvtColor(img, lib_cv2.COLOR_BGR2GRAY)
        rimg = lib_cv2.resize(rimg, newsize)
#        reImg.append(pca.fit_transform(rimg.ravel()).tolist())        
        reImg.append(pca.fit_transform(rimg).T.ravel().tolist())
        print pca.explained_variance_ratio_.cumsum()

    return reImg


def get_pca_data_batch(
                    imgs, 
                    hight = resize_hight, 
                    width = resize_width 
                    ):
    """
    """
    newsize = (hight, width)

    rImgs = [lib_cv2.resize(e, newsize) for e in imgs]
    rImgs = [lib_cv2.cvtColor(e, lib_cv2.COLOR_BGR2GRAY) for e in rImgs]
    rImgs = [e.ravel() for e in rImgs]
        
    pca = RandomizedPCA(n_components = 200, whiten = True)
    
    pImgs = pca.fit_transform(rImgs)    
    
    return pImgs

def get_pca_datum(
                    img, 
                    hight = resize_hight, 
                    width = resize_width 
                    ):
    """
    """
    img_c = []
    newsize = (hight, width)
    rimg = lib_cv2.cvtColor(img, lib_cv2.COLOR_BGR2GRAY)
    rimg = lib_cv2.resize(rimg, newsize)
    
#    img_c.append(rimg.ravel().tolist())
#    return img_c

    pca = RandomizedPCA(n_components = pca_components, whiten = True)
#    img_c.append(pca.fit_transform(rimg.ravel()).tolist())        

    img_c.append(pca.fit_transform(rimg).T.ravel().tolist())   
    #print pca.explained_variance_ratio_.cumsum()
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