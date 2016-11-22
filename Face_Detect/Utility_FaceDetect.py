#coding=utf8

import cv2 as lib_cv2
import os as lib_os
import fnmatch as lib_fnmatch

# Get user supplied values
imagePath = 'abba.png'
cascPath = 'haarcascade_frontalface_alt.xml'
train_face_folder = r"./train_face/"
train_face_temp_folder = r"./train_face/temp/"

def face_detect_returnloc(imagePath, cascPath = cascPath):
    """
    """
# Create the haar cascade
    faceCascade = lib_cv2.CascadeClassifier(cascPath)

# Read the image
    image = lib_cv2.imread(imagePath)
    gray = lib_cv2.cvtColor(image, lib_cv2.COLOR_BGR2GRAY)

# Detect faces in the image
    faces = faceCascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.25,
                                        minNeighbors=2,
                                        minSize=(30,30),
                                        flags = lib_cv2.cv.CV_HAAR_SCALE_IMAGE
                                        )

    return faces

def face_detect_returnimg(imagePath, cascPath):
    """
    """
# Create the haar cascade
    faceCascade = lib_cv2.CascadeClassifier(cascPath)

# Read the image
    image = lib_cv2.imread(imagePath)
    gray = lib_cv2.cvtColor(image, lib_cv2.COLOR_BGR2GRAY)

# Detect faces in the image
    faces = faceCascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.25,
                                        minNeighbors=2,
                                        minSize=(30,30),
                                        flags = lib_cv2.cv.CV_HAAR_SCALE_IMAGE
                                        )
    img = []                                    
    for (x, y, w, h) in faces:
        img.append(image[y:y+h, x:x+w, :])        
    
    return img

def face_detect_batch_returnimg(imageDir, cascPath = cascPath):
    """
    """
    fileList = list(list_allfiles(imageDir))
    nameList = [get_fileShortName(e) for e in fileList]
#    faceCascade = lib_cv2.CascadeClassifier(cascPath)
    img = []                                    
    for fl in fileList:
        image = lib_cv2.imread(fl)
#        gray = lib_cv2.cvtColor(image, lib_cv2.COLOR_BGR2GRAY)

#        faces = faceCascade.detectMultiScale(
#                                    gray,
#                                    scaleFactor=1.25,
#                                    minNeighbors=2,
#                                    minSize=(30,30),
#                                    flags = lib_cv2.cv.CV_HAAR_SCALE_IMAGE
#                                            )
#        for (x, y, w, h) in faces:
        img.append(image)        
    
    return img, nameList




def face_saving(imagePath, faces, save_path = train_face_temp_folder):
    """
    """
    image = lib_cv2.imread(imagePath)
    feature = get_fileShortName(imagePath)
        
    index = 0    
    for (x, y, w, h) in faces:
        lib_cv2.imwrite(save_path + 'face_' + feature + '_'
                        +  str(index) + '.jpg', image[y:y+h, x:x+w,:])
        index = index + 1

# Draw a rectangle around the faces
#    for (x, y, w, h) in faces:
#        lib_cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#    lib_cv2.imshow("Faces found", image)
#    lib_cv2.imwrite("face_detection.jpg", image)
    
    
#    lib_cv2.waitKey(0)
#    lib_cv2.destroyAllWindows()

def face_batch_saving(imageDir, save_path = train_face_temp_folder):
    """
    """
    fileList = list(list_allfiles(imageDir))
    for fl in fileList:
        faces = face_detect_returnloc(fl)
        face_saving(fl, faces)
        
    

def list_allfiles(dirName, patterns='*', single_level = True, 
                  yield_folders = False):
    """
    """    
    patterns = patterns.split(';')
    for path, subDirs, files in lib_os.walk(dirName):
        if yield_folders:
            files.extend(subDirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if lib_fnmatch.fnmatch(name, pattern):
                    yield lib_os.path.join(path, name)
                    break
        if single_level:
            break 


def get_fileShortName(fileName):
    """
    """
    if fileName.find('\\') < 0:
        if fileName.find('/') < 0:
            return fileName.split('.')[-2]                
        else:        
            return fileName.split('/')[-1].split('.')[-2]        
    else:
        return fileName.split('\\')[-1].split('.')[-2]        
    