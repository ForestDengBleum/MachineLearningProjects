# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:56:21 2016

@author: forest.deng
"""
import re as lib_re
import numpy as lib_np
import string as lib_str
#import stop_words as sw
import Utility_RemoveHtmlTags as rht
from stop_words import get_stop_words
from enchant.checker import SpellChecker

#def sigmoid(num):
#    """"
#    """"

def control_nounicode(textInput):
    """
    """
 #   return re.sub(r'[\u4e00-\u9fff]','?', textInput)
    cp0 = lib_re.compile(r'application/vnd.ms-officetheme(.|\n)*NextPart.*')
    textInput = cp0.sub(' ',textInput)
#    cp1 = re.compile(r'[\W+]', re.UNICODE)
    cp2 = lib_re.compile(r'([\x81-\xff]|\x07)')
    return cp2.sub(' ', textInput)
#    return textInput.encode('unicode','replace')        
    

def translate_non_alphanumerics(to_translate, translate_to='_'):
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = lib_str.maketrans(not_letters_or_digits,
                                       translate_to
                                        *len(not_letters_or_digits))
    translate_table = translate_table.decode("latin-1")
    return to_translate.translate(translate_table)

def control_lowercase(textInput):
    """
    """
    return lib_str.lower(textInput)

#def control_nopunctuations(textInput):
#    """
#    """
    

def control_nohtmltags(textInput):
    """
    """
    return rht.filter_tags(textInput)

def control_nonumber(textInput):
    """
    """
    return lib_re.sub(r'[\d+]',' ', textInput)

def control_nopunctuation(textInput):
    """
    """
    textInput = textInput.encode('ascii','xmlcharrefreplace')
    
    filterPunc = list(lib_str.punctuation)
    filterPunc.remove('.')
    filterPunc.remove('#')
    filterPunc.remove('/')
    filterStr = ''.join(filterPunc)
        
    
    rep = ' '*len(filterStr)
    strResult = textInput.translate(lib_str.maketrans(filterStr,rep))
    strResult = lib_re.sub(r'\.(?!([nN][eE][tT]))',' ', strResult)
    strResult = lib_re.sub(r'(?<![cC])#',' ', strResult)
    strResult = lib_re.sub(r'(\/(?![Ss]))|((?<![cCBb])\/)|\
                        ((?<![cCBb])\/(?![Ss]))', ' ', strResult)
    return strResult

def control_nospecchar(textInput):
    """
    """
    excl = ''.join(['\n','\t','\r','\b','\v','\f'])
    repl = ''.join([' ',' ',' ',' ',' ',' '])
    return textInput.translate(lib_str.maketrans(excl,repl))

def control_nocommonwords(textInput):
    """
    """
    
    stop_words = get_stop_words('english')
    #textInput.encode('latin-1')
    mystop_wordsfile = open('Utility_MyStopWords.txt','r')
    mystop_words = list(lib_str.split(lib_re.sub('\n',' ',  
                        mystop_wordsfile.read()),' '))
    mystop_wordsfile.close()
    stop_words.extend(mystop_words)
    words = control_lowercase(textInput)
    for e in stop_words:
        words = lib_re.sub('(\s+' + e + '\s)',' ', words)
    return words
#    stop_words = ['\s' + e + '\s' for e in stop_words]
#    rep = len(stop_words)/100.0
#    if len(stop_words)/100.0 > len(stop_words)/100:
#        rep = len(stop_words)/100 + 1
#    else:
#        rep = len(stop_words)/100    
#    for i =    
#    patterns = '|'.join(stop_words)
    #patterns.join()
#    return re.sub(patterns,' ', words)    

def control_noerrorwords(textInput,jdlist):
    """
    """
    words =[]
    checker = SpellChecker('en_US')
    checker.set_text(textInput)
    for errorword in checker:
        words.append(errorword.word)
    words = list(set(words))
    crossWords = set(words).difference(set(jdlist))
    count = len(crossWords)
    cvtext = textInput

    pWordsFile = open('Utility_MyWords.txt','r')
    pWords = set(list(lib_str.split(lib_re.sub('\n',' ', \
                    pWordsFile.read()),' ')))
    pWordsFile.close()
    
    crossWords = crossWords.difference(pWords)
    
    if count == 0:
        for e in words:
            cvtext = lib_re.sub('(\s+' + e + '\s)',' ', cvtext)
    else:
        for e in crossWords:
            cvtext = lib_re.sub('(\s+' + e + '\s)',' ', cvtext)
    return cvtext
        
def getstringlist(textInput):
    """
    """
    wordsArray = lib_np.array(list(lib_str.split(textInput,' ')))
    wordsArray = list(wordsArray[wordsArray<>''])
    for e in wordsArray[:]:
        if len(e)==1: 
            wordsArray.remove(e)
    return wordsArray

def datapurse_general(textInput):
    """
    """
    # html tag removing
    stringPurse = control_nohtmltags(textInput)
    # non-alpha character
    stringPurse = control_nounicode(stringPurse)
    # punctuation removing
    stringPurse = control_nopunctuation(stringPurse)
    # special char removing
    stringPurse = control_nospecchar(stringPurse)
    stringPurse = control_lowercase(stringPurse)    
    # common words removing
    stringPurse = control_nocommonwords(stringPurse)    
    # get list and return
    # number removing
    stringPurse = control_nonumber(stringPurse)    

    return stringPurse
 
def datapurse_cv(cvtext, jdtext):
    """
    """
    cvWords = datapurse_general(cvtext)
    jdlist = list(set(getstringlist(datapurse_general(jdtext))))
    
    return getstringlist(control_noerrorwords(cvWords,jdlist)), \
                            getstringlist(datapurse_general(jdtext))        

def getJDList(jdtext):
    """
    """
    return getstringlist(datapurse_general(jdtext))    
    
def datapurse_collection_cv(cvCollectionList, jdlist):
    """
    """
    pursedlist = []
    for cvtext in cvCollectionList:
        cvWords = datapurse_general(cvtext)
        pursedlist.append(getstringlist(control_noerrorwords
                            (cvWords,list(set(jdlist)))))
        
    return pursedlist

def getJDTitlelist(jdTitleText):
    """
    """
    pursed_jdTitle = jdTitleText[:-len(lib_str.split(jdTitleText,'.')[-1])]  
    pursed_jdTitle = datapurse_general(pursed_jdTitle)
    pursed_jdTitle = lib_re.sub(r'jd\s',' ', pursed_jdTitle)
    pursed_jdTitle = lib_re.sub(r'(i{1,3}\s)',' ', pursed_jdTitle)
    pursed_jdTitle = lib_re.sub(r'(vi|iv)\s',' ', pursed_jdTitle)
    return getstringlist(pursed_jdTitle)    
    
   