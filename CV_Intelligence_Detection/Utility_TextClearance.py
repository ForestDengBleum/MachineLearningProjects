# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:56:21 2016

@author: forest.deng
"""
import re
import numpy as np
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
    cp0 = re.compile(r'file:\/\/\/.|\n*NextPart')
    s = cp0.match(textInput)
    print s.group(0)
    textInput = cp0.sub(' ',textInput)
    cp1 = re.compile(r'[\W+]', re.UNICODE)
    cp2 = re.compile(r'[\x81-\xff]')
    return cp2.sub(' ',cp1.sub(' ', textInput))
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
    return re.sub(r'[\d+]',' ', textInput)

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
    strResult = re.sub(r'\.(?!([nN][eE][tT]))',' ', strResult)
    strResult = re.sub(r'(?<![cC])#',' ', strResult)
    strResult = re.sub(r'(\/(?![Ss]))|((?<![cCBb])\/)|\
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
    words = control_lowercase(textInput)
    for e in stop_words:
        words = re.sub('(\s+' + e + '\s)',' ', words)
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
    crosswords = set(words).difference(set(jdlist))
    count = len(crosswords)
    cvtext = textInput

    pwordsfile = open('Utility_Mywords.txt','r')
    pwords = set(list(lib_str.split(re.sub('\n',' ', pwordsfile.read()),' ')))
    pwordsfile.close()
    
    crosswords = crosswords.difference(pwords)
    
    if count == 0:
        for e in words:
            cvtext = re.sub('(\s+' + e + '\s)',' ', cvtext)
    else:
        for e in crosswords:
            cvtext = re.sub('(\s+' + e + '\s)',' ', cvtext)
    return cvtext
        
def getstringlist(textInput):
    """
    """
    wordsarray = np.array(list(lib_str.split(textInput,' ')))
    wordsarray = list(wordsarray[wordsarray<>''])
    for e in wordsarray:
        if len(e)==1: 
            wordsarray.remove(e)
    return wordsarray

def datapurse_general(textInput):
    """
    """
    # html tag removing
    stringpurse = control_nohtmltags(textInput)
    # non-alpha character
    stringpurse = control_nounicode(stringpurse)
    # punctuation removing
    stringpurse = control_nopunctuation(stringpurse)
    # special char removing
    stringpurse = control_nospecchar(stringpurse)
    stringpurse = control_lowercase(stringpurse)    
    # common words removing
    stringpurse = control_nocommonwords(stringpurse)    
    # get list and return
    # number removing
    stringpurse = control_nonumber(stringpurse)    

    return stringpurse
 
def datapurse_cv(cvtext, jdtext):
    """
    """
    cvwords = datapurse_general(cvtext)
    jdlist = list(set(getstringlist(datapurse_general(jdtext))))
    return getstringlist(control_noerrorwords(cvwords,jdlist)), getstringlist(jdtext)        
    

   