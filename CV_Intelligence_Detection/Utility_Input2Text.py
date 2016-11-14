# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:44:45 2016

@author: forest.deng
"""

import PyPDF2 as lib_pdf
import docx as lib_word
import string as lib_str
import os as lib_os
import fnmatch as lib_fnmatch

# read different format files
#
#

#extDic ={'pdf':['pdf'],'word':['doc','docx'],
#         'html':['html','htm'],'text':['txt']}

def get_pdftext(fileName):
    """
    """
    pdfFile = open(fileName,'rb')
    pdfReader = lib_pdf.PdfFileReader(pdfFile)
    pageNo = pdfReader.numPages -1
    items =[]
    for i in range(pageNo):
        pageObj = pdfReader.getPage(i)
        items.append(pageObj.extractText())
    pdfFile.close()
    return ' '.join(items)    

def get_wordtext(fileName):
    """
    """
    doc = lib_word.Document(fileName)
    items = []
    for para in doc.paragraphs:
        items.append(para.text)
    #doc.close()
    return ' '.join(items)

def get_htmltext(fileName):
    """
    """
    return open(fileName).read()

# A wrap to read all types files
#
def get_text(fileName):
    """
    """
    extName = lib_str.lower(lib_str.split(fileName,'.')[-1])
    if extName == 'pdf': return get_pdftext(fileName)
    if extName in ['doc','docx']: return get_wordtext(fileName)
    if extName in ['htm','html','mht']: return get_htmltext(fileName)

# Get the file list from a specified folder
#

def list_allfiles(dirName, patterns='*', single_level = False, yield_folders = False):
    """
    """    
    patterns = patterns.split(';')
    for path, subdirs, files in lib_os.walk(dirName):
        if yield_folders:
            files.extend(subdirs)
        files.sort()
        for name in files:
            for pattern in patterns:
                if lib_fnmatch.fnmatch(name, pattern):
                    yield lib_os.path.join(path, name)
                    break
                if single_level:
                    break 

def get_listtext(dirName, patterns='*', single_level = False, yield_folders = False):
    """
    """
    items =[];
    fileList = list(list_allfiles(dirName,patterns,single_level,yield_folders))
    for sfile in fileList:
        items.append(get_text(sfile))
    return items
        
    
        
    