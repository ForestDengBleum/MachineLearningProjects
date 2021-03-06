# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:44:45 2016

@author: forest.deng
"""

import PyPDF2 as lib_pdf
import docx as lib_word

import os as lib_os
import win32com.client

import string as lib_str
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

def get_wordtext2(fileName):
    """
    """
    wordapp = win32com.client.Dispatch('Word.Application')
    doc = wordapp.Documents.Open(fileName,ConfirmConversions=False)
    txt = doc.Range().Text    
    doc.Close()
    wordapp.Quit()
    
    return txt
        
        
    
#   doc = win32com.client.GetObject(fileName)
#    return doc.Range().Text
    #doc.close()


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
    if extName in ['doc','docx']: return get_wordtext2(fileName)
    if extName in ['htm','html','mht']: return get_htmltext(fileName)

# Get the file list from a specified folder
#

def list_allfiles(dirName, patterns='*', single_level = False, 
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

def get_listtext(dirName, patterns='*', single_level = True, 
                 yield_folders = False):
    """
    """
    items =[];
    file_name = []
    fileList = list(list_allfiles(dirName,patterns,single_level,yield_folders))
    for sfile in fileList:
        items.append(get_text(sfile))
        file_name.append(lib_str.split(sfile,'\\')[-1])
    return items, file_name
        
    
        
    