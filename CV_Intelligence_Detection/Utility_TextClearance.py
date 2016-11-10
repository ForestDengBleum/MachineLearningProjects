# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:56:21 2016

@author: forest.deng
"""
import re
import string as lib_str
import Utility_RemoveHtmlTags as rht


def control_lowercase(textInput):
    """
    """
    return lib_str.lower(textInput)

def control_nopunctuations(textInput):
    """
    """
    

def control_nohtmltags(textInput):
    """
    """
    return re.sub(r'[^\w\s]',' ', rht.filter_tags(textInput))

def control_nonumber(textInput):
    """
    """
    return re.sub(r'[\d+]',' ', textInput)


def control_nospecchar(textInput):
    """
    """
        
    