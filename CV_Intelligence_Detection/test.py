# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:08:01 2016

@author: forest.deng
"""
#import re
#from enchant.checker import SpellChecker
#import Utility_RemoveHtmlTags as rht
#import string
#
#dic = et.Dict('en_US')
#
#
#s=file('D:\\new\\cc.html').read()
#news=rht.filter_tags(s)
#rep = ' '*len(string.punctuation)
#news = news.translate(string.maketrans(string.punctuation,rep))
#news = re.sub(r'[\d+]','', news)
##print news
#
#checker = SpellChecker('en_US')
#checker.set_text(news)
#for err in checker:
#    print err.word
# 
import Utility_Input2Text as uit
import Utility_TextClearance as utc

cvtext = uit.get_text('d:\\data\\cv.docx')

jdtext = uit.get_text('d:\\data\\jd.docx')

#print cvtext

#print cvtext
cvlist = utc.datapurse_general(cvtext)
print cvlist
