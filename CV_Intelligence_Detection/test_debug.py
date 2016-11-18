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
#import Algrithm_CVJDSimilarity as ac
import Algorithm_TFIDF_BAYES as at





testcv = uit.get_wordtext2('d:\\data\\cv3.doc')

cvtext1 = uit.get_text('d:\\data\\cv.docx')

cvtext2 = uit.get_text('d:\\data\HID-Kelly Zuo.docx')

jdtext = uit.get_text('d:\\data\\jd.docx')

#print cvtext

#print cvtext
cvtext1 = utc.datapurse_general(cvtext1)
cvtext2 = utc.datapurse_general(testcv)
jdtext = utc.datapurse_general(jdtext)

#print 'cvtext: %s' %cvtext
#print 'jdtext: %s' %jdtext
#print ' '

pursedlist1, jdlist = utc.datapurse_cv(cvtext1, jdtext)
pursedlist2, jdlist = utc.datapurse_cv(cvtext2, jdtext)

pursedlist = [pursedlist1, pursedlist2]

returns = at.get_trained_bayesprobability(pursedlist)


returnM = at.get_trained_tddf_feature(pursedlist)

print returnM
#dic = ac.list_counter(pursedlist)


#cvframe = ac.get_dataframewithper(pursedlist)
#jdframe = ac.get_dataframewithper(jdlist)

#similarity = ac.get_similarity(cvframe, jdframe)

#print similarity

#docs = ' '.join(jdlist)

#t = ac.text_tokenize([docs])
#print t.transpose()

#print pursedlist

#print jdlist
