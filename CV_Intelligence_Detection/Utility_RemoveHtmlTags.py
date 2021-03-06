# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:35:56 2016

@author: forest.deng
"""

# -*- coding: utf-8-*-
import re
#import Utility_TextClearance as utc

def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#
    re_h=re.compile('</?\w+[^>]*>')#HTML tag
    re_comment=re.compile('<!--[^>]*-->')#HTML comments
    s=re_cdata.sub(' ',htmlstr)
    s=re_script.sub(' ',s)
    s=re_style.sub(' ',s)
    s=re_br.sub('\n',s)
    s=re_h.sub(' ',s)
    s=re_comment.sub(' ',s) 
    
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=replaceCharEntity(s)
    return s

#@param htmlstr HTML str.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
                'lt':'<','60':'<',
                'gt':'>','62':'>',
                'amp':'&','38':'&',
                'quot':'"','34':'"',}
    
    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        key=sz.group('name')#remove &;gt;gt
        try:
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            # space replace
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
    return htmlstr

def repalce(s,re_exp,repl_string):
    return re_exp.sub(repl_string,s)

