# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:08:01 2016

@author: forest.deng
"""

from xml.sax.handler import ContentHandler
import xml.sax
import sys
class textHandler(ContentHandler):
    def characters(self, ch):
        sys.stdout.write(ch.encode("Latin-1"))
parser = xml.sax.make_parser( )
handler = textHandler( )
parser.setContentHandler(handler)
parser.parse("D:\\new\\cc.html")

