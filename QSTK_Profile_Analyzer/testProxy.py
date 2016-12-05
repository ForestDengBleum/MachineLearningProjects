# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 14:21:03 2016

@author: forest.deng
"""
import urllib2

password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, r"cdhkproxy:8080", r"forest.deng", r"dog.lot.cat-033")


auth = urllib2.HTTPBasicAuthHandler(password_mgr)
opener1 = urllib2.build_opener(auth)
urllib2.install_opener(opener1)

u = urllib2.urlopen('http://ichart.finance.yahoo.com/table.csv?a=1&c=2000&b=1&e=5&d=11&f=2016&s=%5EDJI')
data = u.read()