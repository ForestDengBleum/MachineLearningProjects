# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:20:03 2016

@author: forest.deng
"""

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt        
        
'''
' Reads data from Yahoo Finance
'''
def readData(
                ls_startDate, 
                ls_endDate, 
                ls_symbols, 
                ls_keys = ['open', 'high', 'low', 
                           'close', 'volume', 'actual_close']
                           ):

    dt_start = dt.datetime(ls_startDate[0], ls_startDate[1], ls_startDate[2]);
    dt_end = dt.datetime(ls_endDate[0], ls_endDate[1], ls_endDate[2]);

    dt_timeofday = dt.timedelta(hours=16);

    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday);

    c_dataobj = da.DataAccess('Yahoo');

    #ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close'];

    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys);
    d_data = dict(zip(ls_keys, ldf_data));

    return [d_data, dt_start, dt_end, dt_timeofday, ldt_timestamps];
        