# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 15:39:06 2016

@author: forest.deng
"""

# Third Party Imports
import copy
import numpy as np

# QSTK Imports
#import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkstudy.EventProfiler as ep

import Utility_FunctionsLib as ufl

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""

def find_events(ls_symbols, d_data):
    ''' Finding the event dataframe '''
    df_close = d_data['actual_close']
    ts_market = df_close['SPY']

    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

	# Time stamps for the event range
    ldt_timestamps = df_close.index


    for s_sym in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symprice_today = df_close[s_sym].ix[ldt_timestamps[i]]
            f_symprice_yest = df_close[s_sym].ix[ldt_timestamps[i - 1]]
            f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            f_marketreturn_today = (f_marketprice_today / 
                                       f_marketprice_yest) - 1

            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            if (f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02):
                df_events[s_sym].ix[ldt_timestamps[i]] = 1

    return df_events

if __name__ == '__main__':
    startDate = [2008,1,1]
    endDate = [2009,1,1]

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    ls_symbols.append('SPY')
      
    d_data = ufl.readData(startDate, endDate, ls_symbols)[0]

#    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
#
#    for s_key in ls_keys:
#        d_data[s_key] = d_data[s_key].fillna(method='ffill')
#        d_data[s_key] = d_data[s_key].fillna(method='bfill')
#        d_data[s_key] = d_data[s_key].fillna(1.0)
        
    dt_feature = ''.join([str(e) for e in startDate]) + '_' + \
                ''.join([str(e) for e in endDate])    
    
    df_events = find_events(ls_symbols, d_data)
    ep.eventprofiler(
                    df_events, 
                    d_data, 
                    i_lookback=20, 
                    i_lookforward=20, 
                    s_filename='EventFind' + dt_feature + '.pdf' , 
                    b_market_neutral=True, 
                    b_errorbars=True,
                    s_market_sym='SPY')