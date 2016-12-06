# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 10:04:16 2016

@author: forest.deng
"""

# QSTK Imports
import QSTK.qstkstudy.EventProfiler as ep

# Third Party Imports
import datetime as dt
import pandas as pd
import numpy as np
import copy

import Utility_FunctionsLib as ufl

# User Inputs
import APP_Bollinger_Calc as abc

def find_bollinger_events(df_bollingerValues):
	''' Finding the event dataframe '''
	
	# Fetching the symbols from the dataframe column. 
	ls_symbols = df_bollingerValues.columns.values
	
	ts_market = df_bollingerValues['SPY']

    # Creating an empty dataframe
	df_events = copy.deepcopy(df_bollingerValues)
	df_events = df_events * np.NAN

	# Time stamps for the event range
	ldt_timestamps = df_bollingerValues.index

	for s_sym in ls_symbols:
		for i in range(1, len(ldt_timestamps)):
			f_eqbollingerValueToday = df_bollingerValues[s_sym].ix[
                                               ldt_timestamps[i]]
			f_eqbollingerValueYesterday = df_bollingerValues[s_sym].ix[
                                               ldt_timestamps[i - 1]]
			f_marketBollingerValueYesterday = ts_market[ldt_timestamps[i]]

			if (f_eqbollingerValueToday <= -2.0 and 
                     f_eqbollingerValueYesterday >= -2.0 and 
                     f_marketBollingerValueYesterday >= 0.0):
				df_events[s_sym].ix[ldt_timestamps[i]] = 1
				
	return df_events
	
def localTest():

    ls_symbols = ['AAPL', 'GLD', 'GOOG', 'XOM', 'SPY'] #last one is market
    dt_start = dt.datetime(2010,12,1)
    dt_end = dt.datetime(2011,12,31)   
    lookbackPeriod = 20
    dt_feature = dt_start.strftime('%Y%m%d') + '_' + dt_end.strftime('%Y%m%d') 
    nameofchartfile = 'EventFind' + dt_feature + '.pdf'

    closingPrices, ldt_timestamps, d_data = ufl.fetchNYSEData(
                                                 dt_start, 
                                                 dt_end, 
                                                 ls_symbols
                                                 )
	
    df_closingprices = pd.DataFrame(
                                    closingPrices, 
                                    columns = ls_symbols, 
                                    index = ldt_timestamps
                                    )
	
    df_bollinger_vals, df_movingavg, df_movingstddev = abc.bollinger(
                                                        df_closingprices, 
                                                        lookbackPeriod
                                                        )
	
    df_events = find_bollinger_events(df_bollinger_vals)
	
    ep.eventprofiler(
                     df_events, 
                     d_data, 
                     i_lookback=20, 
                     i_lookforward=20, 
                     s_filename=nameofchartfile, 
                     b_market_neutral=True, 
                     b_errorbars=True,
                     s_market_sym='SPY'
                     )

if __name__ == '__main__':

    localTest()

