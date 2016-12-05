# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 13:33:30 2016

@author: forest.deng
"""

# QSTK Imports

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Utility_FunctionsLib as ufl

def bollinger(df_closingPrices, lookbackPeriod):
	
	# Fetching the symbols from the dataframe column. 
	ls_symbols = df_closingPrices.columns.values
		
	temp =np.zeros((len(df_closingPrices),0))
	
	# Creating three dataframes which will keep the moving_avg, 
      # moving_stddev and bollinger_vals
	df_movingavg = pd.DataFrame(temp, index = df_closingPrices.index)
	df_movingstddev = pd.DataFrame(temp, index = df_closingPrices.index)
	df_bollinger_vals = pd.DataFrame(temp, index = df_closingPrices.index)

	# For all the symbols
	for symbol in ls_symbols:
		# Calculate the moving avg and assign it to the 
           # df_movingavg for that symbol
		df_movingavg[symbol] = pd.Series(pd.rolling_mean
                                      (
                                      df_closingPrices[symbol], 
                                      lookbackPeriod
                                      ), 
                                      index= df_movingavg.index
                                      )
		
		# Calculate the moving stddev and assign it to 
           # the df_movingstddev for that symbol
		df_movingstddev[symbol] = pd.Series(pd.rolling_std
                                      (
                                      df_closingPrices[symbol], 
                                      lookbackPeriod
                                      ), 
                                      index= df_movingstddev.index
                                      )
		
		# Calculate the bollinger values using 
           # 'Bollinger_val = (price - rolling_mean) / (rolling_std)'
		# and assign it to the df_bollinger_vals for that symbol
		df_bollinger_vals[symbol] = (df_closingPrices[symbol] - 
                                         df_movingavg[symbol])/\
                                         df_movingstddev[symbol]

	# returning the bollinger values, the sma and rolling stddev
	return df_bollinger_vals, df_movingavg, df_movingstddev

def localTest():
    """
    """
    ls_symbols = ['AAPL', 'GLD', 'GOOG', 'XOM']
    dt_start = dt.datetime(2010,12,1)
    dt_end = dt.datetime(2011,12,31)    
    lookbackPeriod = 20

    closingPrices, ldt_timestamps = ufl.fetchNYSEData(dt_start, 
                                                      dt_end, 
                                                      ls_symbols)    

    df_closingprices = pd.DataFrame(closingPrices, 
                                    columns = ls_symbols, 
                                    index = ldt_timestamps)
	
    df_bollinger_vals, df_movingavg, df_movingstddev = bollinger(
                                    df_closingprices, 
                                    lookbackPeriod
                                    )
	
    print df_bollinger_vals.to_string()
                                                      
		
if __name__ == '__main__':

    localTest()