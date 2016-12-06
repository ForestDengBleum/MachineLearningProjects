# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:01:34 2016

@author: forest.deng
"""

#imports
# QSTK Imports
import QSTK.qstkutil.tsutil as tsu

# Third Party Imports
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import time
import itertools as ittl
#from scipy.optimize import minimize

import Utility_FunctionsLib as ufl


'''
' Calculate Portfolio Statistics 
'''
def calcStats(na_normalized_price, lf_allocations):

    na_weighted_price = na_normalized_price * lf_allocations

    na_portf_value = na_weighted_price.sum(axis=1)

    # Calculate daily returns on portfolio
    na_portf_rets = na_portf_value.copy()
    tsu.returnize0(na_portf_rets)

    f_portf_volatility = np.std(na_portf_rets)

    f_portf_avgret = np.mean(na_portf_rets)

    f_portf_sharpe = (f_portf_avgret / f_portf_volatility) * np.sqrt(250)

    f_portf_cumrets = np.cumprod(na_portf_rets + 1);
    
    return [
            f_portf_volatility, 
            f_portf_avgret, 
            f_portf_sharpe, 
            f_portf_cumrets, 
            na_portf_value
            ]

'''
' Simulate and assess performance of multi-stock portfolio
'''
def simulate(ls_startDate, 
             ls_endDate, 
             ls_symbols, 
             lf_allocations, 
             b_print=True):

    start = time.time()
    
    #Check if ls_symbols and lf_allocations have same length
    if len(ls_symbols) != len(lf_allocations):
        print '''ERROR: Make sure symbol and allocation 
                lists have same number of elements.'''
        return

    #Check if lf_allocations adds up to 1
    if sum(lf_allocations) != 1.0:
        print 'ERROR: Make sure allocations add up to 1.'
        return

    #Prepare data for statistics
    d_data_list = ufl.readData(ls_startDate, ls_endDate, ls_symbols)
    d_data = d_data_list[0]

    #Get numpy ndarray of close prices
    na_price = d_data['close'].values;

    na_normalized_price = na_price / na_price[0,:];

    lf_Stats = calcStats(na_normalized_price, lf_allocations);

    #Print results
    if b_print:
        print 'Start Date: ', ls_startDate
        print 'End Date: ', ls_endDate
        print 'Symbols: ', ls_symbols
        print 'Allocation: ', lf_allocations
        print 'Volatility (stdev daily returns): ' , lf_Stats[0]
        print 'Average daily returns: ' , lf_Stats[1]
        print 'Sharpe ratio: ' , lf_Stats[2]
        print 'Final Cumulative daily return: ' , lf_Stats[3][-1]

        print 'Run in: ' , (time.time() - start) , ' seconds.'

    return lf_Stats[0:3]; 

'''
' Optimize portfolio allocations  to maximise Sharpe ratio
'''
def optimize(ls_startDate, ls_endDate, ls_symbols):

    start = time.time();

    #Prepare data for statistics
    ld_alldata = ufl.readData(ls_startDate, ls_endDate, ls_symbols)
    d_data = ld_alldata[0]

    #Get numpy ndarray of close prices (numPy)
    na_price = d_data['close'].values

    na_normalized_price = na_price / na_price[0,:]
    
    symbols_count = len(ls_symbols)

    li_valid = []    
    all_combination = ittl.product(range(11), repeat = symbols_count)
    for e in all_combination:
        if sum(e) == 10:
            li_valid.append(e)
    #Convert to float array that sum to 1
    lf_valid = [];
    for i in li_valid:
        lf_valid.append([j/10.0 for j in i])

    #Calculate Sharpe ratio for each valid allocation
    f_CurrMaxSharpe = 0.0;
    for allocation in lf_valid:
        t_Stats = calcStats(na_normalized_price, allocation)
        if t_Stats[2] > f_CurrMaxSharpe:
            lf_CurrStats = t_Stats
            f_CurrMaxSharpe = t_Stats[2]
            lf_CurrEffAllocation = allocation

    #Plot portfolio daily values over time period
    #Obtain benchmark $SPX data
    d_spx = ufl.readData(ls_startDate, ls_endDate, ['$SPX'])[0]
    na_spxprice = d_spx['close'].values
    na_spxnormalized_price = na_spxprice / na_spxprice[0,:]
    lf_spxStats = calcStats(na_spxnormalized_price, [1])
    #Plot
    #plt.clf();
    fig, ax = plt.subplots()
    ax.plot(ld_alldata[4], lf_spxStats[4])   #SPX
    ax.plot(ld_alldata[4], lf_CurrStats[4])  #Portfolio
    ax.axhline(y=0, color='r')
    ax.legend(['$SPX', 'Portfolio'], loc = 4)
    ax.set_ylabel('Daily Value')
    ax.set_xlabel('Date')
    ax.autoscale_view()
    #plt.savefig('chart.pdf', format='pdf')
    fig.autofmt_xdate()
    #Print results:
    print 'Start Date: ', ls_startDate
    print 'End Date: ', ls_endDate
    print 'Symbols: ', ls_symbols
    print 'Optimal Allocations: ', lf_CurrEffAllocation
    print 'Volatility (stdev daily returns): ' , lf_CurrStats[0]
    print 'Average daily returns: ' , lf_CurrStats[1]
    print 'Sharpe ratio: ' , lf_CurrStats[2]
    print 'Cumulative daily return: ' , lf_CurrStats[3][-1]

    print 'Run in: ' , (time.time() - start) , ' seconds.'
          
'''
' local test
'''
def localTest():
    startDate = [2011,1,1]
    endDate = [2011,12,31]
    print 'Portfolio Simulation by Assigned Allocation: '
    simulate(
             startDate,
             endDate,
             ['AAPL', 'GLD', 'GOOG', 'XOM'], 
             [0.4, 0.3, 0.1, 0.2], 
             True
             )
    print 'Portfolio Optimization Result: '
    print '-----------------------------------------------'
    optimize(
             startDate,
             endDate,
             ['AAPL', 'GLD', 'GOOG', 'XOM']
             )
    

if __name__=='__main__':
    
    localTest()