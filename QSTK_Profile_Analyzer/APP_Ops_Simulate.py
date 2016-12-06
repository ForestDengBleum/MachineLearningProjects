# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 16:58:01 2016

@author: forest.deng
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import Utility_FunctionsLib as ufl

def readOrdersFileIntoDF(filename):

	# opening the filename
	fr = open(filename)
	
	# Lists used for making the dataframe.
	dtList = []
	symbolList = []
	orderTypeList = []
	volumeList = []
	
	# For each line
	for orderString in fr.readlines():
		
		# Stripping off the return line character
		orderString=orderString.strip()
		
		# Splitting the line and getting a List back
		listFromLine = orderString.split(',')
		
		# Adding the dates into dtList. 16,00,00 for 1600 hrs
		dtList.append(dt.datetime(int(listFromLine[0]), 
                                     int(listFromLine[1]), 
                                     int(listFromLine[2]), 
                                     16, 
                                     00, 
                                     00
                                     ))
		
		# Adding the symbols into symbolList
		symbolList.append(listFromLine[3])
		
		# Adding the orders into orderTypeList
		orderTypeList.append(listFromLine[4])
		
		# Adding the number of shares into volumeList
		volumeList.append(listFromLine[5])
	
	# Creating a Dictionary for converting it into DataFrame later
	data = { 'datetime' : dtList, 
               'symbol' : symbolList, 
               'ordertype':orderTypeList, 
               'volume':volumeList }
	
	# Converting the Dictinary into a nice looking Pandas Dataframe
	ordersDataFrame = pd.DataFrame(data)
	
	#Sorting by datetime column #Makes Sense :)
	sortedOrdersDataFrame = ordersDataFrame.sort_index(by=['datetime'])
	sortedOrdersDataFrame = sortedOrdersDataFrame.reset_index(drop=True)
	
	symbolList = list(set(sortedOrdersDataFrame['symbol']))
	
	# Returning it.
	return sortedOrdersDataFrame, symbolList
	
   
def marketsim(initialCash, ordersDataFrame, symbols, dt_end_ex = None):
		
	# reading the boundary dates
    dt_start = ordersDataFrame.datetime[0]
    dt_end = ordersDataFrame.datetime[len(ordersDataFrame)-1]

    if dt_end_ex != None:
        if type(dt_end_ex) == list:
            dt_end = dt.datetime(dt_end_ex[0], dt_end_ex[1], dt_end_ex[2])
        else:
            dt_end = dt_end_ex        

    closingPrices, ldt_timestamps, _ = ufl.fetchNYSEData(dt_start, 
                                                      dt_end, 
                                                      symbols)
	
    num_tradingDays = len(ldt_timestamps)
	
    # For Holdings of the share	
    holdings = pd.DataFrame(np.zeros((1, len(symbols))), 
                            columns = symbols, 
                            index = ['holdings'])
	
    #Cash for the days
    cash = pd.DataFrame(np.zeros((num_tradingDays, 1)), 
                        columns = ['cashinhand'])
	
    #Value for the days
    valueFrame = pd.DataFrame(np.zeros((num_tradingDays, 1)), 
                              columns = ['valueOfPortfolio'])
	
    index = 0
	
    for tradingDayIndex in range(num_tradingDays):
                
        if tradingDayIndex != 0:
            cash.cashinhand.ix[tradingDayIndex] = \
                                cash.cashinhand.ix[tradingDayIndex - 1] 
        else:
            cash.cashinhand.ix[tradingDayIndex] = initialCash

        for tradingOrderDate in ordersDataFrame.datetime:				
            if tradingOrderDate == ldt_timestamps[tradingDayIndex]:
                if ordersDataFrame.ordertype.ix[index] == 'Buy':
                    toBuySymbol = ordersDataFrame.symbol.ix[index]
                    toBuy = symbols.index(toBuySymbol)
                    numShares = ordersDataFrame.volume.ix[index]
                    priceForTheDay = closingPrices[tradingDayIndex, toBuy]
                    cash.cashinhand.ix[tradingDayIndex] = \
                            cash.cashinhand.ix[tradingDayIndex] - \
                                 (priceForTheDay * float(numShares))						
                    holdings[toBuySymbol].ix[0] += int(numShares)
                elif ordersDataFrame.ordertype.ix[index] == 'Sell':
                    toSellSymbol = ordersDataFrame.symbol.ix[index]
                    toSell = symbols.index(toSellSymbol)
                    numShares = ordersDataFrame.volume.ix[index]
                    priceForTheDay = closingPrices[tradingDayIndex, toSell]
                    cash.cashinhand.ix[tradingDayIndex] = \
                                 cash.cashinhand.ix[tradingDayIndex] + \
                                 (priceForTheDay * float(numShares))						
                    holdings[toSellSymbol].ix[0] -= int(numShares)
                else:
                    print "error"
                index += 1
		
            valueFromPortfolio = 0
		
            for symbol in symbols:			
                priceForTheDay = closingPrices[tradingDayIndex, 
                                               symbols.index(symbol)]
                valueFromPortfolio += holdings[symbol].ix[0] * priceForTheDay
            
            valueFrame.valueOfPortfolio.ix[tradingDayIndex] = \
                                    valueFromPortfolio + \
                                    cash.cashinhand.ix[tradingDayIndex]
		
    valueFrame.index = ldt_timestamps
    return holdings, valueFrame, cash

def writeValuesIntoCSV(valuesFilename, valueFrame):
    file = open(valuesFilename, 'w')
    writer = csv.writer(file)
	
    for index in range(len(valueFrame)):
        writer.writerow([valueFrame.index[index].year, 
                         valueFrame.index[index].month, 
                         valueFrame.index[index].day ,
                         int(round(valueFrame.valueOfPortfolio.ix[index], 0))
                         ])
		
def analyze(valueFrame):
    
    symbols = ['$SPX']
    dt_start = valueFrame.index[0]
    dt_end = valueFrame.index[len(valueFrame) - 1]
        
    spxClosingPrices, ldt_timestamps, _ = ufl.fetchNYSEData(
                                                    dt_start, 
                                                    dt_end, 
                                                    symbols
                                                    )
    num_tradingDays = len(ldt_timestamps)
    dailyrets = np.zeros((num_tradingDays, 2))
    cumrets = np.zeros((num_tradingDays, 2))
	
	# The first day prices of the equities
    arr_firstdayPrices = [spxClosingPrices[0,0],
                          valueFrame.valueOfPortfolio.ix[0]]
    

    for i in range(num_tradingDays):
        if i != 0:
            dailyrets[i,0] = ((spxClosingPrices[i,0]/spxClosingPrices[i-1,0]) 
                                - 1)
            dailyrets[i,1] = ((valueFrame.valueOfPortfolio.ix[i]/
                                valueFrame.valueOfPortfolio.ix[i-1]) -1 )
        else:
            dailyrets[i, 0] = 1.0
            dailyrets[i, 1] = 1.0                        

    for i in range(num_tradingDays):
        if i != 0:
            cumrets[i,0] = ((spxClosingPrices[i,0]/arr_firstdayPrices[0]))
            cumrets[i,1] = ((valueFrame.valueOfPortfolio.ix[i]/
                             arr_firstdayPrices[1]))
        else:
            cumrets[i, 0] = 1.0
            cumrets[i, 1] = 1.0
    averageSPXDailyRets = np.average(dailyrets[:,0])
    averagePortfolioDailyRets = np.average(dailyrets[:,1])
	
    stddevSPX = np.std(dailyrets[:,0])
    stddevPort = np.std(dailyrets[:,1])
	
    totalSPXRet = cumrets[-1,0]
    totalPortRet = cumrets[-1,1]
	
    sharpeRatioSPX = (averageSPXDailyRets/stddevSPX) * np.sqrt(250)
    sharpeRatioPort = (averagePortfolioDailyRets/stddevPort) * np.sqrt(250)

    plt.clf();
    fig, ax = plt.subplots()
    ax.plot(ldt_timestamps, cumrets[:,0]);    #SPX
    ax.plot(ldt_timestamps, cumrets[:,1]);    #Portfolio
    ax.axhline(y=0, color='r');
    ax.legend(['$SPX', 'Portfolio'], loc = 4)
    ax.set_ylabel('Cumulative Returns')
    ax.set_xlabel('Date')
    ax.autoscale_view()
    #plt.savefig('chart.pdf', format='pdf')
    fig.autofmt_xdate()

    print "The final value of the portfolio using the sample file is --", \
          valueFrame.valueOfPortfolio.ix[-1]
    print "Details of the Performance of the portfolio"
    print ""
    print "Data Range :", ldt_timestamps[0] ," to ", ldt_timestamps[-1]
    print ""
    print "Sharpe Ratio of Fund :", sharpeRatioPort
    print "Sharpe Ratio of $SPX :", sharpeRatioSPX
    print ""
    print "Total Return of Fund :", totalPortRet
    print "Total Return of $SPX :", totalSPXRet
    print ""
    print "Standard Deviation of Fund :", stddevPort
    print "Standard Deviation of $SPX :", stddevSPX
    print ""
    print "Average Daily Return of Fund :", averagePortfolioDailyRets
    print "Average Daily Return of $SPX :", averageSPXDailyRets

def LocalTest():
    
    #print 'Argument List:', str(sys.argv)
	
    initialCash = 1000000
    ordersFilename = 'order.csv'
    valuesFilename = 'value.csv'
	
	# Reading the data from the file, and getting a NumPy matrix
    ordersDataFrame, symbols = readOrdersFileIntoDF(ordersFilename)
    holdings, valueFrame, cash = marketsim(initialCash, 
                                           ordersDataFrame, 
                                           symbols
                                           )
	
    writeValuesIntoCSV(valuesFilename, valueFrame)
	
    analyze(valueFrame)
	
if __name__ == '__main__':
    
    LocalTest()
