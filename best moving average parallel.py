#!/usr/bin/env python
# coding: utf-8

# In[54]:


import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from datetime import timedelta
from datetime import date


import re
import time
import stat
import contextlib
import functools

import concurrent.futures
from concurrent.futures import wait, ALL_COMPLETED

import urllib
import urllib.request
import pandas_market_calendars as mcal
import os.path
from os import path
from functools import reduce
#import pandas-datareader
#import mpl-finance
#from fbprophet.diagnostics import cross_validation
#rom fbprophet import Prophet
import stockstats
from stockstats import StockDataFrame
import tabulate
import mplfinance as mpf
import matplotlib.dates as mdates
import statsmodels.tsa.stattools as ts
from hurst import compute_Hc

from finta import TA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error

from scipy.stats import ttest_ind


# In[55]:


n_forward = 7

#name = 'GLD'
#name = 'SPY'
#name = 'GOOG'

#strategy = "EMA"
strategy = "SMA"
#indicator = 'Close'
indicator = 'VWP'

w=117
end_date = datetime.date.today()
#end_date = datetime.date.today() - timedelta(weeks=w)
end_date1 = end_date - timedelta(weeks=w)

#- timedelta(weeks=w*2)
start_date = end_date1 - timedelta(weeks=w)

benchName = "^GSPC"
bench = yf.Ticker(benchName)
benchData = bench.history(interval="1d",start=start_date,end=end_date, auto_adjust=True)


# In[56]:



dateindex = benchData.loc[start_date:end_date].index
dateindex_n_forward = [start_date + datetime.timedelta(days=x) for x in range(0, ((end_date+ timedelta(days=n_forward))-start_date).days)]

dateindex2 = benchData.loc[end_date1:end_date].index

dateindex2_n_foward = [end_date1 + datetime.timedelta(days=x) for x in range(0, ((end_date+ timedelta(days=n_forward))-end_date1).days)]

nyse = mcal.get_calendar('NYSE')
nyse_trading_dates= nyse.schedule(start_date=start_date, end_date=end_date+timedelta(days=n_forward))


# In[57]:


#if(len(data)==len(dateindex_)):
if(len(benchData)>len(nyse_trading_dates)):
    frequency=pd.DataFrame(dateindex_n_forward).set_index(0)
    
else:
    frequency=nyse_trading_dates
    #frequency = pd.DataFrame(frequency).set_index(0).index
    
frequency = frequency.index

idx2 = frequency

#https://stackoverflow.com/questions/40815238/python-pandas-convert-index-to-datetime
idx2 = pd.to_datetime(idx2, errors='coerce')


# In[ ]:





# In[58]:


def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return(unique_list)


# In[ ]:





# In[ ]:


pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

cores = int(len(os.sched_getaffinity(0)))

pool1 = concurrent.futures.ProcessPoolExecutor(cores)

one_week_end = start_date
one_week_start = one_week_end - timedelta(weeks=1)

#need to do the two pass trick (i.e. find stocks fully populated a week 9 quarters back)

nyse = mcal.get_calendar('NYSE')
official_trading_dates= nyse.schedule(start_date=start_date, end_date=end_date)

date_time_obj_start = start_date

date_time_obj_end = end_date

official_trading_dates_plus5= nyse.schedule(start_date=date_time_obj_start.strftime('%Y-%m-%d'), end_date=date_time_obj_end+timedelta(days=5))

next_trading_day = official_trading_dates_plus5.index[official_trading_dates_plus5.index > end_date.strftime('%m/%d/%Y')][0]

idx2 = official_trading_dates.index

one_week_trading_dates = nyse.schedule(start_date=one_week_start, end_date=one_week_end)

url = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt'

#should turn this into a function
if path.exists("nasdaqtraded.txt"):
    print("file exists")
    
    filePath = 'nasdaqtraded.txt'
    fileStatsObj = os.stat ( filePath )
    modificationTime = time.ctime ( fileStatsObj [ stat.ST_MTIME ] )

    print("Last Modified Time : ", modificationTime )

    a = datetime.datetime.strptime(modificationTime, "%a %b %d %H:%M:%S %Y")

    if a.date() != datetime.date.today():
        print("not same dates downloading")
        urllib.request.urlretrieve(url, 'nasdaqtraded.txt')
        urllib.request.urlretrieve(url, 'mfundslist.txt')
        urllib.request.urlretrieve(url, 'bonds.txt')
    else:
      print("equal dates, not redownloading")
    
else:
    print("downloading nasdaqtraded.txt")
    urllib.request.urlretrieve(url, 'nasdaqtraded.txt')
    urllib.request.urlretrieve(url, 'mfundslist.txt')
    urllib.request.urlretrieve(url, 'bonds.txt')
    
df1 = pd.read_csv('nasdaqtraded.txt', sep='|')[0:-1]
df2 = pd.read_csv('mfundslist.txt', sep='|')[0:-1]
df3 = pd.read_csv('bonds.txt', sep='|')[0:-1]

#process symbols for bad characters
BAD_CHARS = ['$','.']
pat = '|'.join(['({})'.format(re.escape(c)) for c in BAD_CHARS])

df1 = df1[~df1['Symbol'].str.contains(pat)]
df2 = df2[~df2['Symbol'].str.contains(pat)]
df3 = df3[~df3['Symbol'].str.contains(pat)]

#choose size
size=400
stocks = list(df1["Symbol"].sample(n=int(size/3)))
mfunds = list(df2["Symbol"].sample(n=int(size/3)))
bonds = list(df3["Symbol"].sample(n=int(size/3)))
symbols = unique(stocks + mfunds + bonds)

def dl_one_week(stock):
    return yf.download(stock, start=one_week_start, end=one_week_end, auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')

def dl(stock):
    return yf.download(stock, start=start_date, end=end_date, auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')

def processStocks(symbols):

    futures1 = [pool1.submit(dl_one_week, args) for args in symbols]
    wait(futures1, timeout=None, return_when=ALL_COMPLETED)

    symbols_data_one_week = pd.DataFrame()

    for x in range(0,len(symbols)):
        prices = pd.DataFrame(futures1[x].result())
        prices['Symbol'] = symbols[x]
        prices = prices.loc[~prices.index.duplicated(keep='last')]        
        prices = prices.reset_index()

        symbols_data_one_week = pd.concat([symbols_data_one_week,prices])

    #symbols_data_one_week

    #stocks that existed 9 quarters ago
    vetted_symbols = list(symbols_data_one_week.Symbol.unique())

    pool2 = concurrent.futures.ProcessPoolExecutor(cores)

    futures2 = [pool2.submit(dl, args) for args in vetted_symbols]
    wait(futures2, timeout=None, return_when=ALL_COMPLETED)    

    symbols_data = pd.DataFrame()
    
    for x in range(0,len(vetted_symbols)):
        prices = pd.DataFrame(futures2[x].result())
        prices['Symbol'] = vetted_symbols[x]
        prices = prices.loc[~prices.index.duplicated(keep='last')]        
        prices = prices.reset_index()

        idx1 = prices.index  

        merged = idx1.union(idx2)
        s = prices.reindex(merged)
        df = s.interpolate().dropna(axis=0, how='any')

        if len(df) == len(prices.index):
            symbols_data = pd.concat([symbols_data,df])
            
    symbols_data.to_csv('symbols_data.csv', index = False)

if path.exists('symbols_data.csv'):
    print("data exists")
    
    filePath = 'symbols_data.csv'
    fileStatsObj = os.stat ( filePath )
    modificationTime = time.ctime ( fileStatsObj [ stat.ST_MTIME ] )

    print("Last Modified Time : ", modificationTime )

    a = datetime.datetime.strptime(modificationTime, "%a %b %d %H:%M:%S %Y")

    if a.date() != datetime.date.today():
        print("not same dates downloading stocks")
        
        processStocks(symbols)
        
    else:
        print("equal dates, not redownloading")
    
else:
    print("downloading symbols.txt")
    processStocks(symbols)
    


# In[ ]:


#symbols = ['BTC-USD']
#processStocks(symbols)

symbols_data = pd.read_csv('symbols_data.csv', sep=',')[0:-1]
vetted_symbols = symbols_data.Symbol.unique()


# In[ ]:





# In[ ]:


returnsdf = pd.DataFrame()
returnsl = []

#cumulative returns of 1st half
for i in vetted_symbols:
    subset = symbols_data[symbols_data["Symbol"]==i]
    subset = subset.set_index('Date')[start_date.strftime('%Y-%m-%d'):end_date1.strftime('%Y-%m-%d')]
    
    #print(subset)
    price_data = subset["Close"]
    #print(price_data)
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()
    
    last = cumulative_ret.iloc[-1]
    
    #pd.concat(last,returns)
    returnsl.append(last)

    #plt.plot(cumulative_ret, label=i)
    #plt.legend(loc="upper left",fontsize=8)
    
returnsdf["returns"] = returnsl
returnsdf["stock"] = vetted_symbols

returnsdf = returnsdf.sort_values(by=['returns'], ascending=False)

XPercent = .1
cutoff = round(len(returnsdf)*XPercent,0)

topXPercent = returnsdf['stock'][0:int(cutoff)]
topXPercent


# In[ ]:


dateindex = benchData.loc[start_date:end_date].index
returnsdf[0:int(cutoff)]


# In[ ]:





# In[ ]:


#cumulative returns over test period

for i in topXPercent:
    subset = symbols_data[symbols_data["Symbol"]==i]
    subset = subset.set_index('Date')[(end_date1+timedelta(days=1)).strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
    
    #print(subset)
    price_data = subset["Close"]
    #print(price_data)
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()
    
    last = cumulative_ret.iloc[-1]
    
    #pd.concat(last,returns)
    returnsl.append(last)

    plt.plot(cumulative_ret, label=i)
    plt.legend(loc="upper left",fontsize=8)


# In[ ]:


limit = 100

train_size = 0.5

#minExpectedReturn = 0.0005
minExpectedReturn = 0.0

width1 = len(benchData.loc[start_date:end_date1].index)

#not doing any sales here, so is the +1 necessary?
#even if doing +1 I don't want to use timedelta, but it's okay to use it becuase at most it's +1 actual day (vs trading day) and the filtering will still work
#width2 = len(data.loc[end_date1+timedelta(days=1):end_date].index)
width2 = len(benchData.loc[end_date1+timedelta(days=1):end_date].index)


dateindex = benchData.loc[start_date:end_date].index
dateindex_ = [start_date + datetime.timedelta(days=x) for x in range(0, ((end_date)-start_date).days)]
dateindex_n_forward = [start_date + datetime.timedelta(days=x) for x in range(0, ((end_date+ timedelta(days=n_forward))-start_date).days)]

dateindex2 = benchData.loc[end_date1:end_date].index

dateindex2_n_foward = [end_date1 + datetime.timedelta(days=x) for x in range(0, ((end_date+ timedelta(days=n_forward))-end_date1).days)]

sp500_data = benchData[end_date1+timedelta(days=1):end_date]['Close'].pct_change()
sp500_cumulative_ret_data = (sp500_data + 1).cumprod()
plt.plot(sp500_cumulative_ret_data,label="bench: " + benchName)


# In[ ]:





# In[ ]:


#for symbol in topXPercent:
def processSets(symbol):

    subset = symbols_data[symbols_data["Symbol"]==symbol]
    subset = subset.set_index('Date')
    subset['Forward Close'] = subset['Close'].shift(-n_forward)
    subset['Forward Return'] = (subset['Forward Close'] - subset['Close'])/subset['Close']
    subset['VWP'] = subset['Close']*subset['Volume']
    
    Short_EVWMA = pd.DataFrame(TA.EVWMA(subset,12))
    Long_EVWMA = pd.DataFrame(TA.EVWMA(subset,26))
    Short_EVWMA.columns = ['EVWMA_12']
    Long_EVWMA.columns = ['EVWMA_26']

    #p 209 of ttr doc
    MACD_EVWMA = pd.DataFrame((Short_EVWMA['EVWMA_12'] - Long_EVWMA['EVWMA_26'])/Long_EVWMA['EVWMA_26'])
    MACD_EVWMA.columns = ['MACD-line']

    Signal_EVWMA = pd.DataFrame(ta.ema(MACD_EVWMA["MACD-line"], length=9))
    Signal_EVWMA.columns = ['Signal_EMA_9_MACD']
    subset['MACD_Signal'] = Signal_EVWMA
    
    prices = subset.loc[~subset.index.duplicated(keep='last')]        
    prices = subset.reset_index()

    idx1 = subset.index  

    merged = idx1.union(idx2)
    s = subset.reindex(merged)
    df = s.interpolate().dropna(axis=0, how='any')

    subset = df

    trades = []
    expectedReturns = []

    sdevs = []
    
    conditions_ = []

    #rolling windows
    for i in range(0,width1):
        temp = subset.loc[frequency[i].strftime('%Y-%m-%d'):frequency[i+width2].strftime('%Y-%m-%d')].copy()
        #temp = subset.loc[dateindex[i].strftime('%Y-%m-%d'):dateindex[i+width2].strftime('%Y-%m-%d')].copy()
        #adf_results = ts.adfuller(temp['Close'], 1)
        #H, c, val = compute_Hc(temp['Close'], kind='price', simplified=True)
        
        #data.loc[dateindex[i]:dateindex[i+width2]]

        result = []

        for ma_length in range(20,limit):        

            if strategy == "EMA":

                temp[strategy] = ta.ema(temp[indicator], length=ma_length)
                temp['input'] = [int(x) for x in temp[indicator] > temp[strategy]]

            elif strategy == "SMA":

                temp[strategy] = temp[indicator].rolling(ma_length).mean()
                temp['input'] = [int(x) for x in temp[indicator] > temp[strategy]]

            df = temp.dropna()

            training = df.head(int(train_size * df.shape[0]))
            test = df.tail(int((1 - train_size) * df.shape[0]))

            tr_returns = training[training['input'] == 1]['Forward Return']
            test_returns = test[test['input'] == 1]['Forward Return']

            mean_forward_return_training = tr_returns.mean()
            mean_forward_return_test = test_returns.mean()
            pvalue = ttest_ind(tr_returns,test_returns,equal_var=False)[1]

            result.append({
                'ma_length':ma_length,
                'training_forward_return': mean_forward_return_training,
                'test_forward_return': mean_forward_return_test,
                'p-value':pvalue
            })

        result.sort(key = lambda x : -x['training_forward_return'])

        if strategy == "EMA":
            temp[strategy] = ta.ema(temp[indicator], length=result[0]['ma_length'])

        elif strategy == "SMA":
            temp[strategy] = temp[indicator].rolling(result[0]['ma_length']).mean()        

        conditions = 0

        if (result[0]['p-value'] > .05 and temp.iloc[-1][indicator]>temp.iloc[-1][strategy]):

            if (result[0]['training_forward_return'] > minExpectedReturn and result[0]['test_forward_return'] > minExpectedReturn):
                conditions = conditions + 1

        if conditions >= 1:    
            trades.append(temp.index[-1].strftime('%Y-%m-%d'))
            expectedReturns.append((result[0]['training_forward_return']+result[0]['test_forward_return'])/2)
            sdevs.append(np.std(temp['Forward Return']))
            #print(predRet)
            #print(temp.iloc[-1]['MACD_Signal'])

    #print("starting set")
    set = pd.DataFrame()
    for i in range(0,len(trades)):

        value = pd.DataFrame(subset.loc[trades[i]]).transpose()
        value['ExpectedReturn'] = expectedReturns[i]
        value['sdev'] = sdevs[i]
        value['conditions'] = conditions_[i]
        set = pd.concat([set,value])

    #display(set)

    return set
    #print(set)

pool3 = concurrent.futures.ProcessPoolExecutor(cores)

futures3 = [pool3.submit(processSets, args) for args in topXPercent]

wait(futures3, timeout=None, return_when=ALL_COMPLETED)


# In[ ]:





# In[ ]:



BuyFundsPercent = .75
percentHeldOnSell = 1

strategies = []
holds = []

for f in futures3:
    #throwing a weird date error with one dataframe (had date outside of range)
    
    set = pd.DataFrame(f.result())[(end_date1+timedelta(days=1)).strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]
    
    if (len(set) != 0):
        #display(set)
        #plt.hist(set['sdev'], bins='auto')  # arguments are passed to np.histogram
        #plt.show()
        #plt.hist(set['ExpectedReturn'], bins='auto')  # arguments are passed to np.histogram
        #plt.show()
        #plt.hist(set['Forward Return'], bins='auto')  # arguments are passed to np.histogram
        #plt.show()        

        orderbook = pd.DataFrame()

        #temp = pd.DataFrame([dateToBeSold,1],columns=['date','qty'])
        column_names = ["date", "qty"]

        sellDates = pd.DataFrame(columns = column_names)

        data = symbols_data[symbols_data["Symbol"]==set['Symbol'].unique()[0]].set_index('Date')
        #print(set['Symbol'])

        for i in dateindex2:

            idate = i.strftime('%Y-%m-%d')        

            #process purchases
            if (idate in set.index):

                temp = pd.DataFrame()

                estRet = set.loc[idate]['ExpectedReturn']

                temp['orderside'] = ['buy']        

                if len(data[start_date.strftime('%Y-%m-%d'):idate])-1+n_forward>=len(data[start_date.strftime('%Y-%m-%d'):]):
                    dateToBesold = np.nan    
                    #dateToBeSold = frequency[frequency.get_loc(idate)+n_forward].strftime('%Y-%m-%d')
                    #frequency[pd.DataFrame(frequency).set_index('Date').index.get_loc(dateindex[i])+n_forward].strftime('%Y-%m-%d')
                    temp['valueAtSale'] = np.nan
                else:

                    #dateToBeSold = data.iloc[len(data[start_date:idate])-1+n_forward].name.strftime('%Y-%m-%d') 
                    dateToBeSold = frequency[frequency.get_loc(datetime.datetime.strptime(idate, "%Y-%m-%d").date())+n_forward].strftime('%Y-%m-%d')
                    #frequency[pd.DataFrame(frequency).set_index('Date').index.get_loc(dateindex[i])+n_forward].strftime('%Y-%m-%d')

                    #temp['valueAtSale'] = pd.DataFrame(data.iloc[len(data[start_date:idate])-1+n_forward]).transpose()['Close'].values[0]            
                    temp['valueAtSale'] = data.loc[dateToBeSold]['Close']

                temp['date'] = [idate]
                temp['valueAtPurchase'] = set.loc[idate]['Close']
                temp['estRet'] = estRet
                #temp['qty'] = Qty
                temp['dateBought'] = idate    
                temp['dateToBeSold'] = dateToBeSold
                temp['conditions'] = set.loc[idate]['conditions']

                btemp = pd.DataFrame(columns = column_names)
                btemp["date"]=[dateToBeSold]
                #btemp["qty"]=[Qty]

                sellDates = sellDates.append(btemp,ignore_index=True)

                temp = temp.round(4)

                orderbook = orderbook.append(temp,ignore_index=True)

        for i in dateindex2:

            idate = i.strftime('%Y-%m-%d')        

            #process sales

            if (idate in sellDates.set_index('date').index):    
                temp = pd.DataFrame()

                dateBought = frequency[frequency.get_loc(datetime.datetime.strptime(idate, "%Y-%m-%d").date())-n_forward].strftime('%Y-%m-%d')
                #frequency[pd.DataFrame(frequency).set_index('Date').index.get_loc(dateindex[i])-1-n_forward].strftime('%Y-%m-%d')
                #dateBought = data.iloc[len(data[start_date:idate])-1-n_forward].name.strftime('%Y-%m-%d')   

                dateToBeSold = idate
                temp['dateBought'] = [dateBought]
                temp['dateToBeSold'] = dateToBeSold
                #temp['valueAtPurchase'] = pd.DataFrame(data.iloc[len(data[start_date:idate])-1-n_forward]).transpose()['Close'].values[0]
                temp['valueAtPurchase'] = data.loc[dateBought]['Close']
                estRet = set.loc[dateBought]['ExpectedReturn']
                temp['estRet'] = estRet
                temp['valueAtSale'] = data.loc[dateToBeSold]['Close']
                #temp['valueAtSale'] = pd.DataFrame(data.iloc[len(data[start_date:idate])-1]).transpose()['Close'].values[0]

                #temp['dateToBeSold'] = idate
                #temp['estRet'] = data.loc[idate]['Forward Return']

                temp['orderside'] = ['sell']        
                temp['date'] = [idate]

                #data vs set because set only includes buy dates
                #temp['valueAtSale'] = pd.DataFrame(data.ix[len(data[start_date:idate])-1+n_forward]).transpose()['Close']

                #temp['qty'] = sellDates.set_index('date').loc[idate]['qty']

                temp = temp.round(4)

                orderbook = orderbook.append(temp,ignore_index=True)


        #display(orderbook.sort_values(by=['date','orderside'], ascending=True))

        funds = 1000
        BuyFundsPercent = .75
        percentHeldOnSell = 1

        buyLog = pd.DataFrame()
        sellLog = pd.DataFrame()
        runningLog = pd.DataFrame()

        held = 0
        upper = 0
        lower = 0

        for i in dateindex2:

            temp = pd.DataFrame()
            rtemp = pd.DataFrame()
            _temp = pd.DataFrame()

            t = i.strftime('%Y-%m-%d')

            subset = orderbook[orderbook['date']==t]
            gain = 0
            paid = 0
            
            Qty = 0

            if len(subset) != 0:

                sales = subset[subset['orderside'] == 'sell']

                #print("date " + str(i))

                if len(sales) != 0:                        

                    oldvalue = sales['valueAtPurchase'].values[0]

                    newvalue = sales['valueAtSale'].values[0]            

                    Qty = buyLog.set_index('date').loc[sales['dateBought'].values[0]].values[0]
                    #print("Qty sold " + str(Qty.round(2)))

                    gain = newvalue * Qty

                    _temp['date'] = [i]
                    _temp['qty'] = [Qty]

                    sellLog = sellLog.append(_temp)

                purchases = subset[subset['orderside'] == 'buy']

                if len(purchases) != 0:

                    #ProportionOfFunds = funds * BuyFundsPercent
                    
                    Qty = 0                    
                    
                    paid = 0
                    
                    #reduced total return (when combined macd/sma)
                    #for c in range(1,int(purchases['conditions'].values[0])):
                        
                    ProportionOfFunds = funds * BuyFundsPercent

                    Qty = ProportionOfFunds / purchases['valueAtPurchase'].values[0] + Qty

                    paid = purchases['valueAtPurchase'].values[0]*Qty + paid                        

                    #print(purchases['valueAtPurchase'].values[0])
                    #print("Qty purchased " + str(Qty.round(2)))

                    temp['date'] = [i]
                    temp['qty'] = [Qty]

                    buyLog = buyLog.append(temp)

                funds = funds + gain - paid

                rtemp['date'] =  [i]
                rtemp['funds'] =  [funds]

                if len(sellLog) != 0:
                    remainder = (sum(buyLog['qty'])-sum(sellLog['qty']))            

                else:
                    remainder = (sum(buyLog['qty']))

                rtemp['held'] = remainder
                rtemp['value'] = remainder * data.loc[t]['Close']
                rtemp['portValue'] = funds + remainder * data.loc[t]['Close']

                #print("in " + str(gain))
                #print("out " + str(paid))
                #print("held: " + str(remainder))
                #print("Close Value: " + str(data.loc[i]['Close']))
                #print("held Value: " + str(remainder * data.loc[i]['Close']))
                #print("funds " + str(funds))
                #print("portValue " + str(funds + remainder * data.loc[i]['Close']))
                #print()

                runningLog = runningLog.append(rtemp)


        ret_data =  runningLog.set_index('date')['portValue'].pct_change()
        cumulative_ret_data = (ret_data + 1).cumprod()

        #ret_data2 = data[runningLog.set_index('date').index[1]:runningLog.set_index('date').index[-1]]['Close'].pct_change()
        ret_data2 = data[(end_date1+timedelta(days=1)).strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]['Close'].pct_change().copy()
        cum_ret_data2 = (ret_data2 + 1).cumprod()

        #sp500_data = benchData[runningLog.set_index('date').index[1]:runningLog.set_index('date').index[-1]]['Close'].pct_change()
        plt.plot(cumulative_ret_data,label=" strategy @ " + str(BuyFundsPercent) )

        #This is already plotted earlier
        #plt.plot(cum_ret_data2,label=" hold")        

        #runningLog

        #plt.show()
        strategies.append(cumulative_ret_data.iloc[-1])
        holds.append(cum_ret_data2.iloc[-1])
        print("strategy: " + str(cumulative_ret_data.iloc[-1]) + " vs hold: " + str(cum_ret_data2.iloc[-1]))

plt.legend(loc="upper left",fontsize=8)

plt.xticks(rotation=30) 


# In[ ]:





# In[ ]:



print("SP500: " + str(sp500_cumulative_ret_data.iloc[-1]))

print("Strategy: " + str(pd.DataFrame(strategies).mean().values[0]))
print("Hold: " + str(pd.DataFrame(holds).mean().values[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#plt.plot((set['ExpectedReturn']+1))
#plt.xticks(rotation=30) 
#len(set['ExpectedReturn']+1)


# In[ ]:


#plt.hist(runningLog['portValue'].dropna().pct_change(), bins='auto')  # arguments are passed to np.histogram
#print(runningLog['portValue'].dropna().pct_change().sum())


# In[ ]:




