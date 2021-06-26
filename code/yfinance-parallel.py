#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#inspiration
#https://simply-python.com/2019/01/16/retrieving-stock-statistics-from-yahoo-finance-using-python/
#https://towardsdatascience.com/backtest-your-trading-strategy-with-only-3-lines-of-python-3859b4a4ab44?source=friends_link&sk=ec647b6bb43fe322013248fd1d473015
#https://simply-python.com/2019/01/16/retrieving-stock-statistics-from-yahoo-finance-using-python/

get_ipython().system('pip install --upgrade pip')
#!pip install 

import concurrent.futures
import urllib
import re
import time
import stat
import contextlib
import functools
get_ipython().system('pip install datetime yfinance pandas pandas_market_calendars pandas_datareader mpl-finance stockstats matplotlib fastquant numpy stockstats cryptography mplfinance plotly finta fbprophet tabulate sklearn pandas_ta dataclasses')


# In[ ]:


from concurrent.futures import wait, ALL_COMPLETED
import urllib.request
import datetime
from datetime import timedelta
from datetime import date
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
import os.path
from os import path
from functools import reduce
#import pandas-datareader
#import mpl-finance
from fbprophet.diagnostics import cross_validation
import stockstats
import matplotlib as mpl
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
from fastquant import backtest, get_stock_data
import numpy as np
import tabulate
import mplfinance as mpf
import matplotlib.dates as mdates
from fbprophet import Prophet
from finta import TA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error


# In[ ]:


def MAPE(Y_actual,Y_Predicted):
    mape_ = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape_


# In[ ]:





# In[ ]:


pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

def wwma(values, n):
    """
     J. Welles Wilder's EMA 
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

#weighted moving average
def atrVolume(df, n=50):
    #data = df.copy()
    high = df['high']*df['volume']
    low = df['low']*df['volume']
    close = df['close']*df['volume']
    data = pd.DataFrame()
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr

def bbands(df, n=20):
    #data = df.copy()
    high = df['high']*df['volume']
    low = df['low']*df['volume']
    #close = data['close']
    data = pd.DataFrame()
    upper_band = ta.ema(high, length=n)
    lower_band = ta.ema(low, length=n)
    upper_sdev = np.std(upper_band)
    lower_sdev = np.std(upper_band)
    data['bbands_upper'] = upper_band + upper_sdev * sdev_constant
    data['bbands_lower'] = lower_band - lower_sdev * sdev_constant
    return data

#https://tcoil.info/compute-rsi-for-stocks-with-python-relative-strength-index/#:~:text=RSI%20indicator%20(Relative%20Strength%20Index,1001%2Brsn
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()      # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

cores = int(len(os.sched_getaffinity(0))/2)

pool1 = concurrent.futures.ProcessPoolExecutor(cores)

w = 117

end = datetime.date.today()
#- timedelta(weeks=w*2)
start = end - timedelta(weeks=w)

start2 = end - timedelta(weeks=52)

one_week_end = start
one_week_start = one_week_end - timedelta(weeks=1)

#need to do the two pass trick (i.e. find stocks fully populated a week 9 quarters back)

nyse = mcal.get_calendar('NYSE')
official_trading_dates= nyse.schedule(start_date=start, end_date=end)

#date_time_obj_start = datetime.datetime.strptime(start, '%Y-%m-%d')
date_time_obj_start = start
#date_time_obj_end = datetime.datetime.strptime(end, '%Y-%m-%d')
date_time_obj_end = end

official_trading_dates_plus5= nyse.schedule(start_date=date_time_obj_start.strftime('%Y-%m-%d'), end_date=date_time_obj_end+timedelta(days=5))

next_trading_day = official_trading_dates_plus5.index[official_trading_dates_plus5.index > end.strftime('%m/%d/%Y')][0]

idx2 = official_trading_dates.index

#type(next_trading_day)
#used for forecast only
idx3 = official_trading_dates_plus5.index[(official_trading_dates_plus5.index >= start.strftime('%Y-%m-%d')) & (official_trading_dates_plus5.index <= next_trading_day.strftime('%Y-%m-%d'))]

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
    else:
      print("equal dates, not redownloading")
    
else:
    print("downloading nasdaqtraded.txt")
    urllib.request.urlretrieve(url, 'nasdaqtraded.txt')
    
df = pd.read_csv('nasdaqtraded.txt', sep='|')[0:-1]

#process symbols for bad characters
BAD_CHARS = ['$','.']
pat = '|'.join(['({})'.format(re.escape(c)) for c in BAD_CHARS])

df = df[~df['Symbol'].str.contains(pat)]

#choose size
size=1600
stocks = list(df["Symbol"].sample(n=size))

def dl_one_week(stock):
    return yf.download(stock, start=one_week_start, end=one_week_end, auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')

futures1 = [pool1.submit(dl_one_week, args) for args in stocks]
wait(futures1, timeout=None, return_when=ALL_COMPLETED)

stocks_data_one_week = pd.DataFrame()

for x in range(0,len(stocks)):
    prices = pd.DataFrame(futures1[x].result())
    prices['Symbol'] = stocks[x]
    prices = prices.loc[~prices.index.duplicated(keep='last')]        
    prices = prices.reset_index()
                
    stocks_data_one_week = pd.concat([stocks_data_one_week,prices])
    
stocks_data_one_week

#stocks that existed 9 quarters ago
vetted_symbols = list(stocks_data_one_week.Symbol.unique())

pool2 = concurrent.futures.ProcessPoolExecutor(cores)

def dl(stock):
    return yf.download(stock, start=start, end=end).iloc[:, :6].dropna(axis=0, how='any')

futures2 = [pool2.submit(dl, args) for args in vetted_symbols]
wait(futures2, timeout=None, return_when=ALL_COMPLETED)

stocks_data = pd.DataFrame()

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
        stocks_data = pd.concat([stocks_data,df])

stocks_data.to_csv(start.strftime('%Y-%m-%d')+'-'+end.strftime('%Y-%m-%d')+'-'+str(len(vetted_symbols))+'stocks_data.csv', index = False)


# In[ ]:





# In[ ]:


returnsdf = pd.DataFrame()
returnsl = []

#half = int(len(official_trading_dates)/2)

#HorizonPeriod = half
#weeks - 52 * 252 trading days a year / 52 = # of trading days
HorizonPeriod = int((w-39)*252/52)
HoldoutPeriod = int((w-52)*252/52)

#cumulative returns of 1st half
for i in vetted_symbols:
    subset = stocks_data[stocks_data["Symbol"]==i][1:HoldoutPeriod-1]
    #print(subset)
    price_data = subset["Adj Close"]
    #print(price_data)
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()
    
    last = cumulative_ret[len(cumulative_ret)]
    
    #pd.concat(last,returns)
    returnsl.append(last)

    #plt.plot(cumulative_ret, label=i)
    #plt.legend(loc="upper left",fontsize=8)
    
returnsdf["returns"] = returnsl
returnsdf["stock"] = vetted_symbols

tenpercent = int(len(vetted_symbols)*.1)

top10percent = returnsdf.sort_values(by=['returns'], ascending=False)[1:tenpercent]

#top10percent = top10percent[top10percent['returns']>2.5]

bottom10percent = returnsdf.sort_values(by=['returns'], ascending=True)[1:tenpercent]

print(top10percent)
print(bottom10percent)
#returnsdf

#cumulative returns past HorizonPeriod
for i in top10percent["stock"]:
    subset = stocks_data[stocks_data["Symbol"]==i][HorizonPeriod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    
    price_data = stock["adj close"]
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()

    plt.plot(cumulative_ret, label=i)
    plt.legend(loc="upper left",fontsize=8)
    
plt.show()

#cumulative returns past HorizonPeriod
for i in bottom10percent["stock"]:
    subset = stocks_data[stocks_data["Symbol"]==i][HorizonPeriod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    
    price_data = stock["adj close"]
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()

    plt.plot(cumulative_ret, label=i)
    plt.legend(loc="upper left",fontsize=8)
    
stocklist = list(top10percent["stock"])
print(stocklist)
    


# In[ ]:





# In[ ]:





# In[ ]:


sdev_constant = 2.33

creturns = pd.DataFrame()
hreturns = pd.DataFrame()

for x in stocklist:
    print(x)

    subset = stocks_data[stocks_data["Symbol"]==x]
    
    #[HorizonPeriod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])

    #stock.columns = ["open", "close", "adj close", "low", "close", "volume"]
    stock.index.names = ['dt']

    #EVWMA
    Short_EVWMA = pd.DataFrame(TA.EVWMA(stock,12))
    Long_EVWMA = pd.DataFrame(TA.EVWMA(stock,26))
    Short_EVWMA.columns = ['EVWMA_12']
    Long_EVWMA.columns = ['EVWMA_26']

    #p 209 of ttr doc
    MACD_EVWMA = pd.DataFrame(Short_EVWMA['EVWMA_12'] - Long_EVWMA['EVWMA_26'])
    MACD_EVWMA.columns = ['MACD-line']

    Signal_EVWMA = pd.DataFrame(ta.ema(MACD_EVWMA["MACD-line"], length=9))
    Signal_EVWMA.columns = ['Signal_EMA_9_MACD']
    Signal_EVWMA
    stock['custom'] = Signal_EVWMA['Signal_EMA_9_MACD']
    stock['rsi'] = computeRSI(stock['close']*stock['volume'], 14)
    stock[['bbands_upper', 'bbands_lower']] = bbands(stock)
    stock['atr'] = atrVolume(stock)
    
    stock = stock[HoldoutPeriod:(HorizonPeriod-1)]
    #stock = stock[HoldoutPeriod:]
    
    #stock = stock.set_index('dt')

    mergeCompare = pd.merge(stock['close'].shift(+1),stock['close'], how='inner', left_index=True, right_index=True)

    actualReturns = ((mergeCompare['close_y']-mergeCompare['close_x'])/mergeCompare['close_x'])

    orderbook = pd.DataFrame()

    funds = 1000
    BuyFundsPercent = 1
    percentHeldOnSell = 1

    held = 0
    upper = 0
    lower = 0

    for i in stock.index:
        #print(df.loc[i]['custom'])
        temp = pd.DataFrame()

        estRet = stock.loc[i]['custom']
        rsi = stock.loc[i]['rsi']
        rsiDelta = (stock['rsi']-stock['rsi'].shift(+1))[i]

        #if (estRet > upper):
        #if (estRet > upper and rsi > 20 and rsiDelta > 0)
        if (estRet > upper and rsi > 20 and rsiDelta > 0 and (stock.loc[i]['close']*stock.loc[i]['volume'] < stock.loc[i]['bbands_upper'])):
            temp['order'] = ['buy']

            ProportionOfFunds = funds * BuyFundsPercent
            Qty = ProportionOfFunds / stock.loc[i]['close']
            value = stock.loc[i]['close']*Qty

            funds = funds - value
            held = held + Qty

        #what about over 80 RSI?  Guess it's a hold until it drops?
        #elif (estRet < lower):
        #elif (estRet < lower and rsi < 80 and rsiDelta < 0)
        elif (estRet < lower and rsi < 80 and rsiDelta < 0 and (stock.loc[i]['close']*stock.loc[i]['volume'] > stock.loc[i]['bbands_lower'])):
            temp['order'] = ['sell']

            Qty = held*percentHeldOnSell
            value = stock.loc[i]['close']*Qty

            funds = funds + value
            held = held - Qty

        #if ((estRet < upper) & (etsRet > lower))
        #if ((estRet > lower) and (estRet < upper)):
        else:
            temp['order'] = ['hold']

            Qty = 0
            value = stock.loc[i]['close']*Qty

            funds = funds + value
            held = held - Qty

        temp['date'] = [i]
        temp['estRet'] = estRet
        temp['actRet'] = actualReturns.loc[i]
        temp['price'] = stock.loc[i]['close']
        temp['PropInvValue'] = stock.loc[i]['close']*Qty
        temp['qtyShares'] = Qty
        temp['funds'] = funds
        temp['portValue'] = funds + stock.loc[i]['close']*held
        temp['held'] = held

        temp = temp.round(2)

        orderbook = orderbook.append(temp)

    display(orderbook.dropna().head())
    display(orderbook.dropna().tail())

    import matplotlib.pyplot as plt
    plt.scatter(orderbook['estRet'], orderbook['actRet'],label="x: estRet; y:actRet")
    plt.xlabel('estRet')
    plt.ylabel('actRet')
    plt.show()

    from scipy.stats import pearsonr

    #cov(orderbook['estRet'], orderbook['actRet'])
    corrprep = pd.merge(orderbook.set_index('date')['estRet'][1:],orderbook.set_index('date')['actRet'][1:], how='inner', left_index=True, right_index=True)
    corr, _ = pearsonr(corrprep.dropna()['estRet'][1:], corrprep.dropna()['actRet'][1:])
    print('Pearsons correlation: %.3f' % corr)#df

    print("Strat Return")
    print(orderbook['portValue'][-1:].values[0]/funds)

    print("Hold Return")
    print(stock['close'][-1:].values[0]/stock['close'][0:].values[0])
    
    if (stock['close'][-1:].values[0]/stock['close'][0:].values[0]) > (orderbook['portValue'][-1:].values[0]/funds)*1.5:
        print("removing")
        print(x)
        stocklist.remove(x)

    #plt.plot(orderbook.set_index('date')['portValue'])

    values = orderbook.set_index('date')['portValue']
    #print(len(values))
    ret_value = values.pct_change()[1:]
    cumulative_ret_value = (ret_value + 1).cumprod()

    #show cumulative charts
    og_ret_data = orderbook.set_index('date')['price']
    og_ret_value = og_ret_data.pct_change()[1:]
    #og_merge = pd.merge(df,custom.set_index('dt'), how='inner', left_index=True, right_index=True)
    cumulative_og_ret_data = (og_ret_value + 1).cumprod()

    plt.plot(cumulative_ret_value)
    creturns = creturns.append(pd.DataFrame(cumulative_ret_value[-1:]))
    
    plt.plot(cumulative_og_ret_data)
    hreturns = hreturns.append(pd.DataFrame(cumulative_og_ret_data[-1:]))

    plt.show()

    plt.plot(stock['bbands_upper'])
    plt.plot((stock['close']*stock['volume']))
    plt.plot(stock['bbands_lower'])
    
    plt.show()

print("hold")
print(hreturns.mean())
print("strategy")
print(creturns.mean())

print("SP500")
bench = dl("^GSPC").iloc[:, :6].dropna(axis=0, how='any')
sp500_data = bench[HoldoutPeriod:(HorizonPeriod-1)]['Close']
sp500_ret_value = sp500_data.pct_change()[1:]
sp500_cumulative_ret_data = (sp500_ret_value + 1).cumprod()
sp500_cumulative_ret_data[-1:]


# In[ ]:


sdev_constant = 2.33

creturns = pd.DataFrame()
hreturns = pd.DataFrame()

for i in stocklist:
    print(i)

    subset = stocks_data[stocks_data["Symbol"]==i]
    
    #[HorizonPeriod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])

    #stock.columns = ["open", "close", "adj close", "low", "close", "volume"]
    stock.index.names = ['dt']

    #EVWMA
    Short_EVWMA = pd.DataFrame(TA.EVWMA(stock,12))
    Long_EVWMA = pd.DataFrame(TA.EVWMA(stock,26))
    Short_EVWMA.columns = ['EVWMA_12']
    Long_EVWMA.columns = ['EVWMA_26']

    #p 209 of ttr doc
    MACD_EVWMA = pd.DataFrame(Short_EVWMA['EVWMA_12'] - Long_EVWMA['EVWMA_26'])
    MACD_EVWMA.columns = ['MACD-line']

    Signal_EVWMA = pd.DataFrame(ta.ema(MACD_EVWMA["MACD-line"], length=9))
    Signal_EVWMA.columns = ['Signal_EMA_9_MACD']
    Signal_EVWMA
    stock['custom'] = Signal_EVWMA['Signal_EMA_9_MACD']
    stock['rsi'] = computeRSI(stock['close']*stock['volume'], 14)
    stock[['bbands_upper', 'bbands_lower']] = bbands(stock)
    stock['atr'] = atrVolume(stock)
    
    stock = stock[HorizonPeriod:]
    #stock = stock[HoldoutPeriod:]
    
    #stock = stock.set_index('dt')

    mergeCompare = pd.merge(stock['close'].shift(+1),stock['close'], how='inner', left_index=True, right_index=True)

    actualReturns = ((mergeCompare['close_y']-mergeCompare['close_x'])/mergeCompare['close_x'])

    orderbook = pd.DataFrame()

    funds = 1000
    BuyFundsPercent = 1
    percentHeldOnSell = 1

    held = 0
    upper = 0
    lower = 0

    for i in stock.index:
        #print(df.loc[i]['custom'])
        temp = pd.DataFrame()

        estRet = stock.loc[i]['custom']
        rsi = stock.loc[i]['rsi']
        rsiDelta = (stock['rsi']-stock['rsi'].shift(+1))[i]

        #if (estRet > upper):
        #if (estRet > upper and rsi > 20 and rsiDelta > 0)
        if (estRet > upper and rsi > 20 and rsiDelta > 0 and (stock.loc[i]['close']*stock.loc[i]['volume'] < stock.loc[i]['bbands_upper'])):
            temp['order'] = ['buy']

            ProportionOfFunds = funds * BuyFundsPercent
            Qty = ProportionOfFunds / stock.loc[i]['close']
            value = stock.loc[i]['close']*Qty

            funds = funds - value
            held = held + Qty

        #what about over 80 RSI?  Guess it's a hold until it drops?
        #elif (estRet < lower):
        #elif (estRet < lower and rsi < 80 and rsiDelta < 0)
        elif (estRet < lower and rsi < 80 and rsiDelta < 0 and (stock.loc[i]['close']*stock.loc[i]['volume'] > stock.loc[i]['bbands_lower'])):
            temp['order'] = ['sell']

            Qty = held*percentHeldOnSell
            value = stock.loc[i]['close']*Qty

            funds = funds + value
            held = held - Qty

        #if ((estRet < upper) & (etsRet > lower))
        #if ((estRet > lower) and (estRet < upper)):
        else:
            temp['order'] = ['hold']

            Qty = 0
            value = stock.loc[i]['close']*Qty

            funds = funds + value
            held = held - Qty

        temp['date'] = [i]
        temp['estRet'] = estRet
        temp['actRet'] = actualReturns.loc[i]
        temp['price'] = stock.loc[i]['close']
        temp['PropInvValue'] = stock.loc[i]['close']*Qty
        temp['qtyShares'] = Qty
        temp['funds'] = funds
        temp['portValue'] = funds + stock.loc[i]['close']*held
        temp['held'] = held

        temp = temp.round(2)

        orderbook = orderbook.append(temp)

    display(orderbook.dropna().head())
    display(orderbook.dropna().tail())

    import matplotlib.pyplot as plt
    plt.scatter(orderbook['estRet'], orderbook['actRet'],label="x: estRet; y:actRet")
    plt.xlabel('estRet')
    plt.ylabel('actRet')
    plt.show()

    from scipy.stats import pearsonr

    #cov(orderbook['estRet'], orderbook['actRet'])
    corrprep = pd.merge(orderbook.set_index('date')['estRet'][1:],orderbook.set_index('date')['actRet'][1:], how='inner', left_index=True, right_index=True)
    corr, _ = pearsonr(corrprep.dropna()['estRet'][1:], corrprep.dropna()['actRet'][1:])
    print('Pearsons correlation: %.3f' % corr)#df

    print("Strat Return")
    print(orderbook['portValue'][-1:].values[0]/funds)

    print("Hold Return")
    print(stock['close'][-1:].values[0]/stock['close'][0:].values[0])
    
    #plt.plot(orderbook.set_index('date')['portValue'])

    values = orderbook.set_index('date')['portValue']
    #print(len(values))
    ret_value = values.pct_change()[1:]
    cumulative_ret_value = (ret_value + 1).cumprod()

    #show cumulative charts
    og_ret_data = orderbook.set_index('date')['price']
    og_ret_value = og_ret_data.pct_change()[1:]
    #og_merge = pd.merge(df,custom.set_index('dt'), how='inner', left_index=True, right_index=True)
    cumulative_og_ret_data = (og_ret_value + 1).cumprod()

    plt.plot(cumulative_ret_value)
    creturns = creturns.append(pd.DataFrame(cumulative_ret_value[-1:]))
    
    plt.plot(cumulative_og_ret_data)
    hreturns = hreturns.append(pd.DataFrame(cumulative_og_ret_data[-1:]))

    plt.show()

    plt.plot(stock['bbands_upper'])
    plt.plot((stock['close']*stock['volume']))
    plt.plot(stock['bbands_lower'])
    
    plt.show()

print("hold")
print(hreturns.mean())
print("strategy")
print(creturns.mean())

print("SP500")
bench = dl("^GSPC").iloc[:, :6].dropna(axis=0, how='any')
sp500_data = bench[HorizonPeriod:]['Close']
sp500_ret_value = sp500_data.pct_change()[1:]
sp500_cumulative_ret_data = (sp500_ret_value + 1).cumprod()
sp500_cumulative_ret_data[-1:]


# In[ ]:


tgt_website = r'https://sg.finance.yahoo.com/quote/WDC/key-statistics?p=WDC'
 
def get_key_stats(tgt_website):
 
    # The web page is make up of several html table. By calling read_html function.
    # all the tables are retrieved in dataframe format.
    # Next is to append all the table and transpose it to give a nice one row data.
    df_list = pd.read_html(tgt_website)
    result_df = df_list[0]
 
    for df in df_list[1:]:
        result_df = result_df.append(df)
 
    # The data is in column format.
    # Transpose the result to make all data in single row
    return result_df.set_index(0).T

all_result_df = pd.DataFrame()
url_prefix = 'https://sg.finance.yahoo.com/quote/{0}/key-statistics?p={0}'
for sym in stocklist:
    stock_url = url_prefix.format(sym)
    result_df = get_key_stats(stock_url)
    if len(all_result_df) ==0:
        all_result_df = result_df
    else:
        all_result_df = all_result_df.append(result_df)

all_result_df.dropna(axis=0, how='any')
all_result_df['Symbol'] = stocklist
# Save all results
all_result_df.to_csv('results.csv', index =False)
all_result_df

