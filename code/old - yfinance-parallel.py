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


# In[136]:


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
size=2000
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
        
    stocks_data = pd.concat([stocks_data,df])

stocks_data.to_csv(start.strftime('%Y-%m-%d')+'-'+end.strftime('%Y-%m-%d')+'-'+str(len(vetted_symbols))+'stocks_data.csv', index = False)


# In[143]:


top10percent = top10percent[top10percent['returns']>2.5]
stocklist = list(top10percent["stock"])


# In[137]:


returnsdf = pd.DataFrame()
returnsl = []

half = int(len(official_trading_dates)/2)

w = 117

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

top10percent = top10percent[top10percent['returns']>2.5]

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
    


# In[134]:





# In[144]:


sdev_constant = 2.33

creturns = pd.DataFrame()
hreturns = pd.DataFrame()

for i in stocklist:
    print(i)

    subset = stocks_data[stocks_data["Symbol"]==i]
    subset = subset.dropna()[HoldoutPeriod:(HorizonPeriod-1)]
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
    stock['RSI'] = computeRSI(stock['close']*stock['volume'], 14)
    stock[['bbands_upper', 'bbands_lower']] = bbands(stock)
    stock['ATR'] = atrVolume(stock)
    
    #stock = stock.set_index('dt')

    mergeCompare = pd.merge(stock['close'].shift(+1),stock['close'], how='inner', left_index=True, right_index=True)

    actualReturns = ((mergeCompare['close_y']-mergeCompare['close_x'])/mergeCompare['close_x'])

    orderbook = pd.DataFrame()

    funds = 1000
    BuyFundsPercent = .5
    percentHeldOnSell = 1

    held = 0
    upper = 0
    lower = 0

    for i in stock.index:
        #print(df.loc[i]['custom'])
        temp = pd.DataFrame()

        estRet = stock.loc[i]['custom']
        rsi = stock.loc[i]['RSI']
        rsiDelta = (stock['RSI']-stock['RSI'].shift(+1))[i]

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

    print("Total Return")
    print(orderbook['portValue'][-1:].values[0]/1000)

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

    plt.plot(stock.dropna()['bbands_upper'])
    plt.plot((stock.dropna()['close']*stock.dropna()['volume']))
    plt.plot(stock.dropna()['bbands_lower'])
    
    plt.show()

print("hold")
print(hreturns.mean())
print("strategy")
print(creturns.mean())


# In[ ]:


for i in stocklist:

    subset = stocks_data[stocks_data["Symbol"]==i]
    
    #[HorizonPeriod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    stock.BOLL_WINDOW = 20
    stock.BOLL_STD_TIMES = 2

    price_data = stock["adj close"]

    subset.set_index(subset['Date'], inplace=True) 
    subset.index.name = 'Date'

    #prophet
    ts = subset[["Date","Adj Close"]]
    ts.columns = ['ds', 'y']
    #print(ts)
    #m = Prophet(daily_seasonality=True, yearly_seasonality=True).fit(ts)
    m = Prophet(daily_seasonality=True,yearly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
    m.fit(ts[HorizonPeriod:])
    #forecast = m.make_future_dataframe(periods=0, freq='D')
    forecast = pd.DataFrame(idx3)
    forecast.columns = ['ds']

    # Predict and plot
    pred = m.predict(forecast)

    df = stock
    idx1 = df.index  
    #create entry for next trading day (i.e. idx3)
    merged = idx1.union(idx3)

    expected_1day_return = pred.set_index("ds").yhat.pct_change().shift(-1).multiply(100)
    df["custom"] = expected_1day_return.multiply(-1)

    newdf = df.reindex(merged)

    df = newdf
    #print(df)

    m.plot(pred[HorizonPeriod:])
    plt.title('Prophet: Forecasted Daily Closing Price', fontsize=25)

    #exponential smoothing VAMA
    a = pd.DataFrame(TA.EVWMA(subset))
    #ATR
    b = pd.DataFrame(TA.ATR(subset))

    fit3 = SimpleExpSmoothing(a, initialization_method="estimated").fit()
    fcast3 = fit3.forecast(1).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

    #weighted moving averages
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6}) # add your own style here
    fig = mpf.figure(figsize=(10, 7), style=s)
    ax = fig.add_subplot(2,1,1)
    av = fig.add_subplot(2,1,2, sharex=ax)
    #az = fig.add_subplot(3,1,1)
    mpf.plot(subset[HorizonPeriod:],type='candle',mav=(3,6,9),volume=av,show_nontrading=True, ax=ax)

    my_dpi = 50
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), dpi=my_dpi)

    # title for entire figure
    fig.suptitle(i, fontsize=20)

    #other technical
    axes[0, 0].plot(stock[HorizonPeriod:]["macds"], color="m", label="Signal Line")    
    #axes[0, 1].set_title('Subplot 1', fontsize=14)
    axes[0, 0].plot(stock[HorizonPeriod:]["macd"], color="y", label="MACD")
    axes[0, 0].legend(loc="lower right",fontsize=14)

    axes[0, 1].plot(stock[HorizonPeriod:]["close_10_sma"], color="b", label="SMA")
    axes[0, 1].plot(stock[HorizonPeriod:]["close_12_ema"], color="r", label="EMA")
    axes[0, 1].plot(stock[HorizonPeriod:]["adj close"], color="g", label="Adj Close prices")
    axes[0, 1].plot(stock[HorizonPeriod:]['boll'], color="b", label="BBands")
    axes[0, 1].plot(stock[HorizonPeriod:]['boll_ub'], color="b", label="BBands")
    axes[0, 1].plot(stock[HorizonPeriod:]['boll_lb'], color="b", label="BBands")
    axes[0, 1].plot(stock[HorizonPeriod:]["adj close"], color="g", label="Adj Close prices")
    axes[0, 1].legend(loc="lower right",fontsize=14)

    axes[1, 0].plot(a, color="b", label="Exp Smooth")
    axes[1, 0].plot(fit3.fittedvalues[HorizonPeriod:], marker='o', color='green')
    #line3, = axes[1, 0].plot(fcast3, marker='o', color='green')
    #axes[1, 0].plot(fcast3, marker='o', color='orange')
    #axes[1, 0].legend(loc="lower right", label=fcast3.name)
    #print(pred[len(pred)])
    #print(df["custom"])
    print("VAMA forecast")
    print(fcast3)
    print("fbprophet price forecast")
    #print(pred[len(pred)-1:][['ds','yhat']])
    print(pred[-1:][['ds','yhat']])
    print("expected return")    
    print(pred['yhat'].pct_change()[-1:])

    axes[1, 1].plot(stock[HorizonPeriod:]['rsi_14'], color="b", label="RSI_14")
    axes[1, 1].legend(loc="lower right",fontsize=14)

    #print(stock)

    #axes[2, 1].set_xlabel('cumret', fontsize=14)
    #axes[2, 1].set_title('', fontsize=14)

    plt.show()


# In[ ]:


strats = { 
    #"smac": {"fast_period": 35, "slow_period": 50}, 
    #"rsi": {"rsi_lower": 30, "rsi_upper": 70},
    #"macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "sma_period": 30, "dir_period": 10},
    #"bbands": {"period": 20, "devfactor": 2.0},
    
    #needs to be custom
    #"ema": {"fast_period": 10, "slow_period": 30},
    
    "custom": {"upper_limit": 1.5, "lower_limit":-1.5}
} 

strats_opt = { 
    #"smac": {"fast_period": [35, 50], "slow_period": [100, 200]}, 
    #"rsi": {"rsi_lower": [15, 30], "rsi_upper": [70, 85]},
    #"macd": {"fast_period": [12], "slow_period": [26], "signal_period": [9], "sma_period": [30], "dir_period": [10]},
    #"bbands": {"period": [20], "devfactor": [2.0]},
    
    #"ema": {"fast_period": [9,10,12], "slow_period": [30, 40, 50]},     
    "custom": {"upper_limit": [1.5], "lower_limit":[-1.5]}
}         

pool3 = concurrent.futures.ProcessPoolExecutor(cores)

def back_test(i):
    subset = stocks_data[stocks_data["Symbol"]==i]
    #[HorizonPeriod:]

    #converts date to datetime
    stock = StockDataFrame.retype(subset[["Date","Open", "High", "Low", "Close", "Adj Close", "Volume"]])

    #EVWMA
    Short_EVWMA = pd.DataFrame(TA.EVWMA(subset,9))
    Signal_EVWMA = pd.DataFrame(TA.EVWMA(subset,12))
    Long_EVWMA = pd.DataFrame(TA.EVWMA(subset,26))
    ATR = pd.DataFrame(TA.ATR(subset))
    Short_EVWMA.columns = ['EVWMA_9']
    Signal_EVWMA.columns = ['EVWMA_12']
    Long_EVWMA.columns = ['EVWMA_26']
    ATR.columns = ['ATR']
    MACD_EVWMA = pd.DataFrame(Long_EVWMA['EVWMA_26'] - Short_EVWMA['EVWMA_9'])
    MACD_EVWMA.columns = ['Signal']

    #Adj Close
    ts = subset[["Date","Adj Close"]]
    ts.columns = ['ds', 'y']

    #print(ts)
    m = Prophet(daily_seasonality=True,yearly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
    m.fit(ts[HorizonPeriod:])
    #forecast = m.make_future_dataframe(periods=0, freq='D')
    forecast = pd.DataFrame(idx3)
    forecast.columns = ['ds']

    #forecast2 = pd.DataFrame(df.index)[HorizonPeriodho+50:]
    #forecast2.columns = ['ds']

    # Predict and plot
    pred = m.predict(forecast)

    #pred2 = m.predict(forecast2)

    dfpre = stock
    idx1 = dfpre.index  
    #create entry for next trading day (i.e. idx3)
    merged = idx1.union(idx3)

    newdf = dfpre.reindex(merged)

    #A. fbprophet return
    expected_1day_return = pred.set_index("ds").yhat.pct_change().shift(-1).multiply(100)
    newdf["custom"] = expected_1day_return.multiply(-1)

    #B. no fbprophet
    #newdf['custom'] = ts.set_index('ds')
    #fbprophet
    #newdf["custom"] = pred.set_index('ds')['yhat']

    #rmse
    #delta = len(pred.set_index('ds')['yhat'][HorizonPeriod:])-len(df['close'].dropna())
    rmse = mean_squared_error(newdf['close'].dropna()[HoldoutPeriod:(HorizonPeriod-1)], pred.set_index('ds')['yhat'][HoldoutPeriod:(HorizonPeriod-1)], squared=True)
    mape = MAPE(newdf['close'].dropna()[HoldoutPeriod:(HorizonPeriod-1)], pred.set_index('ds')['yhat'][HoldoutPeriod:(HorizonPeriod-1)])

    #df = dfpre[(dfpre['Date']> "2018-01-01") & (df['Date']<= end)]
    df = newdf[HorizonPeriod:]
    #df["custom"] = (((pred.set_index('ds')['yhat']-ts.set_index('ds')['y'])/ts.set_index('ds')['y']).multiply(-1))[HorizonPeriod:]

    with contextlib.redirect_stdout(None):
        b = backtest("multi", df.dropna(), strats=strats_opt, return_history=True, buy_prop=0.10, sell_prop=1,commission=0.01, init_cash=1000)

    r = {'backtest':b, 'score1':rmse, 'score2':mape, 'name':i, 'forecast':pred}
    return (r)

futures_back = [pool3.submit(back_test, args) for args in stocklist]
wait(futures_back, timeout=None, return_when=ALL_COMPLETED)

res = []
for x in range(0,len(stocklist)):
    res.append(futures_back[x].result())
    
res_data = pd.DataFrame()    
for i in range(0,len(res)):
    res_data = pd.concat([res[i]['backtest'][0],res_data])
    


# In[ ]:


res_names = []

for x in range(0,len(res)):
    res_names.append(res[x]['name'])
    
res_names
stocklist


# In[ ]:





# In[ ]:


#top10percent symbols

strategies = list(res_data.strat_id.unique())
strategies.sort()

#res_data.filter(like='bbi', axis=0)

scores = []
#scores = pd.DataFrame()

for i in strategies:
    #res_data.filter(like=i, axis=0)
    filtered =  res_data['strat_id']==i
    #print(i)
    scores.append((res_data[filtered]["final_value"].mean()))
    #scores = pd.concat(res_data[filtered]["final_value"].mean(),scores)
    
index_min = min(range(len(scores)), key=scores.__getitem__)
index_max = max(range(len(scores)), key=scores.__getitem__)    

choice=index_max

filtered =  res_data['strat_id']==choice
print(res_data[filtered]["final_value"].mean())

print("Choice strategy ",choice)
res_data[res_data['strat_id']==choice].round(2)


# In[ ]:





# In[ ]:





# In[ ]:


res_rmse = []
for i in range(0,len(res)):
    res_rmse.append(res[i]['score1']) 

res_mape = []
for i in range(0,len(res)):
    res_mape.append(res[i]['score2'])
    
res_names = []
for i in range(0,len(res)):
    res_names.append(res[i]['name']) 

res_final = []
for i in range(0,len(res)):
    res_final.append(res[i]['backtest'][0]['final_value'].values[0]) 


scores = pd.DataFrame()
scores['name'] = res_names
scores['rmse'] = pd.DataFrame(res_rmse).round(2)
scores['mape'] = pd.DataFrame(res_mape).round(2)
scores['value'] = pd.DataFrame(res_final).round(2)

return_scores = pd.DataFrame()

return_scores['rmse'] = scores.set_index('name')['rmse']
return_scores['mape'] = scores.set_index('name')['mape']
return_scores['value'] = scores.set_index('name')['value']

print(return_scores)

#

y = return_scores['value']
x = return_scores['mape']

#plt.scatter(x, y)


# In[ ]:





# In[ ]:


#show performance

og_returns = []
fb_returns = []

#useful to sort by mape or fb or org

for i in range(0,len(stocklist)):
    #print(res[i])
    res_temp = res[i]['backtest'][1]
    #print(res[i][1]['orders'][res[i][1]['orders']['strat_id']==choice])
    print(stocklist[i])
    print('rmse')
    print(res[i]['score1'])
    print('mape')
    print(res[i]['score2'])
    
    orders = res_temp['orders'][res_temp['orders']['strat_id']==choice]
    #print(df.to_markdown(orders))
    
    #values = res_data['periodic']['portfolio_value']
    values = res_temp['periodic'][res_temp['periodic']['strat_id']==choice]['portfolio_value']
    #print(len(values))
    ret_value = values.pct_change()[1:]
    cumulative_ret_value = (ret_value + 1).cumprod()
    
    values = res[i]['backtest'][1]['indicators']['CustomIndicator']
    ret_value = values.pct_change()[1:]
    cumulative_for_value = (ret_value + 1).cumprod()
    
    values = res[i]['forecast']['yhat'][HorizonPeriod:]
    ret_value = values.pct_change()[1:]
    cumulative_for_value = pd.DataFrame((ret_value + 1).cumprod())
    cumulative_for_value.reset_index(drop=True, inplace=True)
    plt.plot(cumulative_for_value,label="forecast")
    
    #cumulative_ret_value.reset_index()
    
    subset = stocks_data[stocks_data["Symbol"]==stocklist[i]][HorizonPeriod:]
    
    #print(len(subset))
    
    price_data = subset["Adj Close"]
    #print(price_data)
    
    og_ret_data = price_data.pct_change()[1:]
    cumulative_og_ret_data = (og_ret_data + 1).cumprod()
    
    #cumulative_og_ret_data.reset_index()
    cumulative_og_ret_data.reset_index(drop=True, inplace=True)
    #print(cumulative_og_ret_data.index)
    
    plt.plot(cumulative_ret_value,label="strategy")
    plt.plot(cumulative_og_ret_data,label="hold")
    plt.legend(loc="upper left",fontsize=8)
    #plt.plot(cumulative_for_value)
    
    og_returns.append(cumulative_ret_value[-1:].values[0])
    fb_returns.append(cumulative_og_ret_data[-1:].values[0])
    
    plt.show()
    display(orders[["dt","type","price","size","value","commission","pnl"]].round(2))
    
return_scores['og'] = og_returns
return_scores['fb'] = fb_returns
return_scores['name'] = stocklist
return_scores = return_scores.set_index('name')

#plt.hist(y, bins='auto')  # arguments are passed to np.histogram
plt.hist(return_scores['og'], bins='auto')  # arguments are passed to np.histogram
plt.hist(return_scores['fb'], bins='auto')  # arguments are passed to np.histogram


# In[ ]:





# In[ ]:


print(np.mean(return_scores['og']))
print(np.mean(return_scores['fb']))

stocklistsort = return_scores.sort_values(by=['mape'], ascending=True).index
stocklistsort.values
#= list(stocklistsort.values)

mylist = stocklistsort.values

for i in range(0,len(stocklistsort.values)):
    print(stocklistsort[i])
    
    item = stocklistsort[i]

    #search for the item
    position = stocklist.index(item)

    #print(res[i])
    res_temp = res[position]['backtest'][1]
    #print(res[i][1]['orders'][res[i][1]['orders']['strat_id']==choice])
    
    print('rmse')
    print(res[position]['score1'])
    print('mape')
    print(res[position]['score2'])
    
    orders = res_temp['orders'][res_temp['orders']['strat_id']==choice]
    #print(df.to_markdown(orders))
    
    #values = res_data['periodic']['portfolio_value']
    values = res_temp['periodic'][res_temp['periodic']['strat_id']==choice]['portfolio_value']
    #print(len(values))
    ret_value = values.pct_change()[1:]
    cumulative_ret_value = (ret_value + 1).cumprod()
    
    values = res[position]['backtest'][1]['indicators']['CustomIndicator']
    ret_value = values.pct_change()[1:]
    cumulative_for_value = (ret_value + 1).cumprod()
    
    values = res[position]['forecast']['yhat'][HorizonPeriod:]
    ret_value = values.pct_change()[1:]
    cumulative_for_value = pd.DataFrame((ret_value + 1).cumprod())
    cumulative_for_value.reset_index(drop=True, inplace=True)
    plt.plot(cumulative_for_value,label="forecast")
    
    #cumulative_ret_value.reset_index()
    
    subset = stocks_data[stocks_data["Symbol"]==stocklistsort[position]][HorizonPeriod:]
    
    #print(len(subset))
    
    price_data = subset["Adj Close"]
    #print(price_data)
    
    og_ret_data = price_data.pct_change()[1:]
    cumulative_og_ret_data = (og_ret_data + 1).cumprod()
    
    #cumulative_og_ret_data.reset_index()
    cumulative_og_ret_data.reset_index(drop=True, inplace=True)
    #print(cumulative_og_ret_data.index)
    
    plt.plot(cumulative_ret_value,label="strategy")
    plt.plot(cumulative_og_ret_data,label="hold")
    plt.legend(loc="upper left",fontsize=8)
    #plt.plot(cumulative_for_value)
    
    plt.show()
    display(orders[["dt","type","price","size","value","commission","pnl"]].round(2))


# In[ ]:





# In[ ]:





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

