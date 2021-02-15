#inspiration
#https://simply-python.com/2019/01/16/retrieving-stock-statistics-from-yahoo-finance-using-python/
#https://towardsdatascience.com/backtest-your-trading-strategy-with-only-3-lines-of-python-3859b4a4ab44?source=friends_link&sk=ec647b6bb43fe322013248fd1d473015
#https://simply-python.com/2019/01/16/retrieving-stock-statistics-from-yahoo-finance-using-python/

!pip install --upgrade pip
#!pip install 

import concurrent.futures
import urllib
import re
import time
import stat
import contextlib
import functools

!pip install datetime yfinance pandas pandas_market_calendars pandas_datareader mpl-finance stockstats matplotlib fastquant numpy stockstats cryptography mplfinance plotly finta fbprophet

from concurrent.futures import wait, ALL_COMPLETED
import urllib.request
import datetime
from datetime import timedelta
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import os.path
from os import path
from datetime import date
from functools import reduce
#import pandas-datareader
#import mpl-finance
import stockstats
import matplotlib as mpl
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
from fastquant import backtest, get_stock_data
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mdates
from fbprophet import Prophet
from finta import TA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
%matplotlib inline

pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

cores = int(len(os.sched_getaffinity(0))/2)

pool1 = concurrent.futures.ProcessPoolExecutor()

end = datetime.date.today()
start = end - timedelta(weeks=117)

one_week_end = start
one_week_start = one_week_end - timedelta(weeks=1)

#need to do the two pass trick (i.e. find stocks fully populated a week 9 quarters back)

nyse = mcal.get_calendar('NYSE')
official_trading_dates= nyse.schedule(start_date=start, end_date=end)
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
size=10
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

pool2 = concurrent.futures.ProcessPoolExecutor()

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

returnsdf = pd.DataFrame()
returnsl = []

half = int(len(official_trading_dates)/2)

w = 117

#lookbackperiod = half
#weeks - 52 * 252 trading days a year / 52 = # of trading days
lookbackperiod = int((w-52)*252/52)

#cumulative returns of 1st half
for i in vetted_symbols:
    subset = stocks_data[stocks_data["Symbol"]==i][1:lookbackperiod]
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

bottom10percent = returnsdf.sort_values(by=['returns'], ascending=True)[1:tenpercent]

print(top10percent)
print(bottom10percent)
#returnsdf

#cumulative returns past lookbackperiod
for i in top10percent["stock"]:
    subset = stocks_data[stocks_data["Symbol"]==i][lookbackperiod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    
    price_data = stock["adj close"]
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()

    plt.plot(cumulative_ret, label=i)
    plt.legend(loc="upper left",fontsize=8)
    
plt.show()

#cumulative returns past lookbackperiod
for i in bottom10percent["stock"]:
    subset = stocks_data[stocks_data["Symbol"]==i][lookbackperiod:]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    
    price_data = stock["adj close"]
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()

    plt.plot(cumulative_ret, label=i)
    plt.legend(loc="upper left",fontsize=8)
    
stocklist = list(top10percent["stock"])
print(stocklist)	
	
for i in stocklist:
    subset = stocks_data[stocks_data["Symbol"]==i][lookbackperiod:]
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
    m = Prophet(daily_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
    m.fit(ts)
    forecast = m.make_future_dataframe(periods=0, freq='D')
    pred = m.predict(forecast)
    df = stock
    expected_1day_return = pred.set_index("ds").yhat.pct_change().shift(-1).multiply(100)
    df["custom"] = expected_1day_return.multiply(-1)
    m.plot(pred)
    plt.title('Prophet: Forecasted Daily Closing Price', fontsize=25)

    #exponential smoothing VAMA
    a  = pd.DataFrame(TA.EVWMA(subset))

    fit3 = SimpleExpSmoothing(a, initialization_method="estimated").fit()
    fcast3 = fit3.forecast(3).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])

    #weighted moving averages
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6}) # add your own style here
    fig = mpf.figure(figsize=(10, 7), style=s)
    ax = fig.add_subplot(2,1,1)
    av = fig.add_subplot(2,1,2, sharex=ax)
    #az = fig.add_subplot(3,1,1)
    mpf.plot(subset,type='candle',mav=(3,6,9),volume=av,show_nontrading=True, ax=ax)
    
    my_dpi = 50
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), dpi=my_dpi)

    # title for entire figure
    fig.suptitle(i, fontsize=20)

    #other technical
    axes[0, 0].plot(stock["macds"], color="m", label="Signal Line")    
    #axes[0, 1].set_title('Subplot 1', fontsize=14)
    axes[0, 0].plot(stock["macd"], color="y", label="MACD")
    axes[0, 0].legend(loc="lower right",fontsize=14)
    
    axes[0, 1].plot(stock["close_10_sma"], color="b", label="SMA")
    axes[0, 1].plot(stock["close_12_ema"], color="r", label="EMA")
    axes[0, 1].plot(stock["adj close"], color="g", label="Adj Close prices")
    axes[0, 1].plot(stock['boll'], color="b", label="BBands")
    axes[0, 1].plot(stock['boll_ub'], color="b", label="BBands")
    axes[0, 1].plot(stock['boll_lb'], color="b", label="BBands")
    axes[0, 1].plot(stock["adj close"], color="g", label="Adj Close prices")
    axes[0, 1].legend(loc="lower right",fontsize=14)
    
    axes[1, 0].plot(a, color="b", label="Exp Smooth")
    axes[1, 0].plot(fit3.fittedvalues, marker='o', color='green')
    #line3, = axes[1, 0].plot(fcast3, marker='o', color='green')
    #axes[1, 0].plot(fcast3, marker='o', color='orange')
    #axes[1, 0].legend(loc="lower right", label=fcast3.name)
    print(fcast3)
    
    axes[1, 1].plot(stock['rsi_14'], color="b", label="RSI_14")
    axes[1, 1].legend(loc="lower right",fontsize=14)
    
    #print(stock)
    
    #axes[2, 1].set_xlabel('cumret', fontsize=14)
    #axes[2, 1].set_title('', fontsize=14)

    plt.show()

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

pool3 = concurrent.futures.ProcessPoolExecutor()

def back_test(i):
    subset = stocks_data[stocks_data["Symbol"]==i][lookbackperiod:]
    
    #converts date to datetime
    stock = StockDataFrame.retype(subset[["Date","Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    #print(subset)
    ts = subset[["Date","Adj Close"]]
    ts.columns = ['ds', 'y']
    #print(ts)
    m = Prophet(daily_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
    m.fit(ts)
    forecast = m.make_future_dataframe(periods=0, freq='D')
    pred = m.predict(forecast)
    df = stock
    expected_1day_return = pred.set_index("ds").yhat.pct_change().shift(-1).multiply(100)
    df["custom"] = expected_1day_return.multiply(-1)
    
    #b = backtest("multi", df.dropna())
    with contextlib.redirect_stdout(None):
        b = backtest("multi", df.dropna(), strats=strats_opt,return_history=True)

    return(b)
    #print(b)
    #print(df)

futures_back = [pool3.submit(back_test, args) for args in stocklist]
wait(futures_back, timeout=None, return_when=ALL_COMPLETED)

res = []
for x in range(0,len(stocklist)):
    res.append(futures_back[x].result())
    
res_data = pd.DataFrame()    
for i in range(0,len(res)):
    res_data = pd.concat([res[i][0],res_data])
    
res_data.to_csv(start.strftime('%Y-%m-%d')+'-'+end.strftime('%Y-%m-%d')+'-'+str(len(vetted_symbols))+'res_backtest_data.csv', index = False)
res_data

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
res_data[res_data['strat_id']==choice]

#show performance

for i in range(0,len(stocklist)):
    #print(res[i])
    res_data = res[i][1]
    #print(res[i][1]['orders'][res[i][1]['orders']['strat_id']==choice])
    print(stocklist[i])
    
    orders = res_data['orders'][res_data['orders']['strat_id']==choice]
    #print(df.to_markdown(orders))
    
    #values = res_data['periodic']['portfolio_value']
    values = res_data['periodic'][res_data['periodic']['strat_id']==choice]['portfolio_value']
    #print(len(values))
    ret_value = values.pct_change()[1:]
    cumulative_ret_value = (ret_value + 1).cumprod()
    
    #cumulative_ret_value.reset_index()
    
    subset = stocks_data[stocks_data["Symbol"]==stocklist[i]][lookbackperiod:]
    
    #print(len(subset))
    
    price_data = subset["Adj Close"]
    #print(price_data)
    
    og_ret_data = price_data.pct_change()[1:]
    cumulative_og_ret_data = (og_ret_data + 1).cumprod()
    
    #cumulative_og_ret_data.reset_index()
    cumulative_og_ret_data.reset_index(drop=True, inplace=True)
    #print(cumulative_og_ret_data.index)
    
    plt.plot(cumulative_ret_value)
    plt.plot(cumulative_og_ret_data)
    
    plt.show()
    display(orders[["dt","type","price","size","value","commission","pnl"]])

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