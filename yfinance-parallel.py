from concurrent.futures import wait, ALL_COMPLETED

import urllib.request
import concurrent.futures
import datetime
from datetime import timedelta
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import re
import os.path
from os import path
import time
import stat
from datetime import date
from functools import reduce
#import pandas-datareader
#import mpl-finance
import stockstats
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextlib
from stockstats import StockDataFrame
from fastquant import backtest, get_stock_data
import numpy as np

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
    return yf.download(stock, start=one_week_start, end=one_week_end).iloc[:, :6].dropna(axis=0, how='any')

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


def stocks_table_function(**kwargs):

    print('3 Creating aggregated dataframe with stock stats for last available date + write to CSV file...')

    #ti = kwargs['ti']

    #stocks_prices = ti.xcom_pull(task_ids='fetch_prices_task') # <-- xcom_pull is used to pull the stocks_prices list generated above
    stocks_prices = stocks_data # <-- xcom_pull is used to pull the stocks_prices list generated above

    stocks_adj_close = []

    for i in range(0, len(vetted_symbols)):

        #adj_price= stocks_prices[i][['Date','Adj Close']]

        temp =  stocks_prices['Symbol']==vetted_symbols[i]
        a=stocks_prices[temp]

        adj_price = a[["Date","Adj Close"]]
        #print(adj_price)

        adj_price.set_index('Date', inplace = True)

        adj_price.columns = [vetted_symbols[i]]

        stocks_adj_close.append(adj_price)


    stocks_adj_close = reduce(lambda left,right: pd.merge(left, right, left_index = True, right_index = True ,how='outer'), stocks_adj_close)

    stocks_adj_close.sort_index(ascending = False, inplace = True)

    stocks_adj_close.index = pd.to_datetime(stocks_adj_close.index).date



    stocks_adj_close_f = stocks_adj_close.iloc[0] # <- creates a copy of the full df including last row only

    stocks_adj_close_f = stocks_adj_close_f.reset_index() # <- removing the index transforms the pd.Series into pd.DataFrame

    stocks_adj_close_f.insert(loc = 1, column = 'Date', value = stocks_adj_close_f.columns[1])

    stocks_adj_close_f.columns = ['Symbol', 'Date' , 'Adj. Price']

    stocks_adj_close_f.set_index('Symbol', inplace = True)




    #######################################



    def get_key_stats(tgt_website):

        df_list = pd.read_html(tgt_website)

        result_df = df_list[0]

        for df in df_list[1:]:

             result_df = result_df.append(df)

        return result_df.set_index(0).T



    stats = pd.DataFrame()

    statistics = []



    for i in range(0, len(vetted_symbols)):

        values =(get_key_stats('https://sg.finance.yahoo.com/quote/'+ str(vetted_symbols[i]) +'/key-statistics?p='+ str(vetted_symbols[i])))

        statistics.append(values)



    stats = stats.append(statistics)

    stats.reset_index(drop=True, inplace= True)

    stats.insert(loc=0, column='Symbol', value=pd.Series(stocks)) 

    stats.set_index(['Symbol'], inplace = True)



    stats = stats[['Market cap (intra-day) 5', 'Enterprise value 3', 'Trailing P/E',

                    'Forward P/E 1', 'PEG Ratio (5 yr expected) 1', 'Price/sales (ttm)',

                    'Price/book (mrq)', 'Enterprise value/revenue 3',

                    'Enterprise value/EBITDA 6', 'Beta (5Y monthly)', '52-week change 3',

                    'S&P500 52-week change 3', '52-week high 3', '52-week low 3',

                    '50-day moving average 3', '200-day moving average 3',

                    'Avg vol (3-month) 3', 'Avg vol (10-day) 3', 'Shares outstanding 5',

                    'Float', '% held by insiders 1', '% held by institutions 1',

                    'Forward annual dividend rate 4', 'Forward annual dividend yield 4',

                    'Trailing annual dividend rate 3', 'Trailing annual dividend yield 3',

                    '5-year average dividend yield 4', 'Payout ratio 4', 'Dividend date 3',

                    'Ex-dividend date 4', 'Last split factor 2', 'Last split date 3',

                    'Fiscal year ends', 'Most-recent quarter (mrq)', 'Profit margin',

                    'Operating margin (ttm)', 'Return on assets (ttm)',

                    'Return on equity (ttm)', 'Revenue (ttm)', 'Revenue per share (ttm)',

                    'Quarterly revenue growth (yoy)', 'Gross profit (ttm)', 'EBITDA',

                    'Net income avi to common (ttm)', 'Diluted EPS (ttm)',

                    'Quarterly earnings growth (yoy)', 'Total cash (mrq)',

                    'Total cash per share (mrq)', 'Total debt (mrq)',

                    'Total debt/equity (mrq)', 'Current ratio (mrq)',

                    'Book value per share (mrq)', 'Operating cash flow (ttm)',

                    'Levered free cash flow (ttm)']]



    stats.columns = ['Mkt_Cap', 'Enterpr_Value', 'P/E',

                      'P/E_(Forward)', 'PEG_Ratio', 'P/S',

                      'P/B', 'Enterpr_Value/Revenue','Enterpr_Value/EBITDA', 

                      'Beta_(5Y M)', '52W Change','S&P500_52W_Change', '52W_High', '52W_Low',

                      '50D Mov_Avg', '200D Mov_Avg.',

                      'Avg_Vol_(3M)', 'Avg_Vol_(10D)', 'Outst_Shares',

                      'Float', 'Insiders_Stake_pct', 'Institutions_Stake_pct',

                      'Dividend Rate_(F Annual)', 'Dividend Yield_(F Annual)',

                      'Dividend Rate_(T Annual)', 'Dividend_Yield_(T Annual)',

                      'Dividend Yield_(Avg 5Y)', 'Payout_Ratio', 'Dividend_Date',

                      'Ex-dividend_Date', 'Last_Split_Factor', 'Last_Split_Date',

                      'Fiscal_Year_Ends', 'MRQ', 'ProfMargin',

                      'Operating_Margin', 'ROA', 'ROE', 'Revenue12M', 'Revenue_p/s_(12M)',

                      'Quarterly_Revenue_Growth_(YoY)', 'Gross_Profit(L12M)', 

                      'EBITDA','Net Income 12M', 'Diluted EPS_(L12M)',

                      'Quarterly_Earnings_Growth_(YoY)', 'Total_Cash_(MRQ)',

                      'Total_Cash_Per_Share_(MRQ)', 'Total_Debt_(MRQ)',

                      'Total_Debt/Equity_(MRQ)', 'Current Ratio',

                      'Book Value p/s_(MRQ)', 'Ops CF 12M',

                      'Levered_Free_Cash_Flow_(L12M)']



    stats_c = stats.copy()

    stats_c = stats_c[['Mkt_Cap','P/B' ,'P/E', 'PEG_Ratio', 'ROA', 'ROE', 

                           'Revenue12M', 'Net Income 12M', 'ProfMargin', 'Ops CF 12M', 'Current Ratio']]



    stocks_table_data = pd.merge(stocks_adj_close_f, stats_c, left_index=True, right_index=True)

    

    print('DF Shape: ', stocks_table_data.shape)


    stocks_table_data.to_csv('~stocks_table_data.csv', index = False)

    return(stocks_table_data) 
    #print('Completed')


tbl = stocks_table_function()

#plt.show()
#fig = plt.figure(figsize=(10, 10))
#ax = plt.subplot()
#plot_data = []
#subset["Adj Close"]

for i in vetted_symbols:
    subset = stocks_data[stocks_data["Symbol"]==i]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])
    stock.BOLL_WINDOW = 20
    stock.BOLL_STD_TIMES = 2
    
    price_data = stock["adj close"]
    
    ret_data = price_data.pct_change()[1:]
    
    cumulative_ret = (ret_data + 1).cumprod()
    #cumulative_ret = np.cumprod(1 + ret_data.values) - 1
    
    my_dpi = 50
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20), dpi=my_dpi)

    # title for entire figure
    fig.suptitle('Charts', fontsize=20)

    # edit subplots
    axes[0, 0].plot(stock["macds"], color="m", label="Signal Line")
    axes[0, 0].legend(loc="lower right")
    axes[0, 1].set_title('Subplot 1', fontsize=14)
    axes[0, 1].plot(stock["macd"], color="y", label="MACD")
    axes[0, 1].plot(stock["macds"], color="m", label="Signal Line")
    axes[0, 1].legend(loc="lower right")
    
    axes[1, 0].plot(stock["close_10_sma"], color="b", label="SMA")
    axes[1, 0].plot(stock["close_12_ema"], color="r", label="EMA")
    axes[1, 0].plot(stock["adj close"], color="g", label="Close prices")
    axes[1, 0].legend(loc="lower right")

    axes[1, 1].plot(cumulative_ret, label="Cum Ret")
    axes[1, 1].legend(loc="lower right")
    
    axes[2, 0].plot(stock['rsi_14'], color="b", label="RSI_14")
    axes[2, 0].legend(loc="lower right")
    #print(stock)
    
    axes[2, 1].plot(stock['boll'], color="b", label="BBands")
    axes[2, 1].plot(stock['boll_ub'], color="b", label="BBands")
    axes[2, 1].plot(stock['boll_lb'], color="b", label="BBands")
    axes[2, 1].plot(stock["adj close"], color="g", label="Close prices")
    axes[2, 1].legend(loc="lower right")
        
    #axes[2, 1].set_xlabel('cumret', fontsize=14)
    #axes[2, 1].set_title('', fontsize=14)

    plt.show()

#!pip install fastquant

pool3 = concurrent.futures.ProcessPoolExecutor()

# Utilize single set of parameters
strats = { 
    "smac": {"fast_period": 35, "slow_period": 50}, 
    "rsi": {"rsi_lower": 30, "rsi_upper": 70},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "sma_period": 30, "dir_period": 10},
    "bbands": {"period": 20, "devfactor": 2.0},
    "ema": {"fast_period": 10, "slow_period": 30}
} 

strats_opt = { 
    "smac": {"fast_period": 35, "slow_period": [40, 50]}, 
    "emac": {"fast_period": [9,10,12], "slow_period": [30, 40, 50]}, 
    "rsi": {"rsi_lower": [15, 30], "rsi_upper": 70} 
}         

def back_test(stock):
    #print(stock)
    subset = stocks_data[stocks_data["Symbol"]==stock]
    
    #converts date to datetime
    stock = StockDataFrame.retype(subset[["Date","Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    #subset = subset[["Date","Open", "High", "Low", "Close", "Adj Close", "Volume"]]    
    #stock.columns = ['Date','open','high','low','close','adj close', 'volume']
    #print(stock)
    with contextlib.redirect_stdout(None):
        #print(stock)
        b = backtest("multi", stock, strats=strats_opt)
        
    return(b)
    
futures_back = [pool3.submit(back_test, args) for args in vetted_symbols]
wait(futures_back, timeout=None, return_when=ALL_COMPLETED)

res_data = pd.DataFrame()

for x in range(0,len(vetted_symbols)):
    res_opt = pd.DataFrame(futures_back[x].result())
    res_data = pd.concat([res_opt,res_data])
    
vetted_symbols    
res_data
tbl
