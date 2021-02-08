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

pool = concurrent.futures.ProcessPoolExecutor()

end = datetime.date.today()
start = end - timedelta(weeks=117)

#need to do the two pass trick (i.e. find stocks fully populated a week 9 quarters back)

nyse = mcal.get_calendar('NYSE')
trading_dates= nyse.schedule(start_date=start, end_date=end)
idx2 = trading_dates.index

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
    print("downloading")
    urllib.request.urlretrieve(url, 'nasdaqtraded.txt')
    
df = pd.read_csv('nasdaqtraded.txt', sep='|')[0:-1]

#process symbols for bad characters
BAD_CHARS = ['$','.']
pat = '|'.join(['({})'.format(re.escape(c)) for c in BAD_CHARS])

df = df[~df['Symbol'].str.contains(pat)]

#choose size
size=20
stocks = list(df["Symbol"].sample(n=size))

def dl(stock):
    return yf.download(stock, start=start, end=end).iloc[:, :6].dropna(axis=0, how='any')

futures = [pool.submit(dl, args) for args in stocks]
wait(futures, timeout=None, return_when=ALL_COMPLETED)

stocks_data = pd.DataFrame()

for x in range(0,len(stocks)):
    prices = pd.DataFrame(futures[x].result())
    prices['Symbol'] = stocks[x]
    prices = prices.loc[~prices.index.duplicated(keep='last')]        
    prices = prices.reset_index()
            
    idx1 = prices.index  
        
    merged = idx1.union(idx2)
    s = prices.reindex(merged)
    df = s.interpolate().dropna(axis=0, how='any')
        
    stocks_data = pd.concat([stocks_data,df])

stocks_data.to_csv(start.strftime('%Y-%m-%d')+'-'+end.strftime('%Y-%m-%d')+'-'+str(size)+'stocks_data.csv', index = False)

def stocks_table_function(**kwargs):

    print('3 Creating aggregated dataframe with stock stats for last available date + write to CSV file...')

    #ti = kwargs['ti']

    #stocks_prices = ti.xcom_pull(task_ids='fetch_prices_task') # <-- xcom_pull is used to pull the stocks_prices list generated above
    stocks_prices = stocks_data # <-- xcom_pull is used to pull the stocks_prices list generated above

    stocks_adj_close = []

    for i in range(0, len(stocks)):

        #adj_price= stocks_prices[i][['Date','Adj Close']]

        temp =  stocks_prices['Symbol']==stocks[i]
        a=stocks_prices[temp]

        adj_price = a[["Date","Adj Close"]]
        #print(adj_price)

        adj_price.set_index('Date', inplace = True)

        adj_price.columns = [stocks[i]]

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



    for i in range(0, len(stocks)):

        values =(get_key_stats('https://sg.finance.yahoo.com/quote/'+ str(stocks[i]) +'/key-statistics?p='+ str(stocks[i])))

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


stocks_table_function()

#clear plot
plt.show()
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()
plot_data = []
#subset["Adj Close"]

#charts
from stockstats import StockDataFrame

for i in stocks:
    subset = stocks_data[stocks_data["Symbol"]==i]
    stock = StockDataFrame.retype(subset[["Date","Open", "Close", "Adj Close", "High", "Low", "Volume"]])

    plt.plot(stock["macd"], color="y", label="MACD")
    plt.plot(stock["macds"], color="m", label="Signal Line")
    plt.plot(stock["close_10_sma"], color="b", label="SMA")
    plt.plot(stock["close_12_ema"], color="r", label="EMA")
    plt.plot(stock["adj close"], color="g", label="Close prices")
    plt.legend(loc="lower right")
    plt.show()
    #print(stock)

#backtest
#!pip install fastquant
from stockstats import StockDataFrame
from fastquant import backtest, get_stock_data

pool2 = concurrent.futures.ProcessPoolExecutor()

# Utilize single set of parameters
strats = { 
    "smac": {"fast_period": 35, "slow_period": 50}, 
    "rsi": {"rsi_lower": 30, "rsi_upper": 70} 
} 

strats_opt = { 
    "smac": {"fast_period": 35, "slow_period": [40, 50]}, 
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
    return(backtest("multi", stock, strats=strats_opt))
    
futures_back = [pool2.submit(back_test, args) for args in stocks]
wait(futures, timeout=None, return_when=ALL_COMPLETED)

res_data = pd.DataFrame()

for x in range(0,len(stocks)):
    res_opt = pd.DataFram(futures_back[x].result())
    res_data = pd.concat([res_opt,res_data])
    
res_data
    
