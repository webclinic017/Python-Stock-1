#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas_ta dataclasses')


# In[ ]:


pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

from finta import TA
import pandas_ta as ta
#from fastquant import get_crypto_data, get_stock_data, backtest
#from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import date

def wwma(values, n):
    """
     J. Welles Wilder's EMA 
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, n=50):
    data = df.copy()
    high = data['high']
    low = data['low']
    close = data['close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr

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

#max # of records is 500
import yfinance as yf

end = datetime.date.today()
start = end - timedelta(weeks=117)

#df = yf.download("BTC-USD", start="2019-01-01", end="2021-02-21", auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')
df = yf.download("^GSPC", start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')

df.columns = ["open", "high", "low", "close", "volume"]
df.index.names = ['dt']
outter_df = df

#EVWMA
Short_EVWMA = pd.DataFrame(TA.EVWMA(outter_df,12))
Long_EVWMA = pd.DataFrame(TA.EVWMA(outter_df,26))
Short_EVWMA.columns = ['EVWMA_12']
Long_EVWMA.columns = ['EVWMA_26']

ATR = pd.DataFrame(atr(outter_df))
ATR.columns = ['ATR']

#p 209 of ttr doc
MACD_EVWMA = pd.DataFrame(Short_EVWMA['EVWMA_12'] - Long_EVWMA['EVWMA_26'])
MACD_EVWMA.columns = ['MACD-line']

Signal_EVWMA = pd.DataFrame(ta.ema(MACD_EVWMA["MACD-line"], length=9))
Signal_EVWMA.columns = ['Signal_EMA_9_MACD']
Signal_EVWMA
outter_df['custom'] = Signal_EVWMA['Signal_EMA_9_MACD']
df['RSI'] = computeRSI(df['close']*df['volume'], 14)

outter_df.to_csv('btresults.csv', index =True)

