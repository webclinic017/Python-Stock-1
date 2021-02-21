#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas_ta dataclasses')


# In[46]:


pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

from finta import TA
import pandas_ta as ta
from fastquant import get_crypto_data, get_stock_data, backtest
from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

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

#max # of records is 500
outter_df = get_crypto_data("BTC/USDT", "2019-01-01", "2021-02-21")

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


outter_df.to_csv('btresults.csv', index =True)

