#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras numpy sklearn auto_ml tensorflow')


# In[2]:


#plt.plot(pred['yhat_lower'])
#plt.plot(pred['yhat'])
#plt.plot(pred['yhat_upper'])


# In[3]:


#from fastquant import get_stock_data
#df = get_stock_data("JFC", "2018-01-01", "2019-05-31")
#print(df.head())
#len(df)
#outter_df


# In[4]:


from finta import TA
import pandas_ta as ta
from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from datetime import date
import yfinance as yf

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import tabulate
import keras

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely


end = datetime.date.today()
start = end - timedelta(weeks=117)

df = yf.download("BTC-USD", start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), auto_adjust=True).iloc[:, :6].dropna(axis=0, how='any')

oneyroneqtr=456
#oneyroneqtr=315
length = len(df[0:oneyroneqtr])

def MAPE(Y_actual,Y_Predicted):
    mape_ = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape_


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

sdev_constant = 2.33

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

df.columns = ["open", "high", "low", "close", "volume"]
df.index.names = ['dt']

#EVWMA
Short_EVWMA = pd.DataFrame(TA.EVWMA(df,12))
Long_EVWMA = pd.DataFrame(TA.EVWMA(df,26))
Short_EVWMA.columns = ['EVWMA_12']
Long_EVWMA.columns = ['EVWMA_26']

#p 209 of ttr doc
MACD_EVWMA = pd.DataFrame(Short_EVWMA['EVWMA_12'] - Long_EVWMA['EVWMA_26'])
MACD_EVWMA.columns = ['MACD-line']

Signal_EVWMA = pd.DataFrame(ta.ema(MACD_EVWMA["MACD-line"], length=9))
Signal_EVWMA.columns = ['Signal_EMA_9_MACD']
df['date'] = df.index
df['Signal_EVWMA'] = Signal_EVWMA
df['RSI'] = computeRSI(df['close']*df['volume'], 14)
df[['bbands_upper', 'bbands_lower']] = bbands(df)
df['ATR'] = atrVolume(df)
df['VWP'] = df['close']*df['volume']
df['bbands_upper_dv_VWP'] = df['bbands_upper']/df['VWP']
df['bbands_lower_dv_VWP'] = df['bbands_lower']/df['VWP']
df['bbands_mid'] = (df['bbands_upper']+df['bbands_lower'])/2
df['bbands_mid_dv_VWP'] = df['bbands_mid']/df['VWP']
df['Signal_EVWMA_x_VWP'] = df['Signal_EVWMA']*df['VWP']
df['next_day_close'] = df['close'].shift(-1)


# In[ ]:





# In[5]:


custompre = pd.DataFrame()

for x in range(0,len(df)):
    frames = df.index[x:(x+oneyroneqtr)]
    length = len(frames)
    if length == oneyroneqtr:
        #print(outter_df.loc[frames])
        
        df_inner = df.loc[frames]
        
        ts = df_inner.reset_index()[["dt", "VWP"]]
        ts.columns = ['ds', 'y']

        #strips last day from model
        m = Prophet(daily_seasonality=True,yearly_seasonality=True)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
        m.fit(ts[0:-1])

        #forecast only last day in model (can verify result)
        forecast = pd.DataFrame(df_inner.index)[-1:]
        forecast.columns = ['ds']
        
        #Predict and plot
        pred = m.predict(forecast)
                
        X = df_inner[['date', 'open', 'high', 'low', 'close', 'volume', 'Signal_EVWMA',
       'RSI', 'bbands_upper', 'bbands_lower', 'ATR', 'VWP',
       'bbands_upper_dv_VWP', 'bbands_lower_dv_VWP', 'bbands_mid',
       'bbands_mid_dv_VWP', 'Signal_EVWMA_x_VWP']]

        y = df_inner['next_day_close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        df_train = X_train
        df_train['next_day_close'] = y[X.index]
        df_test = X_test
        df_test['next_day_close'] = y_test[y_test.index]

        column_descriptions = {
          'next_day_close': 'output',
            'date': 'date'
        }
                
        ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
        #, feature_learning=True, fl_data=df_train))

        #ml_predictor.train(df_train,model_names=['DeepLearningRegressor'])
        #ml_predictor.train(df_train,model_names=['XGBRegressor'])
        ml_predictor.train(df_train)

        # Score the model on test data
        test_score = ml_predictor.score(df_test, df_test.next_day_close)

        # auto_ml is specifically tuned for running in production
        # It can get predictions on an individual row (passed in as a dictionary)
        # A single prediction like this takes ~1 millisecond
        # Here we will demonstrate saving the trained model, and loading it again
        file_name = ml_predictor.save()

        trained_model = load_ml_model(file_name)

        # .predict and .predict_proba take in either:
        # A pandas DataFrame
        # A list of dictionaries
        # A single dictionary (optimized for speed in production evironments)
        
        #predictions = trained_model.predict(df_test)
        #display(pd.DataFrame(predictions))
        
        #apply model to sliding window
        predictions2 = pd.DataFrame(trained_model.predict(X))
        predictions2 = predictions2.set_index(df_inner['date'])
        predictions2.columns = ['custom']
        
        #merge=pd.merge(predictions2,df, how='inner', left_index=True, right_index=True)
        
        temp = pd.DataFrame()
        #need date -2 back because we're predicting next day's return. Else I have to do shifting later
        temp['date'] = [frames[-2].strftime('%Y-%m-%d')]
        temp['yhat'] = predictions2['custom'][-1:].values[0]
        
        custompre = custompre.append(temp, ignore_index=True)
        
    else:
        break


# In[6]:



merge=pd.merge(custompre.set_index('date').dropna(),df, how='inner', left_index=True, right_index=True)
plt.scatter(merge['yhat'],merge['next_day_close'])


# In[7]:



merge.index.names = ['dt']
expected_1day_return = ((merge['yhat']-merge['close'])/merge['close']).multiply(100)

merge["custom"] = expected_1day_return


# In[9]:


merge


# In[25]:


#df = pd.read_csv('btresults.csv')
#df = df.set_index('dt')

#inverse
#df['custom'] = df['custom']*1

#mergeCompare = pd.merge(merge['close'].shift(+1),merge['close'], how='inner', left_index=True, right_index=True)

actualReturns = ((merge['next_day_close']-merge['close'])/merge['close'])

orderbook = pd.DataFrame()

funds = 1000
BuyFundsPercent = .25
percentSoldOfHeld = .75

held = 0
upper = 0
lower = 0

for i in merge.index:
    #print(df.loc[i]['custom'])
    temp = pd.DataFrame()

    estRet = merge.loc[i]['custom']
    #rsi = merge.loc[i]['RSI']
    #rsiDelta = (df['RSI']-df['RSI'].shift(+1))[i]
    
    if (estRet > upper):
    #if (estRet > upper and rsi > 20 and rsiDelta > 0)
    #if (estRet > upper and rsi > 20 and rsiDelta > 0 and (df.loc[i]['close']*df.loc[i]['volume'] < df.loc[i]['bbands_upper'])):
        temp['order'] = ['buy']
        
        ProportionOfFunds = funds * BuyFundsPercent
        Qty = ProportionOfFunds / merge.loc[i]['close']
        value = merge.loc[i]['close']*Qty
        
        funds = funds - value
        held = held + Qty
                
    #what about over 80 RSI?  Guess it's a hold until it drops?
    elif (estRet < lower):
    #elif (estRet < lower and rsi < 80 and rsiDelta < 0)
    #elif (estRet < lower and rsi < 80 and rsiDelta < 0 and (df.loc[i]['close']*df.loc[i]['volume'] > df.loc[i]['bbands_lower'])):
        temp['order'] = ['sell']

        Qty = held*percentSoldOfHeld
        value = merge.loc[i]['close']*Qty
        
        funds = funds + value
        held = held - Qty
            
    #if ((estRet < upper) & (etsRet > lower))
    #if ((estRet > lower) and (estRet < upper)):
    else:
        temp['order'] = ['hold']
        
        Qty = 0
        value = merge.loc[i]['close']*Qty
        
        funds = funds + value
        held = held - Qty

    temp['date'] = [i]
    temp['estRet'] = estRet
    temp['actRet'] = actualReturns.loc[i]
    temp['price'] = merge.loc[i]['close']
    temp['PropInvValue'] = merge.loc[i]['close']*Qty
    temp['qtyShares'] = Qty
    temp['funds'] = funds
    temp['portValue'] = funds + merge.loc[i]['close']*held
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

#cov(orderbook['estRet'], orderbook['actRet'])
corrprep = pd.merge(orderbook.set_index('date')['estRet'][1:],orderbook.set_index('date')['actRet'][1:], how='inner', left_index=True, right_index=True)
corr, _ = pearsonr(corrprep.dropna()['estRet'][1:], corrprep.dropna()['actRet'][1:])
print('Pearsons correlation: %.3f' % corr)#df

print("Total Return")
print(orderbook['portValue'][-1:].values[0]/funds)

print("Hold Return")
print(merge['close'][-1:].values[0]/merge['close'][0:].values[0])

values = orderbook.set_index('date')['portValue']
#print(len(values))
ret_value = values.pct_change()[1:]
cumulative_ret_value = (ret_value + 1).cumprod()

#show cumulative charts
og_ret_data = orderbook.set_index('date')['price']
og_ret_value = og_ret_data.pct_change()[1:]
cumulative_og_ret_data = (og_ret_value + 1).cumprod()

plt.plot(cumulative_ret_value)
plt.plot(cumulative_og_ret_data)
plt.show()


plt.plot(merge.dropna()['bbands_upper'])
plt.plot((merge.dropna()['close']*merge.dropna()['volume']))
plt.plot(merge.dropna()['bbands_lower'])
plt.show()


# In[ ]:




