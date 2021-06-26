#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().system('pip install keras numpy sklearn auto_ml tensorflow fredapi')


# In[26]:


#plt.plot(pred['yhat_lower'])
#plt.plot(pred['yhat'])
#plt.plot(pred['yhat_upper'])


# In[27]:


#from fastquant import get_stock_data
#df = get_stock_data("JFC", "2018-01-01", "2019-05-31")
#print(df.head())
#len(df)
#outter_df


# In[28]:


etf_commodities = ['DBO','CORN', 'WEAT', 'SOYB', 'JO', 'SGG', 'BAL', 'COW', 'MOO', 'TAGS', 'KOL' ]
#Gold, Silver, Platinum, Copper, Paladium, Aluminum, Iron, Steel
etf_metals = ['IAU', 'SLV', 'PGM', 'JJC', 'PALL', 'JJU', 'IFUNX', 'SLX']
#US dollar, European Euro, Japanese yen, Pound sterling, Australian dollar, Canadian dollar, Swiss franc, Chinese Yuan Renminbi, Swedish Krona, Peso, India
#defunct: Russia: XRU, Mexico: FXM
etf_foreign_exchanges = ['UUP','FXE','FXY','FXB','FXA','FXC','FXF','CYB', 'FXS', 'INR']
#residential, Ishares all NAmerica
etf_real_estate = ['REZ', 'IYR']
#Russia, Germany, UK, Japan, China, Euro, Euro, Brazil, Latin America, Mexico, India
etf_economies = ['ERUS','EWG','EWU','EWJ','MCHI','EZU','IEUR','EWZ','ILF','EWW','INDA']
#Ishares Investment Grade, IShares core aggregate Investment grade, Short, Total, 1-5 Years, 5-10 Years, 10 Years, Gov/Credit
#defunct:  

etf_spdr_indexes = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']
etf_dow_components = ['MMM','AXP','AMGN','AAPL','BA','CAT','CVX','CSCO','KO','DOW','GS','HD','HON','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PG','CRM','TRV','UNH','VZ','V','WMT','WBA','DIS']

etf_bonds = ['LQD', 'AGG', 'NEAR', 'IUSB', 'ISTB', 'IMTB', 'ILTB', 'GBF']
etf_muni_bonds = ['MUB', 'SUB', 'MEAR']

etf_treasuries = ['AGZ', 'GOVT', 'BIL', 'SHV', 'SHY', 'IEI', 'IEF', 'TLT']

crypto = ['BTCUSD=X','ETH','RPL','BCH','EOS','LTC']

etf_and_Crypto_list = [[etf_commodities, etf_metals, etf_foreign_exchanges, etf_real_estate, etf_economies, etf_bonds, etf_muni_bonds, etf_treasuries, crypto, etf_spdr_indexes, etf_dow_components]]
#,'GOLDAMGBD228NLBM',
FRED_Indicators = ['CPALTT01USQ657N','PAYEMS','IRLTLT01USM156N','MABMM301USM189S','LFWA64TTUSM647S','MANMM101USA189S','MICH','UMCSENT','CSCICP03USM665S','DGS10','DTB3','DGS3MO','CASTHPI','GDPC1','CIVPART','POPTOTUSA647NWDB','MEHOINUSA672N','HOSMEDUSM052N','MORTGAGE30US','TTLHH','CSUSHPINSA','EMRATIO','CPIAUCSL','PSAVERT','LRUN64TTUSQ156S','USSTHPI','NYSTHPI','M2V','GFDEBTN','DFII10','GFDEGDQ188S','CUSR0000SEHA','ETOTALUSQ176N','ERENTUSQ176N','RECPROUSM156N','T5YIFR','BAMLHYH0A0HYM2TRIV','BAMLCC0A1AAATRIV','GVZCLS','DGS1','BAMLCC0A4BBBTRIV','VXVCLS','IC4WSA','WILLMICROCAPPR','WILLLRGCAPVAL','CFNAIDIFF','MZMSL','KCFSI','T5YIE','TOTALSA','DCOILWTICO','USSLIND','AWHAETP','CES0500000003','TCU','WTB3MS','WGS3MO','TWEXB','DEXCHUS','DEXUSUK','CILACBQ158SBOG','CES4348400001','FEDFUNDS','TDSP','PERMIT','GFDEGDQ188S','CP','PRFI','DRSFRMACBS','DRCCLACBS','DRBLACBS','DALLCIACBEP','USROA','USROE','RSAHORUSQ156S','MEFAINUSA672N','COMREPUSQ159N','HDTGPDUSQ163N','POP','NROU','FGCCSAQ027S','TEDRATE', 'VIXCLS', 'NFCI','INDPRO','LES1252881600Q','CUUR0000SEHA','LEU0252918500Q','BAA10Y','BAMLC0A0CM','BAMLH0A3HYC','BOGMBASE','DCOILBRENTEU','DCOILWTICO','DFF','DGS1MO','DGS30','DGS5','FPCPITOTLZGUSA','ICSA','INTDSRUSM193N','M1','M1V','MPRIME','PPIACO','SPCS20RSA','STLFSI2','T10Y2Y','T10Y3M','TB3MS','TREAST','UNRATE','WPU0911']

Indexes = ['^SP500TR', 'QQQ', 'DIA', 'VTWO']


# In[29]:


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
from fredapi import Fred

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import tabulate
import keras

from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
from auto_ml.utils_models import load_ml_model

pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

fred = Fred(api_key='661c0a90e914477da5a7518293de5f8e')

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


# In[30]:


fred_data = pd.DataFrame()

for x in range(0,len(FRED_Indicators)):
    name = FRED_Indicators[x]
    values = pd.DataFrame(fred.get_series(name))
    values['Symbol'] = name
    #values['Date'] = values.index
    values = values.loc[~values.index.duplicated(keep='last')]        
    values = values.reset_index()
    values.columns = ['date','value','symbol']
    #prices = prices.loc[~prices.index.duplicated(keep='last')]        
    #
                
    fred_data = pd.concat([fred_data,values])    


# In[31]:


idx2 = df.index

fred_data_cleaned = pd.DataFrame()
fred_data_cleaned['date'] = df.index
fred_data_cleaned = fred_data_cleaned.set_index('date')

#creates pivot
for x in FRED_Indicators:
    #print(x)
                
    subset = fred_data[fred_data["symbol"]==x]
    subset = subset[(subset['date']>= start) & (subset['date']<= end)]
    
    subset = subset.set_index('date')
    subset = subset.loc[~subset.index.duplicated(keep='last')]        
    #subset = subset.reset_index()
    
    idx1 = subset.index
    
    #merged = pd.merge(df,subset, how='inner', left_index=True, right_index=True)
    
    merged = idx1.union(idx2)
    
    set = subset.reindex(merged)
    
    df2 = set['value'].interpolate().dropna(axis=0, how='any')
    
    #if len(df2) == len(subset.index):
        #print("true")
        #fred_data_cleaned = pd.concat([fred_data_cleaned,df2])
    fred_data_cleaned[x] = pd.DataFrame(df2)
          


# In[32]:


df = pd.merge(fred_data_cleaned,df, how='inner', left_index=True, right_index=True)


# In[41]:





# In[39]:





# In[42]:


setlist = list(df[df.columns.drop('next_day_close')].columns)

custompre = pd.DataFrame()

for x in range(0,len(df)):
    frames = df.index[x:(x+oneyroneqtr)]
    length = len(frames)
    if length == oneyroneqtr:
        #print(outter_df.loc[frames])
        
        #this needs to be -2
        df_inner = df.loc[frames][0:-2]
        
        #ts = df_inner.reset_index()[["dt", "VWP"]]
        ts = df_inner[["date", "VWP"]]
        ts.columns = ['ds', 'y']

        #no need to strip last day from model becuase df_inner (which ts is based on) is already -2
        m = Prophet(daily_seasonality=True,yearly_seasonality=True)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
        #m.fit(ts[0:-1])
        m.fit(ts)

        #this needs to be -1
        #forecast only last day in model (can verify result) needs to be based on df.loc (similar to df_inner)
        forecast = pd.DataFrame(df.loc[frames].index)[-1:]
        forecast.columns = ['ds']
        
        #Predict and plot
        pred = m.predict(forecast)
                
        
        X = pd.DataFrame(df_inner, columns=setlist)
        #X = df_inner[['date', 'open', 'high', 'low', 'close', 'volume', 'Signal_EVWMA','RSI', 'bbands_upper', 'bbands_lower', 'ATR', 'VWP','bbands_upper_dv_VWP', 'bbands_lower_dv_VWP', 'bbands_mid','bbands_mid_dv_VWP', 'Signal_EVWMA_x_VWP']]

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
        temp['date'] = [frames[-1].strftime('%Y-%m-%d')]
        temp['yhat'] = predictions2['custom'][-1:].values[0]
        
        custompre = custompre.append(temp, ignore_index=True)
        
    else:
        break


# In[43]:



merge=pd.merge(custompre.set_index('date').dropna(),df, how='inner', left_index=True, right_index=True)
plt.scatter(merge['yhat'],merge['next_day_close'])


# In[47]:


merge


# In[44]:



merge.index.names = ['dt']
expected_1day_return = ((merge['yhat']-merge['close'])/merge['close']).multiply(100)

merge["custom"] = expected_1day_return


# In[ ]:





# In[ ]:


#propensities = pd.DataFrame()
#propensities = corrprep
#propensities['actRet'] = propensities['actRet']>0
#propensities['actRet'].replace(>0,1)
#propensities['actRet'].replace(<0,-1)
#propensities[propensities['actRet'] > 0] = 1
#propensities[propensities['actRet'] < 0] = -1
#corrprep
#propensities['actRet']

#propensities.sort_values(by=['estRet'], ascending=False)


# In[56]:


#df = pd.read_csv('btresults.csv')
#df = df.set_index('dt')

#inverse
#df['custom'] = df['custom']*1

#mergeCompare = pd.merge(merge['close'].shift(+1),merge['close'], how='inner', left_index=True, right_index=True)

actualReturns = ((merge['next_day_close']-merge['close'])/merge['close'])

orderbook = pd.DataFrame()

funds = 1000
#this is how you leverage (you borrow money at a lower rate than you expect on your return)
BuyFundsPercent = 1
percentSoldOfHeld = 1

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
    elif (estRet <= lower):
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


# In[59]:



#residuals = merge['yhat']-merge['next_day_close']
residuals = orderbook['estRet']-orderbook['actRet']


_ = plt.hist(residuals, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()


# In[57]:


plt.plot(cumulative_ret_value)


# In[ ]:


print(custompre[-1:])
print("Expected return",merge['custom'][-1:].values[0])

