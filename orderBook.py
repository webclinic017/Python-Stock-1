#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pd.set_option('display.max_columns', None) #replace n with the number of columns you want to see completely
pd.set_option('display.max_rows', None) #replace n with the number of rows you want to see completely

import pandas as pd
from fastquant import backtest

from matplotlib import pyplot as plt

df = pd.read_csv('btresults.csv')
df = df.set_index('dt')

#inverse
df['custom'] = df['custom']*1

mergeCompare = pd.merge(df['close'].shift(+1),df['close'], how='inner', left_index=True, right_index=True)

actualReturns = ((mergeCompare['close_y']-mergeCompare['close_x'])/mergeCompare['close_x'])

orderbook = pd.DataFrame()

funds = 1000
BuyFundsPercent = .5
percentHeldOnSell = 1

held = 0
upper = 0
lower = 0

for i in df.index:
    #print(df.loc[i]['custom'])
    temp = pd.DataFrame()

    estRet = df.loc[i]['custom']
    rsi = df.loc[i]['RSI']
    rsiDelta = (df['RSI']-df['RSI'].shift(+1))[i]
    
    #if (estRet > upper):
    #if (estRet > upper and rsi > 20 and rsiDelta > 0)
    if (estRet > upper and rsi > 20 and rsiDelta > 0 and (df.loc[i]['close']*df.loc[i]['volume'] < df.loc[i]['bbands_upper'])):
        temp['order'] = ['buy']
        
        ProportionOfFunds = funds * BuyFundsPercent
        Qty = ProportionOfFunds / df.loc[i]['close']
        value = df.loc[i]['close']*Qty
        
        funds = funds - value
        held = held + Qty
                
    #what about over 80 RSI?  Guess it's a hold until it drops?
    #elif (estRet < lower):
    #elif (estRet < lower and rsi < 80 and rsiDelta < 0)
    elif (estRet < lower and rsi < 80 and rsiDelta < 0 and (df.loc[i]['close']*df.loc[i]['volume'] > df.loc[i]['bbands_lower'])):
        temp['order'] = ['sell']

        Qty = held*percentHeldOnSell
        value = df.loc[i]['close']*Qty
        
        funds = funds + value
        held = held - Qty
            
    #if ((estRet < upper) & (etsRet > lower))
    #if ((estRet > lower) and (estRet < upper)):
    else:
        temp['order'] = ['hold']
        
        Qty = 0
        value = df.loc[i]['close']*Qty
        
        funds = funds + value
        held = held - Qty

    temp['date'] = [i]
    temp['estRet'] = estRet
    temp['actRet'] = actualReturns.loc[i]
    temp['price'] = df.loc[i]['close']
    temp['PropInvValue'] = df.loc[i]['close']*Qty
    temp['qtyShares'] = Qty
    temp['funds'] = funds
    temp['portValue'] = funds + df.loc[i]['close']*held
    temp['held'] = held
    
    temp = temp.round(2)

    orderbook = orderbook.append(temp)

display(orderbook.dropna().head())
display(orderbook.dropna().tail())

import matplotlib.pyplot as plt
plt.scatter(orderbook['estRet'], orderbook['actRet'],label="x: estRet; y:actRet")
plt.xlabel('estRet')
plt.ylabel('actRet')

from scipy.stats import pearsonr

#cov(orderbook['estRet'], orderbook['actRet'])
corrprep = pd.merge(orderbook.set_index('date')['estRet'][1:],orderbook.set_index('date')['actRet'][1:], how='inner', left_index=True, right_index=True)
corr, _ = pearsonr(corrprep.dropna()['estRet'][1:], corrprep.dropna()['actRet'][1:])
print('Pearsons correlation: %.3f' % corr)#df

print("Total Return")
print(orderbook['portValue'][-1:].values[0]/1000)

print("Hold Return")
print(df['close'][-1:].values[0]/df['close'][0:].values[0])

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
plt.plot(cumulative_og_ret_data)

plt.show()

plt.plot(df.dropna()['bbands_upper'])
plt.plot((df.dropna()['close']*df.dropna()['volume']))
plt.plot(df.dropna()['bbands_lower'])

