import pandas as pd
from fastquant import backtest

df = pd.read_csv('btresults.csv')
df = df.set_index('dt')

#inverse
df['custom'] = df['custom']*-1

actualReturns = ((df['close']-df['close'].shift(+1))/df['close'])*100

orderbook = pd.DataFrame()

funds = 1000
BuyFundsPercent = .1
percentFundsOnSell = .5

held = 0
upper = .5
lower = -.5


for i in df.index:
    #print(df.loc[i]['custom'])
    temp = pd.DataFrame()

    estRet = df.loc[i]['custom']
    
    if estRet > upper:
        temp['order'] = ['buy']
        
        ProportionOfFunds = funds * BuyFundsPercent
        Qty = ProportionOfFunds / df.loc[i]['close']
        value = df.loc[i]['close']*Qty
        
        funds = funds - value
        held = held + Qty
                
    elif estRet < lower:        
        temp['order'] = ['sell']

        Qty = held*percentFundsOnSell
        value = df.loc[i]['close']*Qty
        
        funds = funds + value
        held = held - Qty
        
            
    #if ((estRet < upper) & (etsRet > lower))
    if ((estRet > lower) and (estRet < upper)):
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
    temp['held'] = held

    orderbook = orderbook.append(temp)

display(orderbook)
#print(held)
#print(funds)

import matplotlib.pyplot as plt
plt.scatter(orderbook['estRet'], orderbook['actRet'],label="x: estRet; y:actRet")
plt.xlabel('estRet')
plt.ylabel('actRet')

from scipy.stats import pearsonr

#cov(orderbook['estRet'], orderbook['actRet'])
corr, _ = pearsonr(orderbook['estRet'][1:], orderbook['actRet'][1:])
print('Pearsons correlation: %.3f' % corr)
#df

print("Total Return")
print(orderbook['funds'][-1:].values[0]/1000)

print("Hold Return")
print(df['close'][-1:].values[0]/df['close'][0:].values[0])