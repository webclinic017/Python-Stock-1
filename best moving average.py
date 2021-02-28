import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import datetime
from datetime import timedelta
from datetime import date

from scipy.stats import ttest_ind

n_forward = 7
#name = 'BTC-USD'
#name = 'GLD'
name = 'SPY'
name = 'GOOG'

w=117
end_date = datetime.date.today()
end_date1 = end_date - timedelta(weeks=w)

#- timedelta(weeks=w*2)
start_date = end_date1 - timedelta(weeks=w)

ticker = yfinance.Ticker(name)
data = ticker.history(interval="1d",start=start_date,end=end_date, auto_adjust=True)
data['Forward Close'] = data['Close'].shift(-n_forward)
data['Forward Return'] = (data['Forward Close'] - data['Close'])/data['Close']

dateindex = data.loc[start_date:end_date].index

dateindex

limit = 100
n_forward = 7

train_size = 0.5

trades = []
expectedReturns = []

sdevs = []

minExpectedReturn = 0

width1 = len(data.loc[start_date:end_date1].index)
width2 = len(data.loc[end_date1+timedelta(days=1):end_date].index)

for i in range(0,width1):
    temp = data.loc[dateindex[i]:dateindex[i+width2]]
    
    result = []
    
    #temp['Forward Close'] = temp['Close'].shift(-n_forward)
    #temp['Forward Return'] = (temp['Forward Close'] - temp['Close'])/temp['Close']
    
    for sma_length in range(20,limit):
        temp['SMA'] = temp['Close'].rolling(sma_length).mean()
        temp['input'] = [int(x) for x in temp['Close'] > temp['SMA']]
        
        df = temp.dropna()
        
        training = df.head(int(train_size * df.shape[0]))
        test = df.tail(int((1 - train_size) * df.shape[0]))
        
        tr_returns = training[training['input'] == 1]['Forward Return']
        test_returns = test[test['input'] == 1]['Forward Return']
        
        mean_forward_return_training = tr_returns.mean()
        mean_forward_return_test = test_returns.mean()
        pvalue = ttest_ind(tr_returns,test_returns,equal_var=False)[1]

        result.append({
          'sma_length':sma_length,
          'training_forward_return': mean_forward_return_training,
          'test_forward_return': mean_forward_return_test,
          'p-value':pvalue
        })
    
    result.sort(key = lambda x : -x['training_forward_return'])
    
    temp['SMA'] = temp['Close'].rolling(result[0]['sma_length']).mean()
    
    if result[0]['p-value'] > .1:
        #print(result[0]['p-value'])
        if result[0]['training_forward_return'] > minExpectedReturn:
            if result[0]['test_forward_return'] > minExpectedReturn:
                if temp.ix[-1]['Close']>temp.ix[-1]['SMA']:
                    
                    #add to list of trades
                    trades.append(temp.index[-1].strftime('%Y-%m-%d'))
                    expectedReturns.append((result[0]['training_forward_return']+result[0]['test_forward_return'])/2)
                    sdevs.append(np.std(temp['Forward Return']))

                    print(result[0])
                    print(temp[-1:]['Close'])
                    print(temp[-1:]['SMA'])
                
                    plt.plot(temp['Close'],label='BTC')

                    stringLabel = str(result[0]['sma_length']) + " SMA" + " at " + str(n_forward) + " day return " + str(result[0]['test_forward_return'].round(3))
                    #plt.plot(data['Close'].rolling(result[0]['sma_length']).mean(),label = stringLabel)
                    plt.plot(temp['Close'].rolling(result[0]['sma_length']).mean(),label = stringLabel)
                    
                    plt.legend()

                    plt.show()
                    
                    plt.hist(temp['Forward Return'], bins='auto')  # arguments are passed to np.histogram
                    plt.show()
        
        
plt.hist(sdevs, bins='auto')  # arguments are passed to np.histogram
plt.show()
plt.hist(expectedReturns, bins='auto')  # arguments are passed to np.histogram
plt.show()
    

start = 1000

set = pd.DataFrame()
for i in range(0,len(trades)):
    
    value = pd.DataFrame(data.loc[trades[i]]).transpose()
    value['ExpectedReturn'] = expectedReturns[i]
    value['sdev'] = sdevs[i]
    set = pd.concat([set,value])
    #funds = 

plt.hist(set['Forward Return'], bins='auto')  # arguments are passed to np.histogram



orderbook = pd.DataFrame()

dateindex2 = data.loc[end_date1:end_date].index

#temp = pd.DataFrame([dateToBeSold,1],columns=['date','qty'])
column_names = ["date", "qty"]

sellDates = pd.DataFrame(columns = column_names)

#set[dateindex2[1].strftime('%Y-%m-%d')]
for i in dateindex2:
    
    idate = i.strftime('%Y-%m-%d')        
    
    #process purchases
    if (idate in set.index):

        temp = pd.DataFrame()
        
        estRet = set.loc[idate]['ExpectedReturn']

        temp['orderside'] = ['buy']        
        
        if len(data[start_date:idate])-1+n_forward>=len(data[start_date:]):
            dateToBesold = np.nan    
            temp['valueAtSale'] = np.nan
        else:
            dateToBeSold = data.ix[len(data[start_date:idate])-1+n_forward].name.strftime('%Y-%m-%d') 
            
            temp['valueAtSale'] = pd.DataFrame(data.ix[len(data[start_date:idate])-1+n_forward]).transpose()['Close'].values[0]            
         
        temp['date'] = [idate]
        temp['valueAtPurchase'] = set.loc[idate]['Close']
        temp['estRet'] = estRet
        #temp['qty'] = Qty
        temp['dateBought'] = idate        
        temp['dateToBeSold'] = dateToBeSold
        

        btemp = pd.DataFrame(columns = column_names)
        btemp["date"]=[dateToBeSold]
        #btemp["qty"]=[Qty]

        sellDates = sellDates.append(btemp,ignore_index=True)
  
        temp = temp.round(4)

        orderbook = orderbook.append(temp,ignore_index=True)
        
for i in dateindex2:
    
    idate = i.strftime('%Y-%m-%d')        
    
    #process sales
        
    if (idate in sellDates.set_index('date').index):    
        temp = pd.DataFrame()

        dateBought = data.ix[len(data[start_date:idate])-1-n_forward].name.strftime('%Y-%m-%d')        
        dateToBeSold = idate
        temp['dateBought'] = [dateBought]
        temp['dateToBeSold'] = dateToBeSold
        temp['valueAtPurchase'] = pd.DataFrame(data.ix[len(data[start_date:idate])-1-n_forward]).transpose()['Close'].values[0]
        estRet = set.loc[dateBought]['ExpectedReturn']
        temp['estRet'] = estRet
        temp['valueAtSale'] = pd.DataFrame(data.ix[len(data[start_date:idate])-1]).transpose()['Close'].values[0]
        
        #temp['dateToBeSold'] = idate
        #temp['estRet'] = data.loc[idate]['Forward Return']

        temp['orderside'] = ['sell']        
        temp['date'] = [idate]

        #data vs set because set only includes buy dates
        #temp['valueAtSale'] = pd.DataFrame(data.ix[len(data[start_date:idate])-1+n_forward]).transpose()['Close']

        #temp['qty'] = sellDates.set_index('date').loc[idate]['qty']

        temp = temp.round(4)

        orderbook = orderbook.append(temp,ignore_index=True)


orderbook.sort_values(by=['date','orderside'], ascending=True)


funds = 1000
BuyFundsPercent = .25
percentHeldOnSell = 1

buyLog = pd.DataFrame()
sellLog = pd.DataFrame()
runningLog = pd.DataFrame()

held = 0
upper = 0
lower = 0

for i in dateindex2:
    
    temp = pd.DataFrame()
    rtemp = pd.DataFrame()
    _temp = pd.DataFrame()
    
    t = i.strftime('%Y-%m-%d')
    
    subset = orderbook[orderbook['date']==t]
    gain = 0
    paid = 0
    
    if len(subset) != 0:
        
        sales = subset[subset['orderside'] == 'sell']
        
        print("date " + str(i))
        
        if len(sales) != 0:                        
            
            oldvalue = sales['valueAtPurchase'].values[0]

            newvalue = sales['valueAtSale'].values[0]            
            
            Qty = buyLog.set_index('date').loc[sales['dateBought'].values[0]].values[0]
            print("Qty sold " + str(Qty.round(2)))
            
            gain = newvalue * Qty
            
            _temp['date'] = [i]
            _temp['qty'] = [Qty]
            
            sellLog = sellLog.append(_temp)
    
        purchases = subset[subset['orderside'] == 'buy']
        
        if len(purchases) != 0:
    
            ProportionOfFunds = funds * BuyFundsPercent
        
            Qty = ProportionOfFunds / purchases['valueAtPurchase'].values[0]
            print(purchases['valueAtPurchase'].values[0])
            print("Qty purchased " + str(Qty.round(2)))
            
            temp['date'] = [i]
            temp['qty'] = [Qty]
            
            paid = purchases['valueAtPurchase'].values[0]*Qty
                        
            buyLog = buyLog.append(temp)
        
        funds = funds + gain - paid
                    
        rtemp['date'] =  [i]
        rtemp['funds'] =  [funds]
        
        if len(sellLog) != 0:
            remainder = (sum(buyLog['qty'])-sum(sellLog['qty']))            
            
        else:
            remainder = (sum(buyLog['qty']))
        
        rtemp['held'] = remainder
        rtemp['value'] = remainder * data.loc[i]['Close']
        rtemp['portValue'] = funds + remainder * data.loc[i]['Close']
                
        print("in " + str(gain))
        print("out " + str(paid))
        print("held: " + str(remainder))
        print("Close Value: " + str(data.loc[i]['Close']))
        print("held Value: " + str(remainder * data.loc[i]['Close']))
        print("funds " + str(funds))
        print("portValue " + str(funds + remainder * data.loc[i]['Close']))
        print()
            
        runningLog = runningLog.append(rtemp)
                
buyLog
plt.plot(runningLog.set_index('date')['portValue'])
#plt.plot(runningLog.set_index('date')['funds'])
plt.xticks(rotation=30) 
#subset
#plt.plot(runningLog.set_index('date'))

#display(orderbook.dropna())
#plt.plot(orderbook.set_index('date').dropna()['portValue'])
#plt.xticks(rotation=30) 
#plt.show()
#len(orderbook.dropna())

#plt.hist(orderbook['profit'], bins='auto')  # arguments are passed to np.histogram
#orderbook['profit'].dropna().sum()