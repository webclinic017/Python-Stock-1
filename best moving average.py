#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance
import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from datetime import date

from scipy.stats import ttest_ind


# In[116]:


n_forward = 7
name = 'BTC-USD'

w=117
end_date = datetime.date.today()
end_date1 = end_date - timedelta(weeks=w)

#- timedelta(weeks=w*2)
start_date = end_date1 - timedelta(weeks=w)

ticker = yfinance.Ticker(name)
data = ticker.history(interval="1d",start=start_date,end=end_date, auto_adjust=True)
data['Forward Close'] = data['Close'].shift(-n_forward)
data['Forward Return'] = (data['Forward Close'] - data['Close'])/data['Close']


# In[117]:





# In[73]:


dateindex = data.loc[start_date:end_date].index

dateindex


# In[181]:


limit = 100
n_forward = 7

train_size = 0.5

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
        if result[0]['training_forward_return'] > 0:
            if result[0]['test_forward_return'] > 0:
                if temp.ix[-1]['Close']>temp.ix[-1]['SMA']:
        
                    print(result[0])
                    print(temp[-1:]['Close'])
                    print(temp[-1:]['SMA'])
                
                    plt.plot(temp['Close'],label='BTC')

                    stringLabel = str(result[0]['sma_length']) + " SMA" + " at " + str(n_forward) + " day return " + str(result[0]['test_forward_return'].round(3))
                    plt.plot(data['Close'].rolling(result[0]['sma_length']).mean(),label = stringLabel)
                    plt.legend()

                    plt.show()
        
        
    


# In[176]:





# In[179]:





# In[97]:





# In[11]:





# In[12]:


#data['Close'].rolling(result[0]['sma_length']).mean()


# In[13]:





# In[ ]:




