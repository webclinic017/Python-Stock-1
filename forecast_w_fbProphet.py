from fastquant import get_crypto_data, backtest
from fbprophet import Prophet
from matplotlib import pyplot as plt
from fbprophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

outter_df = get_crypto_data("BTC/USDT", "2019-01-01", "2020-05-31")

#1 year and 1 quarter
oneyroneqtr=456
length = len(outter_df[0:oneyroneqttr])

custompre = pd.DataFrame()

def MAPE(Y_actual,Y_Predicted):
    mape_ = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape_

for x in range(0,len(outter_df)):
    frames = outter_df.index[x:(x+oneyroneqttr)]
    length = len(frames)
    if length == oneyroneqtr:
        #print(outter_df.loc[frames])
        
        df = outter_df.loc[frames]
        
        ts = df.reset_index()[["dt", "close"]]
        ts.columns = ['ds', 'y']

        #strips last day from model
        m = Prophet(daily_seasonality=True,yearly_seasonality=True)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
        m.fit(ts[0:-1])

        #forecast only last day in model (can verify result)
        forecast = pd.DataFrame(df.index)[-1:]
        forecast.columns = ['ds']
        
        #Predict and plot
        pred = m.predict(forecast)
        
        temp = pd.DataFrame()
        #print(pred)
        #print(pred['yhat'].values[0])
        temp['date'] = frames[-1:].strftime('%Y-%m-%d')
        temp['yhat'] = pred['yhat']
        custompre = custompre.append(temp, ignore_index=True)
        
    else:
        break

        
# Convert predictions to expected 1 day returns
custom = custompre
custom.columns = ["ds","yhat"]

merge=pd.merge(outter_df,custom.set_index('ds'), how='inner', left_index=True, right_index=True)
merge.index.names = ['dt']
expected_1day_return = ((merge['yhat']-merge['close'])/merge['close']).multiply(100)

df = merge

df["custom"] = expected_1day_return
#.multiply(-1)

df.to_csv('btresults.csv', index =True)
