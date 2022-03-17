import io
import warnings
import sys
from numpy.linalg import LinAlgError
warnings.filterwarnings('ignore')
import pandas as pd
from pandas import read_excel, read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def preprocess(data):
  data['SKU id']=pd.Series([int(id[4:]) for id in data["SKU id"].values])
  data['Warehouse id']=pd.Series([int(id[3:]) for id in data["Warehouse id"].values])
  return data

def arima_forecast(data):

  predictions_1, predictions_2, final_predictions_1, final_predictions_2 = [], [], [], []
  for tsize in range(1,3):
  
    pred_list = []
    for row in range(len(data)):

      min_mape = sys.float_info.max
      pred_ = []
      final_pred = 0.0
      rmse_ = 0.0

      for lag in range(1,2):

        predictions = []
        x = data.iloc[row,3:].values
        train, test = x[:-tsize], x[-tsize:]
        history = [i for i in train]

        for t in range(len(test)):
          
          try :
            model = ARIMA(history, order=(12,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(int(yhat))
            obs = test[t]
            history.append(obs)

          except LinAlgError:
            return predictions_1[:], predictions_2[:], final_predictions_1[:], final_predictions_2[:]

        model = ARIMA(list(x), order=(lag,1,0))
        model_fit = model.fit()
        output = model_fit.get_forecast(steps=1).predicted_mean
        predictions.append(int(output[0]))
        mape = mean_absolute_percentage_error(test, predictions[:-1])
        rmse = sqrt(mean_squared_error(test, predictions[:-1]))

        if mape < min_mape :
          min_mape = mape
          pred_ = predictions[:]
          rmse_ = rmse
          
      pred_list.append(pred_[:])

    expected = list(data.iloc[:,-1])
  
    if tsize == 1:

      print("Size of test data :",tsize)
      for idx in range(len(pred_list)) :

        predicted = pred_list[idx][0]
        final_predicted = pred_list[idx][1]
        predictions_1.append(predicted)
        final_predictions_1.append(final_predicted)

      mape = mean_absolute_percentage_error(expected, predictions_1)
      rmse = sqrt(mean_squared_error(expected, predictions_1))
      print('MAPE : %.3f' % mape)
      print('RMSE : %.3f' % rmse)

    elif tsize == 2:

      print("Size of test data :",tsize)
      for idx in range(len(pred_list)) :

        predicted = pred_list[idx][1]
        final_predicted = pred_list[idx][2]
        predictions_2.append(predicted)
        final_predictions_2.append(final_predicted)

      mape = mean_absolute_percentage_error(expected, predictions_2)
      rmse = sqrt(mean_squared_error(expected, predictions_2))
      print('MAPE : %.3f' % mape)
      print('RMSE : %.3f' % rmse)
  return predictions_1[:], predictions_2[:], final_predictions_1[:], final_predictions_2[:]

def arima(data):

  wh = data.groupby('Warehouse id') 
  wh_dict = {}                             #separate dataframes for each warehouse
  for i in range(1,5):

    wh_data = wh.get_group(i)
    print('\n#### WAREHOUSE {} ####\n'.format(i))
    predictions_1, predictions_2, final_predictions_1, final_predictions_2 = arima_forecast(wh_data)
    wh_data_ = wh_data.copy()
    if len(predictions_1) != 0 :
      wh_data_['Predictions_Last1'] = predictions_1
      wh_data_['June-21_Last1'] = final_predictions_1
    if len(predictions_2) != 0 :
      wh_data_['Predictions_Last2'] = predictions_2
      wh_data_['June-21_Last2'] = final_predictions_2
    wh_dict[str(i)] = wh_data_

  return wh_dict

df_ma = read_excel('anomaly_ma.xlsx')
df_ma.head()

df_ma = preprocess(df_ma)

type_ = "IQR_MA"
wh_dict = arima(df_ma)

print(wh_dict['4'])