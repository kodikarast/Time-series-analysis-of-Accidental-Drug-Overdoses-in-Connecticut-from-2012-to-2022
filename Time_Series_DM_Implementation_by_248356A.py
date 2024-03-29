#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Time series DM implementation by 248356A

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing data file
df_train = pd.read_csv("C:/Users/sanjeewa_10133/Documents/Python Scripts/ADM/CT_New.csv")


# In[3]:


df_train.head()


# In[4]:


eda_train = df_train.copy()
eda_train


# In[5]:


#date and time separation
eda_train['Date'] = pd.to_datetime(eda_train['Date'])

daily_deaths=eda_train.resample('D', on='Date')['Death State'].sum().to_frame()
weekly_deaths=eda_train.resample('W', on='Date')['Death State'].sum().to_frame()
monthly_deaths=eda_train.resample('M', on='Date')['Death State'].sum().to_frame()


# In[71]:


#Calculating autocorrelation of total deaths
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(monthly_deaths)


# In[74]:


#calculating 1st order differencing and auto correlation
f=plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title('1st order Differencing')
ax1.plot(monthly_deaths.diff())

ax2 = f.add_subplot(122)
plot_acf(monthly_deaths.diff().dropna(), ax=ax2)
plt.show()


# In[75]:


#calculating 2nd order differencing and their auto correlations
f=plt.figure()
ax1 = f.add_subplot(121)
ax1.set_title('2nd order Differencing')
ax1.plot(monthly_deaths.diff().diff())

ax2 = f.add_subplot(122)
plot_acf(monthly_deaths.diff().diff().dropna(), ax=ax2)
plt.show()


# In[50]:


#Calulating movig average for total deaths

trend_deaths=eda_train.groupby('Date')['Death State'].sum()
moving_avg=trend_deaths.rolling(window=365,center=True).mean().reset_index()
moving_avg
plt.plot(moving_avg['Date'], moving_avg['Death State'])
plt.xlabel('Date')
plt.ylabel('Avg. Deaths')
plt.title('Avg. Deaths Trend Over Years')
plt.show()


# In[9]:


#seasonal decomposing to analyze trend, seasonal components and residues
from statsmodels.tsa.seasonal import seasonal_decompose


decompose_result = seasonal_decompose(weekly_deaths,model="additive")
decompose_result.plot()


# In[76]:


#Calculation of adfuller stat
from statsmodels.tsa.stattools import adfuller


result = adfuller(monthly_deaths.dropna())
print ('p-value', result[1])

result = adfuller(monthly_deaths.diff().dropna())
print ('p-value', result[1])

result = adfuller(monthly_deaths.diff().diff().dropna())
print ('p-value', result[1])


# p value is not less than 0.05, thus a non-stationary series for 0th order
# p value is lower than the treshold for 1st order. thus p=1 for ARIMA model

# In[12]:


#calculation of auto arima to get an idea on order and seasonal order for SARIMAX calculation
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Grid search for seasonal orders
auto_model = auto_arima(monthly_deaths, seasonal=True, m=12)  # Assuming monthly seasonality
order, seasonal_order = auto_model.order, auto_model.seasonal_order


# In[13]:


order, seasonal_order


# In[65]:


monthly_deaths


# In[46]:


#SARIMAX model for total monthly deaths and fitting the model
model_sarimax_monthly = SARIMAX(monthly_deaths, order=(0, 1, 2), seasonal_order=(1, 1, 1, 12))
results_monthly = model_sarimax_monthly.fit()


# In[47]:


#getting to forecast for 60 more months
forecast_sarimax_monthly = results_monthly.get_forecast(steps=60)
forecast_values_monthly = forecast_sarimax_monthly.predicted_mean


# In[48]:


#Plotting SARIMAX model and time series forecasting
monthly_deaths.plot(legend=True, label='Actual_deaths_monthly', figsize=(15,4))

forecast_values_monthly.plot(legend=True, label= 'Forecast_deaths_monthly')

