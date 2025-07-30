# Stock-price-prediction
import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
import matplotlib.dates as mdates

# fetching data from yfinance

#today=date.today(),strftime("%Y-%m-%d")

def load_data(ticker):
  data=yf.download(ticker,start="2021-01-01",end="2025-01-01")
  data.reset_index(inplace=True)
  return data

data =load_data('IOB.NS') #indian overseas bank
df=data
df
#visualizing of closing price
df['Date']=pd.to_datetime(df['Date'])
plt.figure(figsize=(13,7))
plt.plot(df['Date'],df['Close'], label="IOB Stock Price", color="pink")
plt.title("IOB Stock Price Graphical Representation IOB Stock Price")
plt.xlabel("Date")
plt.ylabel("Price(INR)")
plt.grid(True)
plt.legend()
plt.show
# moving average of 200 days
ma200=df.Close.rolling(200).mean()
ma200
# graphical representation of 200MA
df['Date']=pd.to_datetime(df['Date'])
plt.figure(figsize=(13,7))
plt.plot(df['Date'],df['Close'], label="close price")
plt.plot(df['Date'],ma200, label="200-day moving average")
plt.xlabel("Date")
plt.ylabel("Price(INR)")
plt.grid(True)
plt.title("200 day moving average")
plt.legend()
plt.show

# moving average of 300 days
ma300=df.Close.rolling(300).mean()
ma300
#Graphical representation of 300 days moving charge
df['Date']=pd.to_datetime(df['Date'])
plt.figure(figsize=(13,7))
plt.plot(df['Date'],df['Close'], label="close price")
plt.plot(df['Date'],ma300, label="300-day moving average")
plt.xlabel("Date")
plt.ylabel("Price(INR)")
plt.grid(True)
plt.title("300 day moving average")
plt.legend()
plt.show

# comparison of 200 and 300 Moving average
df['Date']=pd.to_datetime(df['Date'])
plt.figure(figsize=(13,7))
plt.plot(df['Date'],df['Close'], label="Actual Close price")
plt.plot(df['Date'], ma200,label="200 days moving charge")
plt.plot(df['Date'],ma300, label="300 days moving charge")
plt.xlabel("Date")
plt.ylabel("Price(INR)")
plt.title("Comparison of 200 and 300 days moving charge")
plt.legend()
plt.grid(True)
plt.show()
df.shape
#splitting data into training(70%) and testing(30%)
train=pd.DataFrame(data[0:int(len(data)*.70)])
test=pd.DataFrame(data[int(len(data)*0.70):int(len(data))])
print(train.shape)
print(test.shape)
#minMax scalar for nomalization of dataset
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
train_close=train.iloc[:,4:5].values
test_close=test.iloc[:,4:5].values
data_training_array=scaler.fit_transform(train_close)
data_training_array
x_train=[]
y_test=[]
for i in range(100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_test.append(data_training_array[i,0])
x_train,y_train=np.array(x_train),np.array(y_test)
x_train.shape
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
# preparaing data for Linear regression
train_close_flattened = train_close.flatten()
X_lr = np.array([train_close_flattened[i:i + 100] for i in range(len(train_close_flattened) - 100)])
y_lr = train_close_flattened[100:]
# Split into training and validation sets using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
linear_model = LinearRegression()
# Cross-validation
cv_scores = cross_val_score(linear_model, X_lr, y_lr, cv=tscv, scoring="neg_mean_squared_error")
cv_rmse_scores = np.sqrt(-cv_scores)
print("Cross-validation RMSE Scores:", cv_rmse_scores)
print("Mean RMSE:", cv_rmse_scores.mean())
# Train the model on the full training data
linear_model.fit(X_lr, y_lr)
# Predict on the test set
test_close_flattened = test_close.flatten()
X_test_lr = np.array([test_close_flattened[i:i + 100] for i in range(len(test_close_flattened) - 100)])
y_test_lr = test_close_flattened[100:]

y_pred_lr = linear_model.predict(X_test_lr)
# Evaluate the Linear Regression model
test_rmse = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
print("Linear Regression Test RMSE:", test_rmse)
# Plot predictions vs actual prices for test data
plt.figure(figsize=(13, 7))
plt.plot(range(len(y_test_lr)), y_test_lr, label="Actual Prices", color="blue")
plt.plot(range(len(y_pred_lr)), y_pred_lr, label="Predicted Prices", color="orange")
plt.title("Linear Regression - Actual vs Predicted Prices")
plt.xlabel("Time Step")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()
# Predict future price
# Use the last 100 data points from the training set to predict the next price
last_sequence = train_close_flattened[-100:].reshape(1, -1)
future_price_lr = linear_model.predict(last_sequence)
print("Predicted Future Price using Linear Regression:", future_price_lr[0])

