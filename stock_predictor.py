import math
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf 
import os
plt.style.use("fivethirtyeight")


STARTDATE = "2021-01-01"
ENDDATE = "2022-12-19"
STOCKNAME = 'AAPL'

def get_stock_quote(start_date, end_date, stock_name):
	#get the stock quote
	df = yf.download(stock_name, 
	                      start=start_date, 
	                      end=end_date, 
	                      progress=False)
	return df


def test():
	data = get_stock_quote(STARTDATE, ENDDATE, STOCKNAME)

	df = data.filter(['Close'])
	dataset = df.values

	training_data_len = math.ceil(len(dataset) * 0.8)

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_data = scaler.fit_transform(dataset)

	train_data = scaled_data[0:training_data_len, :]

	x_train = []
	y_train = []
	for i in range(60,len(train_data)):
		x_train.append(train_data[i-60:i, 0])
		y_train.append(train_data[i,0])

	x_train, y_train = np.array(x_train), np.array(y_train)
	x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

	shape1 = np.shape(x_train[-1:])
	print(shape1)

#Train Data

#test_data = scaled_data[training_data_len - 60:, :]

def test_model(model):
	x_test = []
	y_test = dataset[training_data_len:, :]
	for i in range(60, len(test_data)):
		x_test.append(test_data[i-60:i, 0])

	x_test = np.array(x_test)

	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


	predictions = model.predict(x_test)
	predictions = scaler.inverse_transform(predictions)

	print(predictions)

	train = data[:training_data_len]
	valid = data[training_data_len:]
	valid['Predictions'] = predictions


	plt.figure(figsize = (12,6))
	plt.title('Model')
	plt.xlabel('Date', fontsize = 18)
	plt.ylabel('Close Price USD ($)', fontsize = 18)
	plt.plot(train['Close'])

	def without_validation():
		plt.plot(valid[['Predictions']])
		plt.legend(['Train','Predictions'], loc = 'lower right')

	def with_validation():
		plt.plot(valid[['Close', 'Predictions']])
		plt.legend(['Train', 'Validation', 'Predictions'], loc = 'lower right')

	with_validation()

	plt.show()



def fit_and_save(name):
	model.fit(x_train, y_train, batch_size = 1, epochs = 1)
	model.save(name)

def reconstruct(name):
	reconstruction_model = tf.keras.models.load_model(name)
	return reconstruction_model




data = get_stock_quote(STARTDATE, ENDDATE, STOCKNAME)

df = data.filter(['Close'])
dataset = df.values

scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(dataset)

x_train = []
y_train = []
for i in range(60,len(train_data)):
	x_train.append(train_data[i-60:i, 0])
	y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))






model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Optimize Model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')





NAME = "tesla_2023.02"
#fit_and_save(NAME)
tesla_model = reconstruct(NAME)


my_list = x_train[-1:].tolist()
print(my_list)


stock_prices = []

for x in my_list[0]:
	stock_prices.append(x)

DAYS = 16


def predicting_days_after(days):
	days_after = []
	for i in range(days+1):
		last_60_days = stock_prices[-60:]

		prediction_list = []
		prediction_list.append(last_60_days)


		predictions = tesla_model.predict(prediction_list)
		stock_prices.append(predictions[0])
		days_after.append(predictions[0])
	return days_after


prediction = predicting_days_after(DAYS)


prediction = scaler.inverse_transform(prediction)
print(prediction)



Start_date = "2022-12-20"
End_date = "2023-01-14"
Stock_name = 'AAPL'


valid_data = get_stock_quote(Start_date, End_date, Stock_name)

valid_data_len = len(valid_data) - 17

valid = valid_data[0:]
print(valid)
valid['Predictions'] = prediction
print(valid)



plt.figure(figsize = (12,6))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Actual', 'Theoretical', 'Predictions'], loc = 'lower right')
plt.show()







