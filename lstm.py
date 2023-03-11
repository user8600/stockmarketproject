import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import plotly.offline as pyo
import plotly.express as px
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('stock_data.csv')

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df[['Close']])

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)


# Reshape the data into a 3-dimensional tensor
#x_train = np.array(train_data['Close']).reshape((-1, 1, 1))
#x_test = np.array(test_data['Close']).reshape((-1, 1, 1))

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, features), return_sequences=True))
model.add(Dropout(0.2))

# Add 4 more LSTM layers
for i in range(4):
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

# Flatten the output of the previous LSTM layers
model.add(Flatten())

# Add 5 dense hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))



# Add the final layer to produce the output
model.add(Dense(output_dim, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import matplotlib.pyplot as plt

# Obtain the predictions on the test data
predictions = model.predict(x_test)

# Inverse transform the predictions to get the original scale
predictions = scaler.inverse_transform(predictions)

# Plot the actual and predicted values
plt.plot(test_data['Close'], label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
