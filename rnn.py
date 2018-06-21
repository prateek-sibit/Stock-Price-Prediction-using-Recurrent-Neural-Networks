# Recurrent Neural Network

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from matplotlib import rcParams

# Configuring some Visualization settings
rcParams['figure.figsize'] = (14,7)
rcParams['font.size'] = 12
sb.set_style('whitegrid')

# Importing the Training Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv',parse_dates=['Date'])
dataset_train.set_index(keys=['Date'],inplace=True)
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# It is recommended to keep the original dataset 
training_set_scaled = scaler.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
""" This is important if we want to add more indicators and to feed it into the 
    Neural Netork """
X_train = np.reshape(X_train, newshape=(X_train.shape[0],X_train.shape[1],1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM Layer and some dropout regularization
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM Layer and some dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a third LSTM Layer and some dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM Layer and some dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train,y_train,batch_size=32,epochs=100)


# Part 3 - Making the predictions and visualising the results
