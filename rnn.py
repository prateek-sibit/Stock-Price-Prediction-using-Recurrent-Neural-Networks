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
sb.set_style('darkgrid')

# Importing the Training Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
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

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, 80):
    # range is (60,80) because we take the data of previous 60 days for the month of January
    # January 2017 has 20 Financial days thus the range 60-(60+20)
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
# Reshaping
X_test = np.reshape(X_test, newshape=(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
# Inverse the scaling to get original scale
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results

# Plotting the Real Google stock price
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Part 4 - Improving and Tuning the RNN Model 

from keras.wrappers.scikit_learn import KerasRegressor
# Used for wrapping the keras model into scikit-learn
from sklearn.model_selection import GridSearchCV

def build_regressor(optimizer,units):
    regressor = Sequential()
    # Adding the first LSTM Layer and some dropout regularization
    regressor.add(LSTM(units=units,return_sequences=True,input_shape=(X_train.shape[1],1)))
    regressor.add(Dropout(0.2))    
    # Adding a second LSTM Layer and some dropout regularization
    regressor.add(LSTM(units=units,return_sequences=True))
    regressor.add(Dropout(0.2))    
    # Adding a third LSTM Layer and some dropout regularization
    regressor.add(LSTM(units=units,return_sequences=True))
    regressor.add(Dropout(0.2))    
    # Adding a fourth LSTM Layer and some dropout regularization
    regressor.add(LSTM(units=units,return_sequences=True))
    regressor.add(Dropout(0.2))
    # Adding a fifth LSTM Layer and some dropout regularization
    regressor.add(LSTM(units=units))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units=1))
    # Compiling the RNN
    regressor.compile(optimizer=optimizer,loss='mean_squared_error')
    return regressor

regressor = KerasRegressor(build_fn=build_regressor)

# Create a Dictionary of Hyperparameters that we want to optimize
parameters = {'batch_size':[25,32,64],
              'epochs':[100,200,500],
              'optimizer':['adam','rmsprop'],
              'units':[50,100,150]}

# Create an object of GridSearchCV and it on the training set like a normal model
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=10)
# Fitting on the training set
grid_search.fit(X_train,y_train)

# Best Parameters
best_parameters = grid_search.best_params_    

