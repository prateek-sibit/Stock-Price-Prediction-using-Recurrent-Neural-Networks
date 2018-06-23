# Stock-Price-Prediction-using-Recurrent-Neural-Networks
Predicting the stock price of Google for the month of January 2017 using previous data from 2012-2016

## About the Files in this Project

- Google_Stock_Price_Prediction.png : A Visual Representation of the comparision of the models performance. The analysis shows us
  that the model is able to capture the trend (rising or falling) of the stocks however it fails at capturing the sudden peaks and
  non linear changes

- Google_Stock_Price_Test.csv : The data in csv format that is used to train the model

- Google_Stock_Price_Test.csv : The data in csv format that is used to test the model

- rnn.py : Main python file containing the project code

 
## Dataset Information :

For this project the Google Stock Data has been divided into 2 parts the training and the testing part.

Range of the dataset : 1/3/2012 - 1/31/2017

Total Number of Samples : 1278
Training Samples : 1257
Testing Samples : 19

## Attribute Information 

Each stock is described by 6 continuous variables. These are summarised below: 

1. Date 
2. Open 
3. High 
4. Low
5. Close
6. Volume 

## Improving the RNN Performance

1. Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.

2. Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).

3. Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.

4. Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.

5. Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.
