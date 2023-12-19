import yfinance as yf
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st



st.title("Stock Price Prediction")


DATA_PATH = "msft_data.json"
user_Input = st.text_input("Enter Stock Ticker","TSLA")

# if os.path.exists(DATA_PATH):
#     # Read from file if we've already downloaded the data.
#     with open(DATA_PATH) as f:
#         msft_hist = pd.read_json(DATA_PATH)
# else:
msft = yf.Ticker(user_Input)
msft_hist = msft.history(period="max")

    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
msft_hist.to_json(DATA_PATH)

st.subheader("Stock Price History")
st.write(msft_hist.head(5))

st.subheader("Closing Price VS Time Chart")
# fig = plt.figure(figsize = (12,6))
# msft_hist.plot.line(y="Close", use_index=True)
# st.pyplot(fig)
st.line_chart(msft_hist["Close"])

# Ensure we know the actual closing price
data = msft_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)

# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]

# st.subheader("Preparing Training Data")
data.head(5)

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# Create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)

combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()

def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit the random forest model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0
        
        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
    
    return pd.concat(predictions)

predictions = backtest(data, model, predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()
weekly_trend = data.shift(1).rolling(7).mean()["Target"]

data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]

full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
predictions = backtest(data.iloc[365:], model, full_predictors)

temp = precision_score(predictions["Target"], predictions["Predictions"])
accuracy_percentage = temp * 100
st.subheader("Overall Results")
# st.text(f"Accuracy: {accuracy_percentage:.2f}%")

# Show how many trades we would make
value_counts = predictions["Predictions"].value_counts()
second_value_count = value_counts.iloc[1]
st.text(f"You can do {second_value_count} Trades the next day with {accuracy_percentage:.2f}% accuracy")

predictions.iloc[-100:].plot()

