import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "Stock Predictions Model.keras"
model = load_model(MODEL_PATH)
 
st.header("Stock Market Predictor")

stock = st.text_input("Enter Stock Symbol", "GOOG")
start = "2012-01-01"
end = "2022-12-31"

 
data = yf.download(stock, start=start, end=end, progress=False)

 
if data is None or data.empty:
    st.error("❌ No stock data found. Please check the symbol or try again later.")
    st.stop()

st.subheader("Stock Data")
st.write(data.tail())
 
close_prices = data['Close'].values.reshape(-1, 1)

train_size = int(len(close_prices) * 0.80)

if train_size <= 100:
    st.error("❌ Not enough data for prediction.")
    st.stop()

data_train = close_prices[:train_size]
data_test = close_prices[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)
 
past_100_days = data_train[-100:]
data_test = np.vstack((past_100_days, data_test))

data_test_scale = scaler.transform(data_test)

 
st.subheader(" Price vs MA50")
ma_50 = data['Close'].rolling(50).mean()

fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r', label="MA50")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig1)

st.subheader(" Price vs MA50 vs MA100")
ma_100 = data['Close'].rolling(100).mean()

fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50, 'r', label="MA50")
plt.plot(ma_100, 'b', label="MA100")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig2)

st.subheader(" Price vs MA100 vs MA200")
ma_200 = data['Close'].rolling(200).mean()

fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100, 'r', label="MA100")
plt.plot(ma_200, 'b', label="MA200")
plt.plot(data['Close'], 'g', label="Close Price")
plt.legend()
st.pyplot(fig3)

 
x, y = [], []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

 
predicted = model.predict(x, verbose=0)

 
scale_factor = 1 / scaler.scale_[0]
predicted = predicted * scale_factor
y = y * scale_factor

 
st.subheader(" Original Price vs Predicted Price")

fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label="Original Price")
plt.plot(predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
