📈 Stock Market Prediction Model

This project is a Stock Market Price Prediction Web Application built using Deep Learning (LSTM) and deployed with Streamlit. It predicts future stock closing prices based on historical market data and visualizes trends using moving averages.

🚀 Features

Fetches real-time historical stock data using Yahoo Finance

Predicts future stock prices using a trained LSTM model

Interactive Streamlit web interface

Visualizations for:

Closing Price vs Moving Average (50, 100, 200 days)

Original Price vs Predicted Price

Supports any stock symbol available on Yahoo Finance

🧠 Model Overview

Algorithm: Long Short-Term Memory (LSTM)

Framework: Keras (TensorFlow backend)

Input: Previous 100 days of closing prices

Output: Predicted closing price

Scaling: MinMaxScaler (0–1 range)

The model was trained on historical stock closing prices and saved as a .keras file.

🛠️ Tech Stack

Python

NumPy

Pandas

yFinance

Scikit-learn

Keras / TensorFlow

Matplotlib

Streamlit

📂 Project Structure
├── app.py                                  # Streamlit web app
├── Stock_Market_Prediction_Model_Creation.ipynb  # Model training notebook
├── Stock Predictions Model.keras           # Trained LSTM model
├── README.md                               # Project documentation

🖥️ How It Works

User enters a stock symbol (e.g., GOOG, AAPL, TSLA)

Historical data is downloaded from Yahoo Finance

Data is split into training and testing sets

Last 100 days are used to predict future prices

Results are visualized with interactive charts

📊 Visualizations Included

Closing Price vs MA50

Closing Price vs MA50 vs MA100

Closing Price vs MA100 vs MA200

Original Price vs Predicted Price
