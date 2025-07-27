import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

from utils import add_indicators  # <-- Import the helper

st.set_page_config(page_title="Stock Price LSTM Predictor", layout="wide")
st.title("📈 Stock Price Trend Prediction using LSTM")

ticker = st.text_input("Enter Stock Ticker(s)(e.g., AAPL, TSLA, MSFT,GOOGL,AMZN,EBAY,AMD,NVDA,FB,CSCO,IBM)", "AAPL,TSLA,MSFT,GOOGL,AMZN,EBAY,AMD,NVDA,FB,CSCO,IBM").upper()

if st.button("Run Prediction"):
    tickers = [t.strip() for t in ticker.split(",")]
    df = yf.download(tickers, start='2015-01-01', end='2024-01-01')['Close']
    # df is now a DataFrame with columns for each ticker's Close price

    for t in tickers:
        st.subheader(f"{t} Analysis")
        stock_df = pd.DataFrame(df[t])
        stock_df.columns = ['Close']
        stock_df = add_indicators(stock_df)
        stock_df = stock_df.dropna()

        # Corrected code:
        st.write("### Close Price and SMA (50-day)")
        st.line_chart(stock_df[['Close', 'SMA_50']])

        st.subheader("RSI Indicator")
        fig, ax = plt.subplots()
        stock_df['RSI'].plot(ax=ax)
        ax.axhline(70, color='red', linestyle='--')
        ax.axhline(30, color='green', linestyle='--')
        st.pyplot(fig)

        st.subheader("MACD Indicator")
        fig2, ax2 = plt.subplots()
        stock_df['MACD'].plot(ax=ax2, label='MACD')
        stock_df['Signal'].plot(ax=ax2, label='Signal')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Bollinger Bands")
        fig3, ax3 = plt.subplots()
        ax3.plot(stock_df['Close'], label='Close')
        ax3.plot(stock_df['20_SMA'], label='20 SMA')
        ax3.fill_between(stock_df.index, stock_df['Upper'], stock_df['Lower'], color='gray', alpha=0.3)
        ax3.legend()
        st.pyplot(fig3)

    # LSTM Preparation
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_df[['Close']])

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
    st.success(f"✅ RMSE: {rmse:.2f}")

    st.subheader("📉 Actual vs Predicted Stock Price")
    result_df = pd.DataFrame({
        'Actual Price': actual_prices.flatten(),
        'Predicted Price': predicted_prices.flatten()
    })
    st.line_chart(result_df)

    st.subheader("Model Summary")
    st.text(model.summary())

    st.subheader("Download Prediction Results")
    result_df.index = stock_df.index[-len(result_df):]  # Align index with stock_df
    result_df.reset_index(inplace=True)
    result_df.rename(columns={'index': 'Date'}, inplace=True)
    st.write(result_df)

    # Download button for CSV
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{ticker}_prediction.csv", "text/csv")
    # Function to add technical indicators to the DataFrame
    st.success("Prediction completed successfully!")
else:
    st.info("Enter a stock ticker and click 'Run Prediction' to see the results.")
# This code is a Streamlit app that allows users to input a stock ticker and view the stock's historical data, technical indicators, and LSTM-based predictions.
# It includes features like downloading the data, calculating indicators like RSI, MACD, and Bollinger Bands, and visualizing the results with charts.
# The app also provides a download option for the prediction results in CSV format and displays the model summary and performance metrics.  