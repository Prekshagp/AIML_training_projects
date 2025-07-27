# 📉 Stock Price Predictor

This is an interactive stock price prediction and analysis tool built with **Streamlit** and **LSTM (Long Short-Term Memory)** models. It allows users to input stock tickers and visualize key technical indicators like **Bollinger Bands**, **MACD**, and **RSI**, along with trend predictions using deep learning.

---

## 🚀 Features

- 📊 **Bollinger Bands**: Visualize stock price volatility and trend boundaries  
- 🔁 **MACD**: Analyze momentum and signal line crossovers  
- 📈 **RSI**: Detect overbought or oversold market conditions  
- 🤖 **LSTM-Based Forecasting**: Predict future price movements using historical data  
- 📥 **Downloadable CSV**: Export prediction results for further analysis  
- 🖥️ **Streamlit UI**: User-friendly interface for input and visualization

---

## ⚙️ Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock-price-predictor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## ▶️ Running the Application

To run the application, execute the following command:
```
streamlit run src/app.py
```

🌐This will start the Streamlit server and open the application in your default web browser.

## 🧑‍🏫 Usage

📝 Enter one or more stock ticker symbols (e.g., AAPL, TSLA, MSFT) in the input box.

📊 View real-time charts showing indicators like Bollinger Bands, RSI, MACD, and SMA.

📈 Analyze trends and predicted future prices with easy-to-read visuals.

📤 Download prediction results as CSV files for further analysis.

## 🤝 Contributing

🙌 Contributions are always welcome!
If you find a bug or want to improve the app, feel free to open an issue or submit a pull request. Let’s make this project better together! 🚀

## 📝 License

Licensed under the **MIT License** ⚖️  
You are free to use, modify, and share this project under the terms in the `LICENSE` file.