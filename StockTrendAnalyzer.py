# Stock Trend Analyzer - Python Script

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------
# User Inputs
# -----------------------
symbol = "AAPL"  # Change to any stock symbol
start_date = "2020-01-01"
end_date = "2024-01-01"

# -----------------------
# Download Stock Data
# -----------------------
data = yf.download(symbol, start=start_date, end=end_date)
print(f"Downloaded {len(data)} rows for {symbol}")

# -----------------------
# Technical Indicators
# -----------------------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data):
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

def compute_bollinger_bands(data, window=20):
    middle_band = data['Close'].rolling(window=window).mean()
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = middle_band + 2 * std_dev
    lower_band = middle_band - 2 * std_dev
    data['BB_Middle'] = middle_band
    data['BB_Upper'] = upper_band
    data['BB_Lower'] = lower_band
    return data

data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = compute_rsi(data)
data = compute_macd(data)
data = compute_bollinger_bands(data)
data['Price_Change'] = data['Close'].pct_change()
data.dropna(inplace=True)

if len(data) < 30:
    raise ValueError("Not enough data to train the model. Try a longer date range.")

# -----------------------
# Train Machine Learning Model
# -----------------------
features = ['MA10', 'MA20', 'RSI', 'MACD', 'Signal_Line', 
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'Price_Change', 'Volume']
X = data[features]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R² Score: {r2_score(y_test, preds):.2f}")

# -----------------------
# Predict Next-Day Price and Trend
# -----------------------
last_row = data[features].iloc[[-1]].values
next_day_price = float(model.predict(last_row))
last_close = float(data['Close'].iloc[-1])

if next_day_price > last_close:
    trend = "Uptrend 📈"
elif next_day_price < last_close:
    trend = "Downtrend 📉"
else:
    trend = "Stable ➡️"

rsi_value = float(data['RSI'].iloc[-1])
if rsi_value > 70:
    rsi_status = "Overbought ⚠️"
elif rsi_value < 30:
    rsi_status = "Oversold 💰"
else:
    rsi_status = "Neutral ✅"

print(f"Last Closing Price: ${last_close:.2f}")
print(f"Predicted Next-Day Price: ${next_day_price:.2f}")
print(f"Trend: {trend}")
print(f"RSI: {rsi_status}")

# -----------------------
# Plot Actual vs Predicted
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close'], label='Actual Close', color='skyblue')
plt.plot(data.index, model.predict(X), label='Predicted Close', color='orange')
plt.title(f"{symbol} Stock: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.show()
