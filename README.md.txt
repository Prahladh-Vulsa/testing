# Stock Trend Analyzer

This project predicts the next-day stock price and trend based on historical stock data using Python.

## Features

- Fetches historical stock data using `yfinance`.
- Computes technical indicators: MA, RSI, MACD, Bollinger Bands.
- Trains a RandomForestRegressor model to predict closing prices.
- Predicts next-day price and market trend.
- Plots actual vs predicted stock prices.

## Requirements

- Python 3.x
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
S