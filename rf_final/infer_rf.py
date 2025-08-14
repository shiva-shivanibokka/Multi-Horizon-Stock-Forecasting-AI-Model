# infer_rf_multi_horizon.py

import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HORIZONS = {
    '1d':  1,
    '1w':  5,
    '1m':  21,
    '6m': 126,
    '1y': 252,
}
WINDOW = 252

def fetch_stock_data(ticker):
    yf_sym = ticker.replace('.', '-').upper()
    df = yf.download(yf_sym, period='3y', interval='1d', progress=False)
    return df[['Open','High','Low','Close','Volume']].dropna()

def prepare_features(df):
    vals = df.sort_index().to_numpy()
    if len(vals) < WINDOW:
        raise ValueError("Not enough data")
    return vals[-WINDOW:].flatten().reshape(1, -1)

def plot_predictions_bar(pred_dict, ticker):
    horizons = list(pred_dict.keys())
    prices = list(pred_dict.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(horizons, prices, color='mediumpurple')
    plt.title(f"{ticker} — Multi-Horizon Forecast")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Predicted Price")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, value in zip(bars, prices):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{value:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def main(ticker):
    model = joblib.load('rf_multi_horizon.pkl')
    features = joblib.load('feature_list_multi.pkl')

    df = fetch_stock_data(ticker)
    X = prepare_features(df)
    X_df = pd.DataFrame(X, columns=features)

    y_pred = model.predict(X_df)[0]
    pred_dict = dict(zip(HORIZONS.keys(), y_pred))
    print("Predictions:", pred_dict)
    plot_predictions_bar(pred_dict, ticker)

if __name__ == '__main__':
    main("AAPL")  # Change ticker symbol here
