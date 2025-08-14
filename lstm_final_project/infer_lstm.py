# infer_lstm_multi_horizon.py

import joblib, torch, yfinance as yf
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from train_lstm import LSTMForecast, compute_technicals

# Load model meta info
meta = joblib.load('lstm_meta.pkl')
HORIZONS, WINDOW = meta['horizons'], meta['window']

# Technical indicators used
feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50',
         'SMA_200', 'RSI_14', 'MOM_1', 'ROC_14', 'MACD']

def prepare_input(ticker):
    yf_sym = ticker.replace('.', '-').upper()
    df = yf.download(yf_sym, period="5y", interval='1d', progress=False)
    tech = compute_technicals(df)
    vals = tech[feats].values
    if vals.shape[0] < WINDOW:
        raise ValueError("Not enough data for inference.")
    X = vals[-WINDOW:].astype(np.float32).reshape(1, WINDOW, -1)
    return X

def plot_predictions_bar(predictions_dict, ticker):
    horizons = list(predictions_dict.keys())
    prices = list(predictions_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(horizons, prices, color='orange')
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
    # Prepare input data
    X = prepare_input(ticker)

    # Load scalers
    feat_scaler = joblib.load('lstm_scaler_feat.pkl')
    targ_scaler = joblib.load('lstm_scaler_targ.pkl')

    # Scale features
    X_scaled = feat_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Load model and predict
    model = LSTMForecast(input_size=X.shape[-1], out_size=len(HORIZONS))
    model.load_state_dict(torch.load('lstm_multi_horizon.pth', map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled)).numpy()

    y_pred = targ_scaler.inverse_transform(pred)

    # Print and plot
    predictions_dict = dict(zip(HORIZONS.keys(), y_pred[0]))
    print("Predictions:", predictions_dict)
    plot_predictions_bar(predictions_dict, ticker)

if __name__ == '__main__':
    main("AAPL")  # Change ticker symbol here
