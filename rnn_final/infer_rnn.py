import torch
import joblib
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from train_rnn import compute_technicals, RNNForecast

# Constants
HORIZONS = {'1d': 1, '1w': 5, '1m': 21, '6m': 126, '1y': 252}
WINDOW = 252

# Technical indicators
feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50',
         'SMA_200', 'RSI_14', 'MOM_1', 'ROC_14', 'MACD']

def prepare_input(ticker):
    yf_sym = ticker.replace('.', '-').upper()
    df = yf.download(yf_sym, period="3y", interval='1d', progress=False)
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
    bars = plt.bar(horizons, prices, color='green')
    plt.title(f"{ticker} — RNN Multi-Horizon Forecast")
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
    feat_scaler = joblib.load('rnn_scaler_feat.pkl')
    targ_scaler = joblib.load('rnn_scaler_targ.pkl')

    # Scale features
    X_scaled = feat_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Load model and predict
    model = RNNForecast(input_size=X.shape[-1], hidden_size=128, num_layers=2, out_size=len(HORIZONS))
    model.load_state_dict(torch.load('rnn_multi_horizon.pth', map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled)).numpy()

    y_pred = targ_scaler.inverse_transform(pred)

    # Print and plot
    predictions_dict = dict(zip(HORIZONS.keys(), y_pred[0]))
    print("Predictions:", predictions_dict)
    plot_predictions_bar(predictions_dict, ticker)

if __name__ == '__main__':
    main("TSLA")  # Change ticker symbol here
