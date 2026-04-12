import os, sys
import torch
import joblib
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rnn import RNNForecast

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# Horizons must match what the model was trained with — loaded from meta
def _load_meta():
    return joblib.load(os.path.join(SAVE_DIR, "rnn_meta.pkl"))


def prepare_input(ticker, window, n_features, feat_scaler):
    """
    Downloads recent OHLCV, computes the same 36-feature set used during
    training (via build_dataset.py compute_features), scales, and returns
    a (1, window, n_features) float32 tensor ready for the model.
    """
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from build_dataset import compute_features, FEATS, fetch_market_data

    yf_sym = ticker.replace(".", "-").upper()
    df_raw = yf.download(yf_sym, period="3y", interval="1d", progress=False)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df_raw.columns, __import__("pandas").MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    vix, sp21, sp63 = fetch_market_data()
    import pandas as pd

    empty_sec = pd.Series(dtype=float)
    tech = compute_features(df_raw, vix, sp21, sp63, empty_sec)

    vals = tech[FEATS].values.astype(np.float32)
    if len(vals) < window:
        raise ValueError(f"Not enough data: need {window} rows, got {len(vals)}")

    X = vals[-window:].reshape(1, window, -1)
    X_scaled = feat_scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)
    return X_scaled.astype(np.float32)


def plot_predictions_bar(predictions_dict, ticker):
    horizons = list(predictions_dict.keys())
    prices = list(predictions_dict.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(horizons, prices, color="green")
    plt.title(f"{ticker} — RNN Multi-Horizon Forecast")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Predicted Price")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar, value in zip(bars, prices):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.show()


def main(ticker):
    meta = _load_meta()
    horizons = meta["horizons"]  # {"1w":5, "1m":21, "6m":126}
    window = meta["window"]  # 252
    n_features = meta["n_features"]  # 36

    feat_scaler = joblib.load(os.path.join(SAVE_DIR, "rnn_scaler_feat.pkl"))
    targ_scaler = joblib.load(os.path.join(SAVE_DIR, "rnn_scaler_targ.pkl"))

    X_scaled = prepare_input(ticker, window, n_features, feat_scaler)

    # Instantiate with the same dimensions used at training time
    n_out = len(horizons)
    model = RNNForecast(
        input_size=n_features, hidden_size=128, num_layers=2, out_size=n_out
    )
    model.load_state_dict(
        torch.load(
            os.path.join(SAVE_DIR, "rnn_multi_horizon.pth"),
            map_location="cpu",
            weights_only=False,
        )
    )
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled)).numpy()

    y_pred = targ_scaler.inverse_transform(pred)
    predictions_dict = dict(zip(horizons.keys(), y_pred[0]))
    print("Predictions:", predictions_dict)
    plot_predictions_bar(predictions_dict, ticker)


if __name__ == "__main__":
    main("TSLA")
