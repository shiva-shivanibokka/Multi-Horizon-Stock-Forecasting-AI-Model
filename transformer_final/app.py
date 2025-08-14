# app.py
import os
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64

from torch import nn

# ——— Load model class ———
from train_transformer import TimeSeriesTransformer

# ——— Constants ———
ARTIFACT_DIR = os.path.dirname(__file__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(ARTIFACT_DIR, "transformer_multi_horizon.pth")
META_PATH = os.path.join(ARTIFACT_DIR, "transformer_meta.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_feat.pkl")

# ——— Flask setup ———
app = Flask(__name__)
CORS(app)

# ——— Load artifacts ———
model = TimeSeriesTransformer().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

meta = joblib.load(META_PATH)
WINDOW = meta["window"]
HORIZONS = meta["horizons"]
scaler_feat = joblib.load(SCALER_PATH)

# ——— Feature engineering ———
def compute_technicals(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['MOM_1'] = df['Close'].diff(1)
    df['ROC_14'] = df['Close'].pct_change(14)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df.dropna()

# ——— Inference Route ———
@app.route("/api/predict/<ticker>", methods=["GET"])
def predict(ticker):
    try:
        yf_sym = ticker.upper().replace(".", "-")
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)

        if df.shape[0] < WINDOW + 1:
            return jsonify({"error": "Not enough data for prediction."})

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = compute_technicals(df)
        df = df.dropna()

        if df.shape[0] < WINDOW:
            return jsonify({"error": "Insufficient technicals for inference."})

        features = df[["Open", "High", "Low", "Close", "Volume", "SMA_10", "SMA_50",
                       "SMA_200", "RSI_14", "MOM_1", "ROC_14", "MACD"]].tail(WINDOW).values

        features_scaled = scaler_feat.transform(features)
        x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(x).cpu().numpy().flatten()

        close_price = df["Close"].iloc[-1]
        preds = {k: round(close_price * (1 + output[i]), 2) for i, k in enumerate(HORIZONS)}

        # ——— Plot last 1y price, SMA50, SMA200 ———
        df_plot = df.tail(252)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_plot.index, df_plot["Close"], label="Close Price")
        ax.plot(df_plot.index, df_plot["SMA_50"], label="SMA 50")
        ax.plot(df_plot.index, df_plot["SMA_200"], label="SMA 200")
        ax.set_title(f"{ticker.upper()} - Past 1Y Price Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return jsonify({
    "predicted_prices": preds,
    "last_close_price": round(close_price, 2),
    "chart_base64": image_base64,
    "price_data": {
        "close": df_plot["Close"].tolist(),
        "sma_50": df_plot["SMA_50"].tolist(),
        "sma_200": df_plot["SMA_200"].tolist(),
        "dates": [d.strftime('%Y-%m-%d') for d in df_plot.index]
    }
})

        

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
