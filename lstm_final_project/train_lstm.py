# train_lstm_multi_horizon.py

import os, sys, math, gc, joblib, yfinance as yf
import pandas as pd, numpy as np, torch
import mlflow
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import (
    check_price_data,
    check_feature_array,
    check_train_test_split,
    check_target_distribution,
    log_dataset_summary,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Hyperparameters
HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252}
MAX_H, WINDOW, IND_WIN = max(HORIZONS.values()), 756, 200
EPOCHS, BATCH, LR = 50, 16, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def compute_technicals(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["MOM_1"] = df["Close"].diff(1)
    df["ROC_14"] = df["Close"].pct_change(14)
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    return df.dropna()


def _download_one(sym):
    try:
        yf_sym = sym.replace(".", "-").upper()
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)
        if df is None or df.empty:
            return sym, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return sym, df if not df.empty else None
    except Exception:
        return sym, None


def download_all(tickers, max_workers=32):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, sym): sym for sym in tickers}
        done = 0
        for future in as_completed(futures):
            sym, df = future.result()
            done += 1
            if df is not None:
                results[sym] = df
            if done % 50 == 0:
                print(f"  Downloaded {done}/{len(tickers)} tickers...")
    print(f"Download complete: {len(results)}/{len(tickers)} tickers succeeded.")
    return results


def build_dataset(tickers):
    X_list, Y_list = [], []
    feats = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "SMA_50",
        "SMA_200",
        "RSI_14",
        "MOM_1",
        "ROC_14",
        "MACD",
    ]
    print(f"Downloading {len(tickers)} tickers in parallel...")
    data = download_all(tickers)
    n_used = 0
    for sym, hist in data.items():
        try:
            hist = check_price_data(hist, sym)
            if hist.empty or len(hist) < WINDOW + MAX_H + IND_WIN:
                continue
            tech = compute_technicals(hist)
            vals = tech[feats].values
            for i in range(vals.shape[0] - (WINDOW + MAX_H) + 1):
                X_list.append(vals[i : i + WINDOW])
                Y_list.append([vals[i + WINDOW + h - 1, 3] for h in HORIZONS.values()])
        except Exception as e:
            print(f"Skip {sym}: {e}")
            continue
    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


class LSTMForecast(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, out_size=5, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "windows_756.npz",
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset cache not found at {dataset_path}.\n"
            "Run  python build_dataset.py  from the project root first."
        )
    print(f"Loading dataset from {dataset_path} ...")
    cache = np.load(dataset_path)
    X = cache["X"]  # (n_windows, 756, 12)
    Y = cache["Y_px"]  # LSTM trains on absolute prices
    print(f"Loaded: X={X.shape}  Y={Y.shape}")

    check_feature_array(X, "X (raw)")
    check_target_distribution(Y, "prices")

    # Chronological split — no shuffle. Random shuffle causes data leakage
    # in financial time series: future windows leak into the training set.
    split = int(0.8 * len(X))
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]

    check_train_test_split(X_tr, X_te)

    bs, seq, fs = X_tr.shape
    feat_scaler = StandardScaler().fit(X_tr.reshape(-1, fs))
    targ_scaler = StandardScaler().fit(Y_tr)
    X_tr_s = feat_scaler.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
    X_te_s = feat_scaler.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
    Y_tr_s, Y_te_s = targ_scaler.transform(Y_tr), targ_scaler.transform(Y_te)

    check_feature_array(X_tr_s, "X_tr (scaled)")
    check_feature_array(X_te_s, "X_te (scaled)")
    log_dataset_summary(X_tr_s, Y_tr_s, n_tickers=len(tickers))

    joblib.dump(feat_scaler, "lstm_scaler_feat.pkl")
    joblib.dump(targ_scaler, "lstm_scaler_targ.pkl")

    train_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(Y_tr_s))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_ds = TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(Y_te_s))
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    model = LSTMForecast(input_size=fs, out_size=len(HORIZONS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    mlflow.set_experiment("stock-forecasting-lstm")
    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(
            {
                "model": "LSTMForecast",
                "window": WINDOW,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "lr": LR,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "split": "chronological 80/20",
            }
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
            train_mse = total_loss / len(train_ds)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += crit(model(xb), yb).item() * xb.size(0)
            val_mse = val_loss / len(val_ds)

            print(
                f"Epoch {epoch:02d}/{EPOCHS}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}"
            )
            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse}, step=epoch)

        torch.save(model.state_dict(), "lstm_multi_horizon.pth")
        joblib.dump({"window": WINDOW, "horizons": HORIZONS}, "lstm_meta.pkl")
        mlflow.log_artifact("lstm_multi_horizon.pth")
        print("Done. Model and scalers saved.")


if __name__ == "__main__":
    main()
