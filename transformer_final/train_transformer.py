import os
import sys
import math
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import (
    check_price_data, check_feature_array,
    check_train_test_split, check_target_distribution,
    log_dataset_summary,
)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feat_size=12,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor = nn.Linear(d_model, len(HORIZONS))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])


def fetch_sp500_tickers():
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "META",
        "TSLA",
        "BRK-B",
        "JPM",
        "UNH",
        "JNJ",
        "V",
        "XOM",
        "PG",
        "MA",
        "HD",
        "CVX",
        "MRK",
        "ABBV",
        "PEP",
        "KO",
        "AVGO",
        "COST",
        "WMT",
        "BAC",
        "MCD",
        "CRM",
        "ACN",
        "LLY",
        "TMO",
        "CSCO",
        "ABT",
        "DHR",
        "NEE",
        "TXN",
        "NKE",
        "PM",
        "ORCL",
        "QCOM",
        "AMGN",
        "IBM",
        "INTC",
        "HON",
        "CAT",
        "GS",
        "MS",
        "BLK",
        "SPGI",
        "RTX",
        "AXP",
        "AMD",
        "NFLX",
        "PYPL",
        "ADBE",
        "NOW",
        "SNOW",
        "UBER",
        "SHOP",
        "SQ",
        "PLTR",
    ]


def compute_technicals(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["MOM_1"] = df["Close"].diff(1)
    df["ROC_14"] = df["Close"].pct_change(14)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    return df.dropna()


def _download_one(sym):
    try:
        yf_sym = sym.replace(".", "-").upper()
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)
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
    X_list, Y_ret_list, Y_px_list, LC_list = [], [], [], []
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
            if len(hist) < WINDOW + MAX_H + IND_WIN:
                continue
            tech = compute_technicals(hist)
            arr = tech[feats].values
            for i in range(len(arr) - WINDOW - MAX_H + 1):
                win = arr[i : i + WINDOW]
                base = win[-1, 3]
                fut = [arr[i + WINDOW + h - 1, 3] for h in HORIZONS.values()]
                rets = [(f / base - 1) for f in fut]
                X_list.append(win)
                Y_ret_list.append(rets)
                Y_px_list.append(fut)
                LC_list.append(base)
            n_used += 1
        except Exception as e:
            print(f"Skip {sym}: {e}")
            continue
    return (
        np.stack(X_list),
        np.array(Y_ret_list, dtype=np.float32),
        np.array(Y_px_list, dtype=np.float32),
        np.array(LC_list, dtype=np.float32),
    )


def compute_metrics(y_true, y_pred, is_return=False):
    for idx, name in enumerate(HORIZONS):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        mse = mean_squared_error(t, p)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(t, p)
        if is_return:
            sign_acc = np.mean(np.sign(t) == np.sign(p)) * 100
            print(
                f"{name:>3} → MSE:{mse:.4f}, RMSE:{rmse:.4f}, MAE:{mae:.4f}, SignAcc:{sign_acc:.2f}%"
            )
        else:
            mape = np.mean(np.abs((t - p) / t)) * 100
            r2 = r2_score(t, p)
            # Directional accuracy: did we predict the right price direction?
            # (100 - MAPE is meaningless for financial forecasting)
            dir_acc = np.mean(np.sign(p - t[0]) == np.sign(t - t[0])) * 100
            print(
                f"{name:>3} → MSE:{mse:.2f}, RMSE:{rmse:.2f}, MAE:{mae:.2f}, MAPE:{mape:.2f}%, R²:{r2:.4f}, DirAcc:{dir_acc:.1f}%"
            )
    print()


if __name__ == "__main__":
    tickers = fetch_sp500_tickers()
    X, Y_ret, Y_px, LC = build_dataset(tickers)
    print(f"Data shapes: X={X.shape}, Ret={Y_ret.shape}, Px={Y_px.shape}")

    # Check for NaN/Inf in the raw feature array before scaling
    check_feature_array(X, "X (raw)")

    # Check target distribution for directional bias
    check_target_distribution(Y_ret, "returns")

    # Chronological split — past trains, future tests.
    # Random shuffle would leak future windows into the training set.
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    Ret_tr, Ret_te = Y_ret[:split], Y_ret[split:]
    Px_tr, Px_te = Y_px[:split], Y_px[split:]
    LC_tr, LC_te = LC[:split], LC[split:]

    # Verify the split is not degenerate
    check_train_test_split(X_tr, X_te)

    fs = X_tr.shape[2]
    sc_feat = StandardScaler().fit(X_tr.reshape(-1, fs))
    sc_ret = StandardScaler().fit(Ret_tr)
    X_tr_s = sc_feat.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
    X_te_s = sc_feat.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
    Ret_tr_s = sc_ret.transform(Ret_tr)
    Ret_te_s = sc_ret.transform(Ret_te)

    # Check for NaN/Inf after scaling (can happen if a feature has zero variance)
    check_feature_array(X_tr_s, "X_tr (scaled)")
    check_feature_array(X_te_s, "X_te (scaled)")

    log_dataset_summary(X_tr_s, Ret_tr_s, n_tickers=len(tickers))

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_s).float().to(DEVICE),
        torch.from_numpy(Ret_tr_s).float().to(DEVICE),
    )
    te_ds = TensorDataset(
        torch.from_numpy(X_te_s).float().to(DEVICE),
        torch.from_numpy(Ret_te_s).float().to(DEVICE),
    )
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH)

    model = TimeSeriesTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3)
    stop = EarlyStopping(PATIENT)
    loss_fn = nn.MSELoss()
    scaler_amp = GradScaler()

    mlflow.set_experiment("stock-forecasting-transformer")
    with mlflow.start_run(run_name="transformer"):
        mlflow.log_params(
            {
                "model": "TimeSeriesTransformer",
                "window": WINDOW,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "patience": PATIENT,
                "optimizer": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dropout": 0.2,
                "split": "chronological 80/20",
            }
        )

        for ep in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for xb, yb in tr_dl:
                opt.zero_grad()
                with autocast(enabled=(DEVICE.type == "cuda")):
                    out = model(xb)
                    loss = loss_fn(out, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler_amp.step(opt)
                scaler_amp.update()
                train_loss += loss.item() * xb.size(0)
            tr_mse = train_loss / len(tr_ds)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in te_dl:
                    val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
            val_mse = val_loss / len(te_ds)

            print(f"Epoch {ep}/{EPOCHS}  train_mse={tr_mse:.4f}  val_mse={val_mse:.4f}")
            mlflow.log_metrics({"train_mse": tr_mse, "val_mse": val_mse}, step=ep)
            sched.step(val_mse)
            if stop.step(val_mse):
                print(f"Early stopping at epoch {ep}")
                break

        model.eval()
        with torch.no_grad():
            Ret_te_p = model(torch.from_numpy(X_te_s).float().to(DEVICE)).cpu().numpy()
        Ret_te_p = sc_ret.inverse_transform(Ret_te_p)
        Px_te_p = LC_te[:, None] * (1 + Ret_te_p)

        print("\nValidation metrics:")
        compute_metrics(Ret_te, Ret_te_p, is_return=True)
        compute_metrics(Px_te, Px_te_p, is_return=False)

        # Log final val MAE for each horizon so runs are comparable in the MLflow UI
        for idx, name in enumerate(HORIZONS):
            mae = mean_absolute_error(Px_te[:, idx], Px_te_p[:, idx])
            mlflow.log_metric(f"val_mae_{name}", mae)

        torch.save(model.state_dict(), "transformer_multi_horizon.pth")
        joblib.dump({"window": WINDOW, "horizons": HORIZONS}, "transformer_meta.pkl")
        joblib.dump(sc_feat, "scaler_feat.pkl")
        joblib.dump(sc_ret, "scaler_ret.pkl")
        joblib.dump(X_te_s, "X_te.pkl")
        joblib.dump(Ret_te, "Ret_te.pkl")
        joblib.dump(Px_te, "Px_te.pkl")
        joblib.dump(LC_te, "LC_te.pkl")
        joblib.dump(X_tr_s, "X_tr.pkl")
        joblib.dump(Ret_tr, "Ret_tr.pkl")
        joblib.dump(Px_tr, "Px_tr.pkl")
        joblib.dump(LC_tr, "LC_tr.pkl")

        mlflow.log_artifact("transformer_multi_horizon.pth")
        print("Training complete. Artifacts saved.")

    print("Training complete.")
