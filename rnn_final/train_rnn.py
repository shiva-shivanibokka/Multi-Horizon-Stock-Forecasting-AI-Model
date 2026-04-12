# train_rnn_multi_horizon.py

import os, sys, math, joblib, yfinance as yf, pandas as pd, numpy as np, torch, gc
import mlflow
from torch import nn
from tqdm import tqdm

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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

HORIZONS = {"1w": 5, "1m": 21, "6m": 126}  # 1d and 1y removed — too noisy / uncertain
N_HORIZONS = len(HORIZONS)
MAX_H = max(HORIZONS.values())
WINDOW = 252
IND_WIN = 200
EPOCHS = 50
BATCH = 512
LR = 1e-3
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


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
    print(f"Downloading {len(tickers)} tickers in parallel...")
    data = download_all(tickers)
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
    n_used = 0
    for sym, hist in data.items():
        try:
            hist = check_price_data(hist, sym)
            if len(hist) < WINDOW + MAX_H + IND_WIN:
                continue
            tech = compute_technicals(hist)
            arr = tech[feats].values
            for i in range(arr.shape[0] - WINDOW - MAX_H + 1):
                X_list.append(arr[i : i + WINDOW])
                Y_list.append([arr[i + WINDOW + h - 1, 3] for h in HORIZONS.values()])
        except Exception as e:
            print(f"Skip {sym}: {e}")
            continue
    return np.stack(X_list), np.array(Y_list, dtype=np.float32)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class RNNForecast(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, out_size=3, dropout=0.2
    ):  # out_size=3 matches the 3 HORIZONS (1w, 1m, 6m)
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def predict_on_cpu(model, X, batch_size):
    model = model.to("cpu")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            preds.append(model(xb).numpy())
    return np.vstack(preds)


def compute_metrics(name, y_true, y_pred, lc_arr):
    """lc_arr: last close price for each window — used for directional accuracy."""
    print(f"\n=== {name} set metrics ===")
    for i, key in enumerate(HORIZONS):
        t, p = y_true[:, i], y_pred[:, i]
        mse = mean_squared_error(t, p)
        mae = mean_absolute_error(t, p)
        mape = mean_absolute_percentage_error(t, p)
        # Directional accuracy: did the model predict the right direction vs entry price?
        dir_acc = np.mean(np.sign(p - lc_arr) == np.sign(t - lc_arr)) * 100
        print(
            f" {key:>3}: MSE={mse:.2f}, RMSE={math.sqrt(mse):.2f}, MAE={mae:.2f}, "
            f"MAPE={mape:.2%}, R²={r2_score(t, p):.4f}, DirAcc={dir_acc:.1f}%"
        )


def main():
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "windows_252.npz",
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset cache not found at {dataset_path}.\n"
            "Run  python build_dataset.py  from the project root first."
        )
    print(f"Loading dataset from {dataset_path} ...")
    # mmap_mode='r' — X_seq stays on disk, only sliced rows load into RAM
    cache = np.load(dataset_path, mmap_mode="r")
    X = cache["X_seq"]  # (N, 252, 36) — memory-mapped
    Y = np.array(cache["Y"])  # (N, 3)  — small, load fully
    print(f"Loaded: X={X.shape}  Y={Y.shape}")

    # Chronological split — no shuffle to avoid look-ahead bias
    split = int(0.8 * len(X))
    X_tr = np.array(X[:split])  # copy fold into RAM
    X_te = np.array(X[split:])
    Y_tr, Y_te = Y[:split], Y[split:]

    # Last close price for each window = last time-step, Close column (index 3)
    LC_tr = X_tr[:, -1, 3].copy()
    LC_te = X_te[:, -1, 3].copy()

    check_feature_array(X_tr, "X_tr (raw)")
    check_train_test_split(X_tr, X_te)

    returns_1w = (Y_tr[:, 0] - LC_tr) / (LC_tr + 1e-8)
    check_target_distribution(returns_1w, "1w returns (train)")

    # Fit scaler on a sample to avoid OOM on very large datasets
    fit_idx = np.random.choice(len(X_tr), min(20_000, len(X_tr)), replace=False)
    feat_scaler = StandardScaler().fit(X_tr[fit_idx].reshape(-1, X_tr.shape[-1]))
    targ_scaler = StandardScaler().fit(Y_tr)

    X_tr_s = feat_scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    del X_tr
    X_te_s = feat_scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)
    del X_te
    Y_tr_s = targ_scaler.transform(Y_tr)
    Y_te_s = targ_scaler.transform(Y_te)

    check_feature_array(X_tr_s, "X_tr (scaled)")
    log_dataset_summary(X_tr_s, Y_tr_s, n_tickers=len(X) // WINDOW)

    joblib.dump(feat_scaler, "rnn_scaler_feat.pkl")
    joblib.dump(targ_scaler, "rnn_scaler_targ.pkl")

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr_s).float(), torch.tensor(Y_tr_s).float()),
        batch_size=BATCH,
        shuffle=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_te_s).float(), torch.tensor(Y_te_s).float()),
        batch_size=BATCH,
        pin_memory=True,
    )

    # BUG FIX: pass out_size=N_HORIZONS (3), not the stale default of 5
    model = RNNForecast(X.shape[2], out_size=N_HORIZONS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_steps = len(train_dl) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=0.2, anneal_strategy="cos"
    )
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=PATIENCE)
    best_val = float("inf")
    ckpt_path = "rnn_best.pth"

    mlflow.set_experiment("stock-forecasting-rnn")
    with mlflow.start_run(run_name="rnn"):
        mlflow.log_params(
            {
                "model": "RNNForecast",
                "window": WINDOW,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "lr": LR,
                "hidden_size": 128,
                "num_layers": 2,
                "out_size": N_HORIZONS,
                "optimizer": "AdamW",
                "scheduler": "OneCycleLR",
                "split": "chronological 80/20",
            }
        )

        epoch_bar = tqdm(
            range(1, EPOCHS + 1), desc="[rnn]", unit="ep", dynamic_ncols=True
        )
        for ep in epoch_bar:
            model.train()
            total = 0
            bar = tqdm(
                train_dl,
                desc=f"  train ep{ep}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            for xb, yb in bar:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                total += loss.item() * xb.size(0)
                bar.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += loss_fn(model(xb), yb).item() * xb.size(0)

            train_mse = total / len(train_dl.dataset)
            val_mse = val_loss / len(test_dl.dataset)
            epoch_bar.set_postfix(
                train=f"{train_mse:.4f}", val=f"{val_mse:.4f}", best=f"{best_val:.4f}"
            )
            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse}, step=ep)

            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), ckpt_path)

            if stopper.step(val_mse):
                print(f"\nEarly stopping at epoch {ep}")
                break

        # Load best checkpoint
        model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        )

        n_feat = X.shape[2]
        joblib.dump(
            {"window": WINDOW, "horizons": HORIZONS, "n_features": n_feat},
            "rnn_meta.pkl",
        )
        torch.save(model.state_dict(), "rnn_multi_horizon.pth")
        mlflow.log_artifact("rnn_multi_horizon.pth")

        compute_metrics(
            "TRAIN",
            Y_tr,
            targ_scaler.inverse_transform(predict_on_cpu(model, X_tr_s, BATCH)),
            LC_tr,
        )
        compute_metrics(
            "VALID",
            Y_te,
            targ_scaler.inverse_transform(predict_on_cpu(model, X_te_s, BATCH)),
            LC_te,
        )


if __name__ == "__main__":
    main()
