import os
import math
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 1) Forecast horizons (trading days)
HORIZONS = {'1d':1, '1w':5, '1m':21, '6m':126, '1y':252}
MAX_H    = max(HORIZONS.values())
WINDOW   = 756
IND_WIN  = 200
EPOCHS   = 50
BATCH    = 128
PATIENT  = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float('inf')
        self.counter   = 0
    def step(self, loss):
        if self.best - loss > self.min_delta:
            self.best    = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=WINDOW):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        pos = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=DEVICE) * -(math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feat_size=12, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.input_proj  = nn.Linear(feat_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        enc_layer        = nn.TransformerEncoderLayer(
                             d_model, nhead, dim_feedforward,
                             dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor   = nn.Linear(d_model, len(HORIZONS))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :])

def fetch_sp500_tickers():
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
    return df['Symbol'].tolist()

def compute_technicals(df):
    df = df.copy()
    df['SMA_10']  = df['Close'].rolling(10).mean()
    df['SMA_50']  = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain   = gain.rolling(14).mean()
    avg_loss   = loss.rolling(14).mean()
    rs         = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100/(1+rs))
    df['MOM_1']  = df['Close'].diff(1)
    df['ROC_14'] = df['Close'].pct_change(14)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    return df.dropna()

def build_dataset(tickers):
    X_list, Y_ret_list, Y_px_list, LC_list = [], [], [], []
    days_fetch = WINDOW + MAX_H + IND_WIN + 14 + 10
    for sym in tickers:
        yf_sym = sym.replace('.', '-').upper()
        hist = yf.download(yf_sym, period=f"{days_fetch}d", interval='1d', progress=False)
        hist = hist[['Open','High','Low','Close','Volume']].dropna()
        if len(hist) < WINDOW + MAX_H + IND_WIN:
            continue
        tech = compute_technicals(hist)
        arr = tech[['Open','High','Low','Close','Volume','SMA_10','SMA_50','SMA_200','RSI_14','MOM_1','ROC_14','MACD']].values
        for i in range(len(arr) - WINDOW - MAX_H + 1):
            win = arr[i:i+WINDOW]
            base = win[-1,3]
            fut = [arr[i+WINDOW+h-1,3] for h in HORIZONS.values()]
            rets = [(f/base - 1) for f in fut]
            X_list.append(win)
            Y_ret_list.append(rets)
            Y_px_list.append(fut)
            LC_list.append(base)
    return (
        np.stack(X_list),
        np.array(Y_ret_list, dtype=np.float32),
        np.array(Y_px_list, dtype=np.float32),
        np.array(LC_list, dtype=np.float32)
    )

def compute_metrics(y_true, y_pred, is_return=False):
    for idx, name in enumerate(HORIZONS):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        mse  = mean_squared_error(t, p)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(t, p)
        if is_return:
            sign_acc = np.mean(np.sign(t)==np.sign(p)) * 100
            print(f"{name:>3} → MSE:{mse:.4f}, RMSE:{rmse:.4f}, MAE:{mae:.4f}, SignAcc:{sign_acc:.2f}%")
        else:
            mape = np.mean(np.abs((t - p) / t)) * 100
            r2   = r2_score(t, p)
            acc  = 100 - mape
            print(f"{name:>3} → MSE:{mse:.2f}, RMSE:{rmse:.2f}, MAE:{mae:.2f}, MAPE:{mape:.2f}%, R²:{r2:.4f}, Acc:{acc:.2f}%")
    print()

if __name__=='__main__':
    tickers = fetch_sp500_tickers()
    X, Y_ret, Y_px, LC = build_dataset(tickers)
    print(f"Data shapes: X={X.shape}, Ret={Y_ret.shape}, Px={Y_px.shape}")

    idx = np.arange(len(X)); np.random.seed(42); np.random.shuffle(idx)
    split = int(0.8*len(idx))
    tr, te = idx[:split], idx[split:]
    X_tr, X_te = X[tr], X[te]
    Ret_tr, Ret_te = Y_ret[tr], Y_ret[te]
    Px_tr,  Px_te  = Y_px[tr], Y_px[te]
    LC_tr,  LC_te  = LC[tr],   LC[te]

    fs = X_tr.shape[2]
    sc_feat = StandardScaler().fit(X_tr.reshape(-1,fs))
    sc_ret  = StandardScaler().fit(Ret_tr)
    X_tr_s = sc_feat.transform(X_tr.reshape(-1,fs)).reshape(X_tr.shape)
    X_te_s = sc_feat.transform(X_te.reshape(-1,fs)).reshape(X_te.shape)
    Ret_tr_s = sc_ret.transform(Ret_tr)
    Ret_te_s = sc_ret.transform(Ret_te)

    tr_ds = TensorDataset(torch.from_numpy(X_tr_s).float().to(DEVICE), torch.from_numpy(Ret_tr_s).float().to(DEVICE))
    te_ds = TensorDataset(torch.from_numpy(X_te_s).float().to(DEVICE), torch.from_numpy(Ret_te_s).float().to(DEVICE))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH)

    model = TimeSeriesTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,'min',patience=3)
    stop  = EarlyStopping(PATIENT)
    loss_fn = nn.MSELoss()
    scaler_amp = GradScaler()

    for ep in range(1,EPOCHS+1):
        model.train(); train_loss=0
        for xb,yb in tr_dl:
            opt.zero_grad()
            with autocast(enabled=(DEVICE.type=='cuda')):
                out = model(xb); loss = loss_fn(out,yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            scaler_amp.step(opt); scaler_amp.update()
            train_loss += loss.item()*xb.size(0)
        tr_mse = train_loss/len(tr_ds)

        model.eval(); val_loss=0
        with torch.no_grad():
            for xb,yb in te_dl:
                val_loss += loss_fn(model(xb),yb).item()*xb.size(0)
        val_mse = val_loss/len(te_ds)

        print(f"Epoch {ep}/{EPOCHS} Train MSE:{tr_mse:.4f} Val MSE:{val_mse:.4f}")
        sched.step(val_mse)
        if stop.step(val_mse):
            print(f"Early stopping at epoch {ep}")
            break

        model.eval()
    with torch.no_grad():
        Ret_te_p = model(torch.from_numpy(X_te_s).float().to(DEVICE)).cpu().numpy()
    Ret_te_p = sc_ret.inverse_transform(Ret_te_p)
    Px_te_p  = LC_te[:,None] * (1+Ret_te_p)

    print("\n=== Return Metrics (Val) ===")
    compute_metrics(Ret_te, Ret_te_p, is_return=True)
    print("=== Price Metrics (Val)  ===")
    compute_metrics(Px_te, Px_te_p, is_return=False)

    # Inference on training set
    with torch.no_grad():
        Ret_tr_p = model(torch.from_numpy(X_tr_s).float().to(DEVICE)).cpu().numpy()
    Ret_tr_p = sc_ret.inverse_transform(Ret_tr_p)
    Px_tr_p  = LC_tr[:, None] * (1 + Ret_tr_p)

    print("\n=== Return Metrics (Train) ===")
    compute_metrics(Ret_tr, Ret_tr_p, is_return=True)
    print("=== Price Metrics (Train)  ===")
    compute_metrics(Px_tr, Px_tr_p, is_return=False)

    # Save everything needed for inference
    torch.save(model.state_dict(), 'transformer_multi_horizon.pth')
    joblib.dump({'window':WINDOW,'horizons':HORIZONS}, 'transformer_meta.pkl')
    joblib.dump(sc_feat, 'scaler_feat.pkl')
    joblib.dump(sc_ret,  'scaler_ret.pkl')

    # Validation data
    joblib.dump(X_te_s,  'X_te.pkl')
    joblib.dump(Ret_te,  'Ret_te.pkl')
    joblib.dump(Px_te,   'Px_te.pkl')
    joblib.dump(LC_te,   'LC_te.pkl')

    # Training data
    joblib.dump(X_tr_s,  'X_tr.pkl')
    joblib.dump(Ret_tr,  'Ret_tr.pkl')
    joblib.dump(Px_tr,   'Px_tr.pkl')
    joblib.dump(LC_tr,   'LC_tr.pkl')

    print("Training complete.")

