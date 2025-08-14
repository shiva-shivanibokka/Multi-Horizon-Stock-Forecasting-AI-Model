# train_lstm_multi_horizon.py

import math, gc, joblib, yfinance as yf
import pandas as pd, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Hyperparameters
HORIZONS = {'1d':1, '1w':5, '1m':21, '6m':126, '1y':252}
MAX_H, WINDOW, IND_WIN = max(HORIZONS.values()), 756, 200
EPOCHS, BATCH, LR = 50, 16, 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def compute_technicals(df):
    df = df.copy()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(14).mean(), loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['MOM_1'] = df['Close'].diff(1)
    df['ROC_14'] = df['Close'].pct_change(14)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    return df.dropna()

def build_dataset(tickers):
    X_list, Y_list = [], []
    for sym in tickers:
        yf_sym = sym.replace('.', '-').upper()
        print(f"Fetching: {yf_sym}")
        hist = yf.download(yf_sym, period="5y", interval='1d', progress=False)
        if hist.empty or len(hist) < WINDOW + MAX_H + IND_WIN:
            continue
        tech = compute_technicals(hist)
        feats = ['Open','High','Low','Close','Volume','SMA_10','SMA_50',
                 'SMA_200','RSI_14','MOM_1','ROC_14','MACD']
        vals = tech[feats].values
        for i in range(vals.shape[0] - (WINDOW + MAX_H) + 1):
            X_list.append(vals[i : i+WINDOW])
            Y_list.append([vals[i+WINDOW+h-1, 3] for h in HORIZONS.values()])
    X = np.stack(X_list).astype(np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, out_size=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def main():
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
    X, Y = build_dataset(tickers)

    n = len(X)
    idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)
    split = int(0.8 * n)
    X_tr, Y_tr = X[idx[:split]], Y[idx[:split]]
    X_te, Y_te = X[idx[split:]], Y[idx[split:]]

    bs, seq, fs = X_tr.shape
    feat_scaler = StandardScaler().fit(X_tr.reshape(-1, fs))
    targ_scaler = StandardScaler().fit(Y_tr)
    X_tr_s = feat_scaler.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
    X_te_s = feat_scaler.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
    Y_tr_s, Y_te_s = targ_scaler.transform(Y_tr), targ_scaler.transform(Y_te)

    joblib.dump(feat_scaler, 'lstm_scaler_feat.pkl')
    joblib.dump(targ_scaler, 'lstm_scaler_targ.pkl')

    train_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(Y_tr_s))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    model = LSTMForecast(input_size=fs, out_size=len(HORIZONS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch:02d}/{EPOCHS} — Train MSE: {total_loss / len(train_ds):.4f}")

    torch.save(model.state_dict(), 'lstm_multi_horizon.pth')
    joblib.dump({'window':WINDOW,'horizons':HORIZONS}, 'lstm_meta.pkl')
    print("Done. Model and scalers saved.")

if __name__ == '__main__':
    main()
