# train_rnn_multi_horizon.py

import math, joblib, yfinance as yf, pandas as pd, numpy as np, torch, gc
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

HORIZONS = {'1d':1, '1w':5, '1m':21, '6m':126, '1y':252}
MAX_H    = max(HORIZONS.values())
WINDOW   = 252
IND_WIN  = 200
EPOCHS   = 50
BATCH    = 16
LR       = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_technicals(df):
    df = df.copy()
    df['SMA_10']  = df['Close'].rolling(10).mean()
    df['SMA_50']  = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100/(1+rs))
    df['MOM_1'] = df['Close'].diff(1)
    df['ROC_14'] = df['Close'].pct_change(14)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    return df.dropna()

def build_dataset(tickers):
    X_list, Y_list = [], []
    days = WINDOW + MAX_H + IND_WIN + 26
    for sym in tickers:
        yf_sym = sym.replace('.', '-').upper()
        hist = (yf.download(yf_sym, period=f"{days}d", interval='1d', progress=False)
                  [['Open','High','Low','Close','Volume']].dropna())
        if len(hist) < WINDOW + MAX_H + IND_WIN:
            continue
        tech = compute_technicals(hist)
        feats = ['Open','High','Low','Close','Volume','SMA_10','SMA_50','SMA_200','RSI_14','MOM_1','ROC_14','MACD']
        arr = tech[feats].values
        for i in range(arr.shape[0] - WINDOW - MAX_H + 1):
            X_list.append(arr[i:i+WINDOW])
            Y_list.append([arr[i+WINDOW+h-1, 3] for h in HORIZONS.values()])
    return np.stack(X_list), np.array(Y_list, dtype=np.float32)

class RNNForecast(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, out_size=5, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          nonlinearity='tanh', dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

def predict_on_cpu(model, X, batch_size):
    model = model.to('cpu')
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float()
            preds.append(model(xb).numpy())
    return np.vstack(preds)

def compute_metrics(name, y_true, y_pred):
    print(f"\n=== {name} set metrics ===")
    for i, key in enumerate(HORIZONS):
        t, p = y_true[:, i], y_pred[:, i]
        mse, mae, mape = mean_squared_error(t, p), mean_absolute_error(t, p), mean_absolute_percentage_error(t, p)
        print(f" {key:>3}: MSE={mse:.2f}, RMSE={math.sqrt(mse):.2f}, MAE={mae:.2f}, MAPE={mape:.2%}, R²={r2_score(t, p):.4f}, Acc={(1 - mape)*100:.2f}%")


def main():
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]['Symbol'].tolist()
    X, Y = build_dataset(tickers)
    idx = np.arange(len(X)); np.random.seed(42); np.random.shuffle(idx)
    tr_idx, te_idx = idx[:int(0.8*len(idx))], idx[int(0.8*len(idx)):]
    X_tr, Y_tr, X_te, Y_te = X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx]
    
    feat_scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    targ_scaler = StandardScaler().fit(Y_tr)
    X_tr_s = feat_scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    X_te_s = feat_scaler.transform(X_te.reshape(-1, X_te.shape[-1])).reshape(X_te.shape)
    Y_tr_s, Y_te_s = targ_scaler.transform(Y_tr), targ_scaler.transform(Y_te)

    joblib.dump(feat_scaler, 'rnn_scaler_feat.pkl')
    joblib.dump(targ_scaler, 'rnn_scaler_targ.pkl')

    train_dl = DataLoader(TensorDataset(torch.tensor(X_tr_s).float(), torch.tensor(Y_tr_s).float()), batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.tensor(X_te_s).float(), torch.tensor(Y_te_s).float()), batch_size=BATCH)

    model = RNNForecast(X.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(1, EPOCHS+1):
        model.train(); total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        print(f"Epoch {ep:02d}/{EPOCHS}: Train MSE={total/len(train_dl.dataset):.4f}, Val MSE={val_loss/len(test_dl.dataset):.4f}")

    joblib.dump({'window': WINDOW, 'horizons': HORIZONS}, 'rnn_meta.pkl')
    torch.save(model.state_dict(), 'rnn_multi_horizon.pth')
    compute_metrics("TRAIN", Y_tr, targ_scaler.inverse_transform(predict_on_cpu(model, X_tr_s, BATCH)))
    compute_metrics("VALID", Y_te, targ_scaler.inverse_transform(predict_on_cpu(model, X_te_s, BATCH)))

if __name__ == '__main__':
    main()
