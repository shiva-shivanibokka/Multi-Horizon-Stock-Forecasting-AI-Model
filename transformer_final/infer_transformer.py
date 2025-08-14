import os
import math
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load metadata and scalers
meta     = joblib.load("transformer_meta.pkl")
WINDOW   = meta['window']
HORIZONS = meta['horizons']
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sc_feat  = joblib.load("scaler_feat.pkl")
sc_ret   = joblib.load("scaler_ret.pkl")

# Load saved data
X_tr_s = joblib.load("X_tr.pkl")
Ret_tr = joblib.load("Ret_tr.pkl")
Px_tr  = joblib.load("Px_tr.pkl")
LC_tr  = joblib.load("LC_tr.pkl")

X_te_s = joblib.load("X_te.pkl")
Ret_te = joblib.load("Ret_te.pkl")
Px_te  = joblib.load("Px_te.pkl")
LC_te  = joblib.load("LC_te.pkl")

# Model definitions
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

# Load trained model
model = TimeSeriesTransformer().to(DEVICE)
model.load_state_dict(torch.load("transformer_multi_horizon.pth", map_location=DEVICE))
model.eval()

# Inference
with torch.no_grad():
    Ret_tr_pred = model(torch.from_numpy(X_tr_s).float().to(DEVICE)).cpu().numpy()
    Ret_te_pred = model(torch.from_numpy(X_te_s).float().to(DEVICE)).cpu().numpy()

Ret_tr_pred = sc_ret.inverse_transform(Ret_tr_pred)
Ret_te_pred = sc_ret.inverse_transform(Ret_te_pred)

Px_tr_pred = LC_tr[:, None] * (1 + Ret_tr_pred)
Px_te_pred = LC_te[:, None] * (1 + Ret_te_pred)

# Metric printers
def print_return_metrics(y_true, y_pred, label):
    print(f"\n--- {label} (Return) ---")
    for idx, name in enumerate(HORIZONS):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        mse  = mean_squared_error(t, p)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(t, p)
        sign_acc = np.mean(np.sign(t) == np.sign(p)) * 100
        print(f"{name:>2}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, SignAcc={sign_acc:.2f}%")

def print_price_metrics(y_true, y_pred, label):
    print(f"\n--- {label} (Price) ---")
    for idx, name in enumerate(HORIZONS):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        mse  = mean_squared_error(t, p)
        rmse = math.sqrt(mse)
        mae  = mean_absolute_error(t, p)
        mape = np.mean(np.abs((t - p) / t)) * 100
        r2   = r2_score(t, p)
        acc  = 100 - mape
        print(f"{name:>2}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R²={r2:.4f}, Acc={acc:.2f}%")

# Final output
print_return_metrics(Ret_tr, Ret_tr_pred, "Training")
print_price_metrics(Px_tr, Px_tr_pred, "Training")
print_return_metrics(Ret_te, Ret_te_pred, "Validation")
print_price_metrics(Px_te, Px_te_pred, "Validation")
