"""
train_transformer.py
Trains a custom multi-horizon Transformer on S&P 500 stock data.

Why a custom Transformer instead of pytorch-forecasting TFT?
-------------------------------------------------------------
pytorch-forecasting's TFT is excellent for single time series with rich
metadata (e.g. forecasting sales for one store). For our use case — 487
independent stocks trained simultaneously — its TimeSeriesDataSet constructor
preprocesses every encoder-decoder combination across all tickers before
training starts, which takes 30+ minutes just for epoch 0.

Our custom Transformer gives us the same architectural benefits that matter
for financial data:
  - Self-attention over the full input window (sees all 756 days equally)
  - Quantile output (p10, p50, p90) via pinball loss — same as TFT
  - Monte Carlo Dropout for uncertainty at inference time
  - Multi-head attention (4 heads learn different patterns simultaneously)
  - Positional encoding so the model knows the order of time steps

We skip TFT's Variable Selection Network and static embeddings because:
  - All 487 stocks use the same 12 features, so variable selection adds
    complexity without clear benefit
  - Static embeddings (ticker identity) would require embedding 487 categories
    and would likely overfit on the training tickers

Training uses the pre-built windows from build_dataset.py (windows_756.npz)
so there is no per-epoch data preprocessing overhead.

Run:
    python train_transformer.py
    (run python build_dataset.py from the project root first)
"""

import os
import sys
import math
import joblib
import logging

import numpy as np
import torch
import mlflow
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import (
    check_feature_array,
    check_train_test_split,
    check_target_distribution,
    log_dataset_summary,
)

HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "6m": 126, "1y": 252}
WINDOW = 756
EPOCHS = 30
BATCH = 512  # large batch — GPU handles this easily on RTX 4060
PATIENT = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Quantile levels for the output — p10, p50, p90
QUANTILES = [0.1, 0.5, 0.9]

print(f"Training on: {DEVICE}")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss) -> bool:
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 756):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class QuantileTransformer(nn.Module):
    """
    Transformer encoder that outputs p10, p50, p90 for each of the 5 horizons.

    Architecture:
      - Linear projection: 12 features -> d_model dimensions
      - Positional encoding: tells the model where each day sits in the window
      - 2 Transformer encoder layers with 4 attention heads
      - Output head: d_model -> n_horizons * n_quantiles (5 * 3 = 15 outputs)

    The output is reshaped to (batch, n_horizons, n_quantiles) so each
    horizon has its own p10, p50, and p90 estimate.

    Trained with pinball (quantile) loss instead of MSE so the model
    learns to produce calibrated uncertainty intervals directly.
    """

    def __init__(
        self,
        feat_size: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        n_horizons: int = 5,
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        self.input_proj = nn.Linear(feat_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, n_horizons * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.output_head(x[:, -1, :])
        return x.view(-1, self.n_horizons, self.n_quantiles)


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: list
) -> torch.Tensor:
    """
    Pinball (quantile) loss for multi-quantile regression.

    For each quantile q:
      loss = q * max(y - y_hat, 0) + (1-q) * max(y_hat - y, 0)

    pred:   (batch, n_horizons, n_quantiles)
    target: (batch, n_horizons)  — the true returns/prices for each horizon

    Returns the mean loss across batch, horizons, and quantiles.
    """
    q = torch.tensor(quantiles, dtype=torch.float32, device=pred.device)
    target = target.unsqueeze(-1).expand_as(pred)
    errors = target - pred
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


if __name__ == "__main__":
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

    logger.info("Loading dataset from %s ...", dataset_path)
    cache = np.load(dataset_path)
    X = cache["X"]  # (n_windows, 756, 12)
    Y_ret = cache["Y_ret"]  # (n_windows, 5)  — log returns per horizon
    Y_px = cache["Y_px"]  # (n_windows, 5)  — absolute prices per horizon
    LC = cache["LC"]  # (n_windows,)     — last close price of each window
    logger.info("Loaded: X=%s  Y_ret=%s", X.shape, Y_ret.shape)

    check_feature_array(X, "X (raw)")
    check_target_distribution(Y_ret[:, 0], "1d returns")

    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    Ret_tr, Ret_te = Y_ret[:split], Y_ret[split:]
    Px_tr, Px_te = Y_px[:split], Y_px[split:]
    LC_tr, LC_te = LC[:split], LC[split:]

    check_train_test_split(X_tr, X_te)

    fs = X_tr.shape[2]
    sc_feat = StandardScaler().fit(X_tr.reshape(-1, fs))
    sc_ret = StandardScaler().fit(Ret_tr)

    X_tr_s = sc_feat.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
    X_te_s = sc_feat.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
    Ret_tr_s = sc_ret.transform(Ret_tr)
    Ret_te_s = sc_ret.transform(Ret_te)

    check_feature_array(X_tr_s, "X_tr (scaled)")
    check_feature_array(X_te_s, "X_te (scaled)")
    log_dataset_summary(X_tr_s, Ret_tr_s, n_tickers=len(X) // 756)

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_s).float(),
        torch.from_numpy(Ret_tr_s).float(),
    )
    te_ds = TensorDataset(
        torch.from_numpy(X_te_s).float(),
        torch.from_numpy(Ret_te_s).float(),
    )
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False, pin_memory=True)

    model = QuantileTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", patience=3, factor=0.5
    )
    stopper = EarlyStopping(PATIENT)
    amp = GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    mlflow.set_experiment("stock-forecasting-transformer")
    with mlflow.start_run(run_name="quantile-transformer"):
        mlflow.log_params(
            {
                "model": "QuantileTransformer",
                "window": WINDOW,
                "horizons": list(HORIZONS.keys()),
                "quantiles": QUANTILES,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "patience": PATIENT,
                "optimizer": "AdamW",
                "lr": LR,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 256,
                "dropout": 0.2,
                "loss": "pinball",
                "split": "chronological 80/20",
            }
        )

        best_val = float("inf")

        for ep in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    pred = model(xb)
                    loss = pinball_loss(pred, yb, QUANTILES)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
                train_loss += loss.item() * xb.size(0)

            tr_loss = train_loss / len(tr_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in te_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(xb)
                    val_loss += pinball_loss(pred, yb, QUANTILES).item() * xb.size(0)

            val_loss /= len(te_ds)

            logger.info(
                "Epoch %d/%d  train=%.4f  val=%.4f  lr=%.2e",
                ep,
                EPOCHS,
                tr_loss,
                val_loss,
                opt.param_groups[0]["lr"],
            )

            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss}, step=ep)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(SAVE_DIR, "transformer_multi_horizon.pth"),
                )
                logger.info("  New best — checkpoint saved.")

            if stopper.step(val_loss):
                logger.info("Early stopping at epoch %d", ep)
                break

        # Evaluate on test set
        model.load_state_dict(
            torch.load(
                os.path.join(SAVE_DIR, "transformer_multi_horizon.pth"),
                map_location=DEVICE,
                weights_only=False,
            )
        )
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in te_dl:
                all_preds.append(model(xb.to(DEVICE)).cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)  # (n_test, 5, 3)
        p50_scaled = preds[:, :, 1]  # median predictions (scaled returns)
        p50_ret = sc_ret.inverse_transform(p50_scaled)
        p50_px = LC_te[:, None] * (1 + p50_ret)

        for idx, name in enumerate(HORIZONS):
            t = Px_te[:, idx]
            p = p50_px[:, idx]
            mae = mean_absolute_error(t, p)
            dir_acc = np.mean(np.sign(p - LC_te) == np.sign(t - LC_te)) * 100
            logger.info("  %s: MAE=%.2f  DirAcc=%.1f%%", name, mae, dir_acc)
            mlflow.log_metric(f"val_mae_{name}", mae)
            mlflow.log_metric(f"val_dir_acc_{name}", dir_acc)

        joblib.dump(sc_feat, os.path.join(SAVE_DIR, "scaler_feat.pkl"))
        joblib.dump(sc_ret, os.path.join(SAVE_DIR, "scaler_ret.pkl"))
        joblib.dump(
            {
                "window": WINDOW,
                "horizons": HORIZONS,
                "model_type": "QuantileTransformer",
            },
            os.path.join(SAVE_DIR, "transformer_meta.pkl"),
        )

        mlflow.log_artifact(os.path.join(SAVE_DIR, "transformer_multi_horizon.pth"))
        logger.info("Training complete.")
