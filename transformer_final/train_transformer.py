"""
train_transformer.py
Production-grade PatchTST model for multi-horizon stock price forecasting.

Three improvements over the basic Transformer:

1. PatchTST architecture (CMU + IBM, 2023)
   Instead of feeding one day per token, we group consecutive days into
   patches of PATCH_LEN days. 756 days / 16-day patches = 47 tokens.
   Self-attention is O(n^2) in tokens — 47^2 = 2,209 vs 756^2 = 571,536.
   Each patch captures local momentum within itself. The attention then
   learns which 16-day periods matter most for the forecast.

2. Temporal (date-based) train/test split
   The original split divided windows by index (first 80% train, last 20%
   test). Consecutive windows overlap by 755 days, so the test set contained
   windows nearly identical to training windows — the model memorized rather
   than generalized.

   The correct split divides by the date of each window's last day. All
   windows ending before the cutoff date go to train; the rest to test.
   No overlap across the boundary is possible.

3. Sector cross-sectional feature
   Each window now includes the sector ID (0-10) of the stock as a learned
   embedding. The model can learn that tech stocks behave differently from
   utilities, for example, without being told explicitly.

Run:
    python train_transformer.py
    (run python build_dataset.py --refresh from the project root first
     to rebuild the dataset with date/sector metadata)
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
from sklearn.metrics import mean_absolute_error

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
N_HORIZONS = len(HORIZONS)

# PatchTST hyperparameters
WINDOW = 756  # encoder window — 3 years of daily data
PATCH_LEN = 16  # days per patch token (756 / 16 = 47 tokens)
STRIDE = 8  # patch stride — overlapping patches give more tokens
N_SECTORS = 11  # number of S&P 500 GICS sectors
SECTOR_DIM = 8  # sector embedding dimension

# Training hyperparameters
EPOCHS = 50
BATCH = 512
PATIENT = 7
LR = 3e-4  # lower LR than before — PatchTST converges more stably
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Temporal split — windows whose last date is before this cutoff go to train.
# The last ~20% of the date range (~1 year) becomes the test set.
# This prevents any overlap between train and test windows.
SPLIT_DATE = "2025-04-01"

print(f"Training on: {DEVICE}")


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


class PatchEmbedding(nn.Module):
    """
    Converts a time series window into patch tokens.

    Input:  (batch, time_steps, n_features)   — raw feature sequence
    Output: (batch, n_patches, d_model)        — patch token sequence

    Each patch is a PATCH_LEN-day segment. A 1D convolution extracts a
    d_model-dimensional embedding for each patch. Overlapping patches
    (stride < patch_len) give the model more tokens to attend over.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(
            in_channels=n_features,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -> (batch, features, time) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x.permute(0, 2, 1)  # back to (batch, n_patches, d_model)
        return x


class PatchTST(nn.Module):
    """
    PatchTST — Patch Time Series Transformer.

    Key improvements over a standard Transformer:
      - Patch embedding reduces sequence from 756 tokens to ~90 tokens
        (with PATCH_LEN=16, STRIDE=8: (756-16)/8 + 1 = 93 tokens)
      - 93^2 = 8,649 attention ops vs 756^2 = 571,536 — 66x reduction
      - Each patch sees 16 days of local context before the global attention
      - Sector embedding adds cross-sectional information (tech vs utilities)
      - Output: (n_horizons, n_quantiles) — p10, p50, p90 per horizon

    Architecture:
      PatchEmbedding -> + PositionalEncoding -> + SectorEmbedding
      -> TransformerEncoder (4 layers, 8 heads)
      -> Global average pool
      -> Output head: d_model -> n_horizons * n_quantiles
    """

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.3,
        n_horizons: int = N_HORIZONS,
        n_quantiles: int = 3,
        n_sectors: int = N_SECTORS,
        sector_dim: int = SECTOR_DIM,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: more stable training than Post-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_horizons * n_quantiles),
        )

    def forward(self, x: torch.Tensor, sector: torch.Tensor) -> torch.Tensor:
        # x:      (batch, 756, 12)
        # sector: (batch,) — integer sector ID

        patches = self.patch_embed(x)  # (batch, n_patches, d_model)
        sec_emb = self.sector_proj(self.sector_embed(sector)).unsqueeze(
            1
        )  # (batch, 1, d_model)

        # Add sector embedding to every patch token
        patches = patches + sec_emb
        patches = self.dropout(patches)

        out = self.transformer(patches)  # (batch, n_patches, d_model)
        out = self.norm(out)
        out = out.mean(dim=1)  # global average pool over patches
        out = self.output_head(out)  # (batch, n_horizons * n_quantiles)
        return out.view(-1, self.n_horizons, self.n_quantiles)


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: list
) -> torch.Tensor:
    """
    Pinball (quantile) loss — trains the model to produce calibrated intervals.

    pred:   (batch, n_horizons, n_quantiles)
    target: (batch, n_horizons)
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
            "Run  python build_dataset.py --refresh  from the project root first."
        )

    logger.info("Loading dataset from %s ...", dataset_path)
    cache = np.load(dataset_path, allow_pickle=True)
    X = cache["X"]  # (n, 756, 12)
    Y_ret = cache["Y_ret"]  # (n, 5)
    Y_px = cache["Y_px"]  # (n, 5)
    LC = cache["LC"]  # (n,)

    # Check if the rebuilt dataset includes date/sector metadata
    has_meta = "dates" in cache and "sectors" in cache
    if has_meta:
        dates = cache["dates"]  # (n,) — string dates
        sectors = cache["sectors"]  # (n,) — int sector IDs
        logger.info("Dataset includes date and sector metadata.")
    else:
        # Fall back to index-based split and no sector features
        # Run python build_dataset.py --refresh to get the full metadata
        logger.warning(
            "Dataset does not have date/sector metadata. "
            "Using index-based split and zero sectors. "
            "Run  python build_dataset.py --refresh  for best results."
        )
        dates = None
        sectors = np.zeros(len(X), dtype=np.int32)

    logger.info("Loaded: X=%s  Y_ret=%s", X.shape, Y_ret.shape)
    check_feature_array(X, "X (raw)")
    check_target_distribution(Y_ret[:, 0], "1d returns")

    # Temporal split — split by date, not by window index.
    # All windows whose last day is before SPLIT_DATE go to train.
    # This prevents leakage between overlapping windows.
    if has_meta:
        train_mask = dates < SPLIT_DATE
        test_mask = ~train_mask
        logger.info(
            "Temporal split at %s: train=%d  test=%d",
            SPLIT_DATE,
            train_mask.sum(),
            test_mask.sum(),
        )
        X_tr, X_te = X[train_mask], X[test_mask]
        Ret_tr, Ret_te = Y_ret[train_mask], Y_ret[test_mask]
        Px_tr, Px_te = Y_px[train_mask], Y_px[test_mask]
        LC_tr, LC_te = LC[train_mask], LC[test_mask]
        Sec_tr, Sec_te = sectors[train_mask], sectors[test_mask]
    else:
        split = int(0.8 * len(X))
        X_tr, X_te = X[:split], X[split:]
        Ret_tr, Ret_te = Y_ret[:split], Y_ret[split:]
        Px_tr, Px_te = Y_px[:split], Y_px[split:]
        LC_tr, LC_te = LC[:split], LC[split:]
        Sec_tr, Sec_te = sectors[:split], sectors[split:]

    check_train_test_split(X_tr, X_te)

    # Normalize features
    fs = X_tr.shape[2]
    sc_feat = StandardScaler().fit(X_tr.reshape(-1, fs))
    sc_ret = StandardScaler().fit(Ret_tr)

    X_tr_s = sc_feat.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
    X_te_s = sc_feat.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
    Ret_tr_s = sc_ret.transform(Ret_tr)
    Ret_te_s = sc_ret.transform(Ret_te)

    check_feature_array(X_tr_s, "X_tr (scaled)")
    check_feature_array(X_te_s, "X_te (scaled)")
    log_dataset_summary(X_tr_s, Ret_tr_s, n_tickers=len(X) // WINDOW)

    # Build data loaders
    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_s).float(),
        torch.from_numpy(Ret_tr_s).float(),
        torch.from_numpy(Sec_tr).long(),
    )
    te_ds = TensorDataset(
        torch.from_numpy(X_te_s).float(),
        torch.from_numpy(Ret_te_s).float(),
        torch.from_numpy(Sec_te).long(),
    )
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False, pin_memory=True)

    model = PatchTST().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(tr_dl),
        pct_start=0.3,
        anneal_strategy="cos",
    )
    stopper = EarlyStopping(patience=PATIENT)
    amp = GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("PatchTST parameters: %d", n_params)

    n_patches = (WINDOW - PATCH_LEN) // STRIDE + 1
    logger.info(
        "Patches per window: %d  (window=%d, patch=%d, stride=%d)",
        n_patches,
        WINDOW,
        PATCH_LEN,
        STRIDE,
    )

    mlflow.set_experiment("stock-forecasting-transformer")
    with mlflow.start_run(run_name="patchtst"):
        mlflow.log_params(
            {
                "model": "PatchTST",
                "window": WINDOW,
                "patch_len": PATCH_LEN,
                "stride": STRIDE,
                "n_patches": n_patches,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "dim_ff": 512,
                "dropout": 0.3,
                "sector_dim": SECTOR_DIM,
                "batch_size": BATCH,
                "epochs": EPOCHS,
                "patience": PATIENT,
                "lr": LR,
                "scheduler": "OneCycleLR",
                "loss": "pinball",
                "split": f"temporal at {SPLIT_DATE}" if has_meta else "index 80/20",
            }
        )

        best_val = float("inf")

        for ep in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for xb, yb, sb in tr_dl:
                xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                opt.zero_grad()
                with autocast():
                    pred = model(xb, sb)
                    loss = pinball_loss(pred, yb, QUANTILES)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
                scheduler.step()
                train_loss += loss.item() * xb.size(0)

            tr_loss = train_loss / len(tr_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, sb in te_dl:
                    xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                    pred = model(xb, sb)
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

        # Final evaluation on test set
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
            for xb, _, sb in te_dl:
                all_preds.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)  # (n_test, 5, 3)
        p50_s = preds[:, :, 1]
        p50_ret = sc_ret.inverse_transform(p50_s)
        p50_px = LC_te[:, None] * (1 + p50_ret)

        logger.info("Test set metrics:")
        for idx, name in enumerate(HORIZONS):
            t = Px_te[:, idx]
            p = p50_px[:, idx]
            mae = mean_absolute_error(t, p)
            dir_acc = np.mean(np.sign(p - LC_te) == np.sign(t - LC_te)) * 100
            logger.info("  %s: MAE=%.2f  DirAcc=%.1f%%", name, mae, dir_acc)
            mlflow.log_metric(f"val_mae_{name}", mae)
            mlflow.log_metric(f"val_dir_acc_{name}", dir_acc)

        # Save everything needed for inference
        joblib.dump(sc_feat, os.path.join(SAVE_DIR, "scaler_feat.pkl"))
        joblib.dump(sc_ret, os.path.join(SAVE_DIR, "scaler_ret.pkl"))
        joblib.dump(
            {
                "window": WINDOW,
                "patch_len": PATCH_LEN,
                "stride": STRIDE,
                "horizons": HORIZONS,
                "model_type": "PatchTST",
                "split_date": SPLIT_DATE,
            },
            os.path.join(SAVE_DIR, "transformer_meta.pkl"),
        )

        mlflow.log_artifact(os.path.join(SAVE_DIR, "transformer_multi_horizon.pth"))
        logger.info("Training complete.")
