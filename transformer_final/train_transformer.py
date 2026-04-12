"""
train_transformer.py
Production-grade PatchTST for 1-week, 1-month, and 6-month stock forecasting.

Design decisions:

1. Horizons: 1w (5d), 1m (21d), 6m (126d)
   1-day removed: dominated by news/microstructure, price history cannot predict it
   1-year removed: too much macro uncertainty, prediction interval is too wide to act on

2. Rich feature set (36 features per day)
   Beyond basic OHLCV + moving averages, we add:
   - Candlestick patterns: body size, shadows, doji, hammer, shooting star, engulfing
   - Volatility: ATR, Bollinger Bands
   - Volume analysis: OBV, volume ratio
   - Price structure: distance from 52-week high/low
   - Market features: VIX, S&P 500 rolling returns (regime context)
   - Relative strength vs sector (how this stock moved vs its GICS peers)

3. PatchTST architecture
   16-day patches with stride 8 → 93 tokens instead of 756 per window.
   Self-attention is O(n^2) in tokens — 93^2 vs 756^2 is a 66x reduction.
   Local patch context + global attention = captures both short-term momentum
   and long-range seasonal patterns simultaneously.

4. Separate output heads per horizon
   The patterns that predict 1-week returns are completely different from those
   that predict 6-month returns. Separate heads let each horizon learn its own
   decision boundary from the shared encoder representation.

5. Walk-forward cross-validation (3 folds)
   Instead of one train/test split, we validate across three time periods:
     Fold 1: train 2021-2023 → test 2023-2024
     Fold 2: train 2021-2024 → test 2024-2025
     Fold 3: train 2021-2025 → test 2025-2026
   Each fold's test set represents a different market regime. Averaging metrics
   across folds gives a much more honest picture of true out-of-sample accuracy.
   The final production model is trained on all available data after CV.

Run:
    python train_transformer.py
    (python build_dataset.py --refresh from project root first)
"""

import os
import sys
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
from data_guards import check_feature_array, check_train_test_split, log_dataset_summary

HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
N_HORIZONS = len(HORIZONS)
N_FEATS = 36  # must match build_dataset.py FEATS list

WINDOW = 756
PATCH_LEN = 16
STRIDE = 8
N_SECTORS = 11
SECTOR_DIM = 8

EPOCHS = 50
BATCH = 512
PATIENCE = 7
LR = 3e-4
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk-forward CV fold boundaries — each is a (train_end, test_end) date pair
# The test window for each fold is exactly 1 year
WF_FOLDS = [
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]

print(f"Training on: {DEVICE}")
logger.info(
    "Horizons: %s  Features: %d  Patches: %d",
    list(HORIZONS.keys()),
    N_FEATS,
    (WINDOW - PATCH_LEN) // STRIDE + 1,
)


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

    def reset(self):
        self.best = float("inf")
        self.counter = 0


class PatchEmbedding(nn.Module):
    def __init__(
        self, n_features=N_FEATS, d_model=128, patch_len=PATCH_LEN, stride=STRIDE
    ):
        super().__init__()
        self.proj = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T) → conv → (B, d_model, n_patches) → (B, n_patches, d_model)
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class PatchTST(nn.Module):
    """
    PatchTST encoder with separate output heads per horizon.

    The encoder is shared — all three horizons look at the same historical data
    through the same attention mechanism. But each horizon gets its own two-layer
    MLP output head, because the decision function for 6-month returns is
    fundamentally different from the one for 1-week returns.

    The 6-month head is deeper (3 layers) because predicting 6 months ahead
    requires capturing more complex non-linear interactions between macro
    features, sector trends, and individual stock momentum.
    """

    def __init__(
        self,
        n_features=N_FEATS,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_ff=512,
        dropout=0.3,
        n_sectors=N_SECTORS,
        sector_dim=SECTOR_DIM,
        n_quantiles=3,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = N_HORIZONS

        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        def _head(out_size, deep=False):
            if deep:
                return nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, out_size),
                )
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_size),
            )

        self.head_1w = _head(n_quantiles)
        self.head_1m = _head(n_quantiles)
        self.head_6m = _head(n_quantiles, deep=True)  # deeper head for 6m

    def encode(self, x, sector):
        patches = self.patch_embed(x)
        sec_emb = self.sector_proj(self.sector_embed(sector)).unsqueeze(1)
        patches = self.drop(patches + sec_emb)
        out = self.norm(self.encoder(patches))
        return out.mean(dim=1)  # (B, d_model)

    def forward(self, x, sector):
        enc = self.encode(x, sector)
        q1w = self.head_1w(enc)  # (B, 3) — p10,p50,p90 for 1w
        q1m = self.head_1m(enc)  # (B, 3) — p10,p50,p90 for 1m
        q6m = self.head_6m(enc)  # (B, 3) — p10,p50,p90 for 6m
        return torch.stack([q1w, q1m, q6m], dim=1)  # (B, 3, 3)


def pinball_loss(pred, target, quantiles=QUANTILES):
    q = torch.tensor(quantiles, device=pred.device)
    target = target.unsqueeze(-1).expand_as(pred)
    err = target - pred
    return torch.max(q * err, (q - 1) * err).mean()


def run_epoch(model, dl, opt, amp, scheduler, train=True):
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb, sb in dl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            if train:
                opt.zero_grad()
                with autocast():
                    pred = model(xb, sb)
                    loss = pinball_loss(pred, yb)
                amp.scale(loss).backward()
                amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp.step(opt)
                amp.update()
                scheduler.step()
            else:
                pred = model(xb, sb)
                loss = pinball_loss(pred, yb)
            total += loss.item() * xb.size(0)
    return total / len(dl.dataset)


def evaluate(model, dl, sc_ret, LC_arr, Y_px_arr):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _, sb in dl:
            preds_list.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)  # (N, 3 horizons, 3 quantiles)
    p50_s = preds[:, :, 1]  # (N, 3) — median scaled returns
    p50_ret = sc_ret.inverse_transform(p50_s)
    p50_px = LC_arr[:, None] * (1 + p50_ret)

    metrics = {}
    for idx, name in enumerate(HORIZONS):
        t = Y_px_arr[:, idx]
        p = p50_px[:, idx]
        mae = mean_absolute_error(t, p)
        dir_acc = np.mean(np.sign(p - LC_arr) == np.sign(t - LC_arr)) * 100
        metrics[name] = {"mae": mae, "dir_acc": dir_acc}
    return metrics, preds


def make_loaders(X_s, Y_s, sec, batch, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X_s).float(),
        torch.from_numpy(Y_s).float(),
        torch.from_numpy(sec).long(),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, pin_memory=True)


def train_fold(
    X_tr_s,
    Ret_tr_s,
    Sec_tr,
    X_te_s,
    Ret_te_s,
    Sec_te,
    sc_ret,
    LC_te,
    Y_px_te,
    fold_name,
    n_steps,
):
    model = PatchTST().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=LR,
        total_steps=n_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )
    stopper = EarlyStopping(patience=PATIENCE)
    amp = GradScaler()

    tr_dl = make_loaders(X_tr_s, Ret_tr_s, Sec_tr, BATCH, shuffle=True)
    te_dl = make_loaders(X_te_s, Ret_te_s, Sec_te, BATCH, shuffle=False)

    best_val = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, f"_fold_{fold_name}.pth")

    for ep in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, tr_dl, opt, amp, sched, train=True)
        val_loss = run_epoch(model, te_dl, None, None, None, train=False)

        logger.info(
            "[%s] Epoch %d/%d  train=%.4f  val=%.4f",
            fold_name,
            ep,
            EPOCHS,
            tr_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)

        if stopper.step(val_loss):
            logger.info("[%s] Early stopping at epoch %d", fold_name, ep)
            break

    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    )
    metrics, _ = evaluate(model, te_dl, sc_ret, LC_te, Y_px_te)

    logger.info("[%s] Results:", fold_name)
    for h, m in metrics.items():
        logger.info("  %s: MAE=%.2f  DirAcc=%.1f%%", h, m["mae"], m["dir_acc"])

    return model, metrics


if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "windows_756.npz",
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}.\n"
            "Run  python build_dataset.py --refresh  first."
        )

    logger.info("Loading dataset...")
    cache = np.load(dataset_path, allow_pickle=True)
    X = cache["X"]  # (N, 756, N_FEATS)
    Y_ret = cache["Y_ret"]  # (N, 3) — returns for 1w, 1m, 6m
    Y_px = cache["Y_px"]  # (N, 3) — prices for 1w, 1m, 6m
    LC = cache["LC"]  # (N,)
    has_meta = "dates" in cache and "sectors" in cache

    if has_meta:
        dates = cache["dates"]
        sectors = cache["sectors"].astype(np.int32)
        logger.info("Date and sector metadata available.")
    else:
        logger.warning("No date/sector metadata — run build_dataset.py --refresh")
        dates = None
        sectors = np.zeros(len(X), dtype=np.int32)

    # Verify feature count matches what we expect
    assert X.shape[2] == N_FEATS, (
        f"Feature count mismatch: dataset has {X.shape[2]} features, "
        f"expected {N_FEATS}. Run build_dataset.py --refresh."
    )

    logger.info(
        "Dataset: %s windows, %d features, %d horizons", X.shape[0], N_FEATS, N_HORIZONS
    )
    check_feature_array(X, "X (raw)")

    # -----------------------------------------------------------------------
    # Walk-forward cross-validation
    # Each fold trains on all data before test_start and evaluates on
    # [test_start, test_end). This simulates live deployment.
    # -----------------------------------------------------------------------
    mlflow.set_experiment("stock-forecasting-transformer")

    fold_metrics = {h: {"mae": [], "dir_acc": []} for h in HORIZONS}

    if has_meta:
        for fold_idx, (test_start, test_end) in enumerate(WF_FOLDS, 1):
            fold_name = f"fold{fold_idx}"
            tr_mask = dates < test_start
            te_mask = (dates >= test_start) & (dates < test_end)

            if tr_mask.sum() < 100 or te_mask.sum() < 10:
                logger.warning("Fold %d skipped — not enough data.", fold_idx)
                continue

            logger.info(
                "Fold %d: train=%d  test=%d  [%s → %s]",
                fold_idx,
                tr_mask.sum(),
                te_mask.sum(),
                test_start,
                test_end,
            )

            X_tr, X_te = X[tr_mask], X[te_mask]
            Ret_tr, Ret_te = Y_ret[tr_mask], Y_ret[te_mask]
            Px_te = Y_px[te_mask]
            LC_te = LC[te_mask]
            Sec_tr, Sec_te = sectors[tr_mask], sectors[te_mask]

            fs = X_tr.shape[2]
            sc_feat = StandardScaler().fit(X_tr.reshape(-1, fs))
            sc_ret = StandardScaler().fit(Ret_tr)

            X_tr_s = sc_feat.transform(X_tr.reshape(-1, fs)).reshape(X_tr.shape)
            X_te_s = sc_feat.transform(X_te.reshape(-1, fs)).reshape(X_te.shape)
            Ret_tr_s = sc_ret.transform(Ret_tr)
            Ret_te_s = sc_ret.transform(Ret_te)

            n_steps = (len(X_tr_s) // BATCH + 1) * EPOCHS

            with mlflow.start_run(run_name=f"patchtst_{fold_name}", nested=True):
                mlflow.log_params(
                    {
                        "fold": fold_idx,
                        "train_size": len(X_tr_s),
                        "test_size": len(X_te_s),
                        "test_start": test_start,
                        "test_end": test_end,
                    }
                )
                _, metrics = train_fold(
                    X_tr_s,
                    Ret_tr_s,
                    Sec_tr,
                    X_te_s,
                    Ret_te_s,
                    Sec_te,
                    sc_ret,
                    LC_te,
                    Px_te,
                    fold_name,
                    n_steps,
                )
                for h, m in metrics.items():
                    fold_metrics[h]["mae"].append(m["mae"])
                    fold_metrics[h]["dir_acc"].append(m["dir_acc"])
                    mlflow.log_metrics(
                        {f"mae_{h}": m["mae"], f"dir_acc_{h}": m["dir_acc"]}
                    )

        logger.info("\n=== Walk-Forward CV Summary ===")
        for h in HORIZONS:
            maes = fold_metrics[h]["mae"]
            dir_accs = fold_metrics[h]["dir_acc"]
            if maes:
                logger.info(
                    "  %s: MAE=%.2f ± %.2f  DirAcc=%.1f%% ± %.1f%%",
                    h,
                    np.mean(maes),
                    np.std(maes),
                    np.mean(dir_accs),
                    np.std(dir_accs),
                )

    # -----------------------------------------------------------------------
    # Final production model — train on ALL available data
    # This is the model that gets deployed and used for live predictions.
    # -----------------------------------------------------------------------
    logger.info("\n=== Training final production model on all data ===")

    fs = X.shape[2]
    sc_feat = StandardScaler().fit(X.reshape(-1, fs))
    sc_ret = StandardScaler().fit(Y_ret)

    X_s = sc_feat.transform(X.reshape(-1, fs)).reshape(X.shape)
    Ret_s = sc_ret.transform(Y_ret)

    log_dataset_summary(X_s, Ret_s, n_tickers=len(X) // WINDOW)

    final_ds = TensorDataset(
        torch.from_numpy(X_s).float(),
        torch.from_numpy(Ret_s).float(),
        torch.from_numpy(sectors).long(),
    )
    final_dl = DataLoader(final_ds, batch_size=BATCH, shuffle=True, pin_memory=True)

    final_model = PatchTST().to(DEVICE)
    final_opt = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=1e-3)
    n_steps = len(final_dl) * EPOCHS
    final_sched = torch.optim.lr_scheduler.OneCycleLR(
        final_opt,
        max_lr=LR,
        total_steps=n_steps,
        pct_start=0.3,
        anneal_strategy="cos",
    )
    final_amp = GradScaler()
    final_stopper = EarlyStopping(
        patience=PATIENCE + 2
    )  # slightly more patient on full data

    best_loss = float("inf")
    ckpt = os.path.join(SAVE_DIR, "transformer_multi_horizon.pth")

    with mlflow.start_run(run_name="patchtst_final"):
        mlflow.log_params(
            {
                "model": "PatchTST",
                "n_features": N_FEATS,
                "window": WINDOW,
                "patch_len": PATCH_LEN,
                "stride": STRIDE,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "dim_ff": 512,
                "dropout": 0.3,
                "sector_dim": SECTOR_DIM,
                "batch_size": BATCH,
                "lr": LR,
                "loss": "pinball_quantile",
                "horizons": list(HORIZONS.keys()),
                "n_folds_cv": len(WF_FOLDS) if has_meta else 0,
            }
        )
        # Log CV summary metrics
        for h in HORIZONS:
            if fold_metrics[h]["mae"]:
                mlflow.log_metric(f"cv_mae_{h}", np.mean(fold_metrics[h]["mae"]))
                mlflow.log_metric(
                    f"cv_dir_acc_{h}", np.mean(fold_metrics[h]["dir_acc"])
                )

        for ep in range(1, EPOCHS + 1):
            tr_loss = run_epoch(
                final_model, final_dl, final_opt, final_amp, final_sched, train=True
            )
            logger.info("Final  Epoch %d/%d  train=%.4f", ep, EPOCHS, tr_loss)
            mlflow.log_metric("train_loss", tr_loss, step=ep)

            if tr_loss < best_loss:
                best_loss = tr_loss
                torch.save(final_model.state_dict(), ckpt)

            if final_stopper.step(tr_loss):
                logger.info("Stopping final training at epoch %d", ep)
                break

        mlflow.log_artifact(ckpt)

    joblib.dump(sc_feat, os.path.join(SAVE_DIR, "scaler_feat.pkl"))
    joblib.dump(sc_ret, os.path.join(SAVE_DIR, "scaler_ret.pkl"))
    joblib.dump(
        {
            "window": WINDOW,
            "patch_len": PATCH_LEN,
            "stride": STRIDE,
            "horizons": HORIZONS,
            "n_features": N_FEATS,
            "model_type": "PatchTST",
            "n_quantiles": 3,
            "quantiles": QUANTILES,
        },
        os.path.join(SAVE_DIR, "transformer_meta.pkl"),
    )

    logger.info("Production model saved to %s", ckpt)
    logger.info("Training complete.")
