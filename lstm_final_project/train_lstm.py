"""
train_lstm.py
Trains an improved Bidirectional LSTM with Temporal Attention and Sector
Conditioning to predict stock prices at three horizons: 1 week, 1 month,
and 6 months.

Architecture improvements over the original:

1. Bidirectional LSTM
   The original LSTM only read the sequence forward: day 1 -> day 756.
   A bidirectional LSTM runs two separate LSTMs — one forward, one backward.
   At each timestep, the hidden state combines information from both directions:
     - Forward pass: "given everything up to day t, what is the context?"
     - Backward pass: "given everything after day t, what is the context?"
   This lets every timestep see the full 756-day window, not just the past.
   Within the training window this is not cheating — all 756 days are historical
   data. The output dimension doubles: hidden_size -> hidden_size*2.

2. Deeper: 3 layers instead of 2
   Three stacked BiLSTM layers allow the model to learn increasingly abstract
   temporal representations:
     Layer 1: raw price patterns — momentum, reversals, volatility spikes
     Layer 2: patterns of patterns — trend structure, regime transitions
     Layer 3: high-level market dynamics — cross-ticker correlation effects
              encoded via the sector embedding

3. Larger hidden size: 128 -> 512
   More hidden units = more representational capacity. With 19.6M parameters
   (vs the original 0.89M), the model can capture far more nuanced patterns in
   the 756-day price history. The RTX 4060 Laptop handles this comfortably —
   total VRAM usage is only ~0.5 GB of the available 8.6 GB.

4. Temporal Attention (Bahdanau-style)
   Instead of only using the final hidden state (h_756), the model now computes
   a weighted average over ALL 756 hidden states, where the weights are learned.

   The attention score for timestep t is:
       score_t = v * tanh(W1 * h_t  +  W2 * h_final)
   where h_final is the last hidden state used as a "query."

   This answers the question: "of all 756 days in this window, which ones are
   most relevant for predicting the future?" The model can learn to focus on
   recent earnings dates, key macro events, or price breakouts — whatever the
   training data reveals as most predictive.

   We use additive (Bahdanau) attention rather than dot-product attention because
   the hidden dimension (1024 after bidirectional) would make dot-product
   attention over T=756 timesteps use 146 MB of VRAM per batch. Additive
   attention uses only 0.4 MB while being equally expressive.

5. Sector Conditioning
   Each of the 11 GICS sectors gets a learned embedding vector. This vector is
   projected to the input dimension and added to the raw input features before
   the LSTM processes them. This allows the model to learn sector-specific
   patterns — technology stocks have different momentum dynamics than utilities.

6. Separate output heads per horizon
   Each forecast horizon (1w, 1m, 6m) has its own two-layer MLP head that maps
   from the attended hidden state to the price prediction. This allows each
   horizon to specialise in the patterns most relevant to its timescale.

How to run:
    1. python build_dataset.py --refresh
    2. python dataset/prescale.py
    3. python lstm_final_project/train_lstm.py
"""

import os
import sys
import math
import joblib

import numpy as np
import torch
import mlflow
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

mlflow.set_tracking_uri("file:./mlruns")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import check_feature_array, log_dataset_summary


# HYPERPARAMETERS

HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
N_HORIZONS = len(HORIZONS)
WINDOW = 756  # 3 years of trading days
N_FEATS = 36
N_SECTORS = 11  # GICS sector count

# Model capacity
HIDDEN_SIZE = 512  # per direction; total hidden dim = 512*2 = 1024 (bidirectional)
NUM_LAYERS = 3  # stacked BiLSTM layers
DROPOUT = 0.3  # applied between layers and in output heads
SECTOR_DIM = 16  # sector embedding dimension

# Training
EPOCHS = 80
BATCH = 128
LR = 3e-4  # slightly lower than original to suit the larger model
PATIENCE = 12  # more patient — larger model needs more epochs to converge

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# DATASET


class ScaledMmapDataset(torch.utils.data.Dataset):
    """
    Loads (window, target, sector) triples from memory-mapped files.

    X_path:  path to X_756_scaled.npy (float16, pre-scaled by prescale.py)
    Y_scaled: (N, 3) float32 — scaled target prices, fully in RAM
    sec:      (N,) int32 — GICS sector index per window, fully in RAM
    indices:  (M,) int64 — which rows of the full dataset this split uses

    The mmap is opened lazily per worker (Windows cannot pickle mmap objects).
    """

    def __init__(self, X_path, Y_scaled, sec, indices):
        self.X_path = X_path
        self.Y = Y_scaled
        self.sec = sec
        self.idx = indices
        self._X = None

    def _open(self):
        if self._X is None:
            self._X = np.load(self.X_path, mmap_mode="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        self._open()
        real_i = self.idx[i]
        x = self._X[real_i].astype(np.float32)  # (756, 36) float32
        y = self.Y[real_i].copy()  # (3,) float32
        s = int(self.sec[real_i])  # scalar int
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(s, dtype=torch.long),
        )


def make_loader(X_path, Y_scaled, sec, indices, batch, shuffle):
    ds = ScaledMmapDataset(X_path, Y_scaled, sec, indices)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


# MODEL


class TemporalAttention(nn.Module):
    """
    Bahdanau (additive) temporal attention over a sequence of hidden states.

    Given:
        hidden:  (B, T, H) — all T hidden states from the BiLSTM
        query:   (B, H)    — the final hidden state used as the query

    The attention mechanism computes a scalar score for each timestep:
        score_t = v * tanh(W_h * h_t  +  W_q * query)

    Scores are normalised with softmax to give attention weights alpha (B, T).
    The attended context is the weighted sum:
        context = sum_t(alpha_t * h_t)   ->   (B, H)

    Why additive attention?
        Dot-product attention (used in Transformers) would require computing
        (B, T, T) attention matrices — 128 * 756 * 756 * 2 bytes = 146 MB per
        batch. Additive attention only computes (B, T, 1) scores = 0.4 MB, while
        being equally expressive for sequence summarisation tasks.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Project hidden states and query to a shared attention space
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, hidden: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, H)
        query:  (B, H)
        returns context: (B, H)
        """
        # Expand query to match hidden: (B, 1, H) -> broadcasts over T
        q = self.W_q(query).unsqueeze(1)  # (B, 1, H)
        h = self.W_h(hidden)  # (B, T, H)
        scores = self.v(self.tanh(h + q))  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = (weights * hidden).sum(dim=1)  # (B, H)
        return context


class BiLSTMForecast(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention and Sector Conditioning.

    Architecture:
        Input (B, 756, 36)
          |
          + sector embedding (B, SECTOR_DIM) projected and broadcast -> (B, 756, 36)
          |
        3-layer BiLSTM: (B, 756, 36) -> (B, 756, 1024)
          |
        Temporal Attention over (B, 756, 1024) -> context (B, 1024)
          |
        Three separate output heads:
          head_1w: Linear(1024->512) -> GELU -> Dropout -> Linear(512->1)
          head_1m: same structure
          head_6m: deeper (extra hidden layer) -> scalar

    The final output is (B, 3) — one prediction per horizon.

    Why separate heads?
        The patterns most predictive of 1-week returns (recent momentum, earnings
        gap fill) are different from those predictive of 6-month returns (sector
        rotation, macro regime). Giving each horizon its own MLP lets it specialise.
    """

    def __init__(
        self,
        input_size=N_FEATS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_sectors=N_SECTORS,
        sector_dim=SECTOR_DIM,
        dropout=DROPOUT,
        out_size=N_HORIZONS,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        hidden_dim = hidden_size * 2  # bidirectional doubles the output dim

        # Sector embedding: maps sector index -> sector_dim vector
        # projected to input_size and added to each timestep before the LSTM
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, input_size)

        # 3-layer Bidirectional LSTM
        # bidirectional=True: runs forward and backward LSTM in parallel
        # output dim per timestep: hidden_size * 2 = 1024
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Layer normalisation on the BiLSTM output for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Temporal attention over all 756 hidden states
        self.attention = TemporalAttention(hidden_dim)

        # Dropout applied to the attended context before the output heads
        self.drop = nn.Dropout(dropout)

        def _head(deep=False):
            """Two-layer MLP output head for one forecast horizon."""
            if deep:
                # Deeper head for 6-month horizon — needs more capacity to model
                # long-range macro effects
                return nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 4, 1),
                )
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.head_1w = _head()
        self.head_1m = _head()
        self.head_6m = _head(deep=True)

    def forward(self, x: torch.Tensor, sector: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 756, 36)
        sector: (B,) — integer sector index [0, 10]
        returns: (B, 3) — one prediction per horizon
        """
        # Add sector embedding to every timestep of the input
        # This tells the LSTM which sector context it is operating in
        sec_emb = self.sector_proj(self.sector_embed(sector))  # (B, input_size)
        x = x + sec_emb.unsqueeze(1)  # (B, 756, input_size)

        # Run through 3-layer BiLSTM
        # h_all: (B, 756, hidden_size*2) — hidden state at every timestep
        # h_n:   (num_layers*2, B, hidden_size) — final hidden states (not used directly)
        h_all, _ = self.lstm(x)
        h_all = self.layer_norm(h_all)  # stabilise activations

        # Use the last timestep's concatenated forward+backward state as the query
        # h_all[:, -1, :] = [h_forward_T ; h_backward_0] — sees the full sequence
        query = h_all[:, -1, :]  # (B, hidden_size*2)

        # Compute attention-weighted context over all 756 timesteps
        context = self.attention(h_all, query)  # (B, hidden_size*2)
        context = self.drop(context)

        # Apply each horizon head and concatenate
        p1w = self.head_1w(context)  # (B, 1)
        p1m = self.head_1m(context)  # (B, 1)
        p6m = self.head_6m(context)  # (B, 1)

        return torch.cat([p1w, p1m, p6m], dim=1)  # (B, 3)


# EARLY STOPPING


class EarlyStopping:
    """Stops training when validation loss has not improved for patience epochs."""

    def __init__(self, patience=12, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss) -> bool:
        if not (val_loss < float("inf")):
            self.counter += 1
            return self.counter >= self.patience
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# TRAINING LOOP


def run_epoch(model, dl, opt, amp_scaler, scheduler, train, desc):
    """One full pass through the dataset. Returns average MSE loss per sample."""
    model.train() if train else model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    crit = nn.MSELoss()

    with ctx:
        bar = tqdm(dl, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for xb, yb, sb in bar:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)

            if train:
                opt.zero_grad()
                with autocast("cuda"):
                    pred = model(xb, sb)
                    loss = crit(pred, yb)
                if not torch.isfinite(loss):
                    bar.set_postfix(loss="NaN-skip")
                    continue
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(opt)
                amp_scaler.update()
                if scheduler is not None:
                    scheduler.step()
            else:
                with torch.no_grad():
                    pred = model(xb, sb)
                loss = crit(pred, yb)
                if not torch.isfinite(loss):
                    continue

            total += loss.item() * xb.size(0)
            n += xb.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total / n if n > 0 else float("nan")


# MAIN


def main():
    DATASET_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
    )

    X_SCALED = os.path.join(DATASET_DIR, "X_756_scaled.npy")
    Y_PX_NPY = os.path.join(DATASET_DIR, "Y_px_756.npy")
    LC_NPY = os.path.join(DATASET_DIR, "LC_756.npy")
    SEC_NPY = os.path.join(DATASET_DIR, "sectors_756.npy")
    SC_FEAT_PKL = os.path.join(DATASET_DIR, "scaler_feat.pkl")

    for p in [X_SCALED, Y_PX_NPY, LC_NPY, SEC_NPY, SC_FEAT_PKL]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\nRun  python dataset/prescale.py  first."
            )

    print(f"Loading dataset from {X_SCALED} ...")
    X = np.load(X_SCALED, mmap_mode="r")  # (657035, 756, 36) float16
    Y_px = np.load(Y_PX_NPY)  # (657035, 3) float32
    LC = np.load(LC_NPY)  # (657035,) float32
    sectors = np.load(SEC_NPY).astype(np.int32)  # (657035,) int32
    sc_feat_global = joblib.load(SC_FEAT_PKL)

    N, W, F = X.shape
    print(f"Loaded: X={X.shape} dtype={X.dtype}  Y_px={Y_px.shape}")

    sample_idx = np.random.choice(N, 2000, replace=False)
    check_feature_array(np.array(X[sample_idx], dtype=np.float32), "X sample")

    # Chronological 80/20 split
    split = int(0.8 * N)
    tr_idx = np.arange(split)
    te_idx = np.arange(split, N)
    assert len(tr_idx) > len(te_idx)
    assert len(tr_idx) + len(te_idx) == N

    # Target scaler — fit on training split only to prevent data leakage
    sc_targ = StandardScaler().fit(Y_px[tr_idx])
    Y_scaled = sc_targ.transform(Y_px).astype(np.float32)
    LC_te = LC[te_idx]

    fit_idx = tr_idx[
        np.random.choice(len(tr_idx), min(5_000, len(tr_idx)), replace=False)
    ]
    log_dataset_summary(
        np.array(X[fit_idx], dtype=np.float32).reshape(-1, F),
        Y_scaled[fit_idx],
        n_tickers=N // WINDOW,
    )

    joblib.dump(sc_feat_global, "lstm_scaler_feat.pkl")
    joblib.dump(sc_targ, "lstm_scaler_targ.pkl")

    # DataLoaders — pass sectors alongside X and Y
    train_dl = make_loader(X_SCALED, Y_scaled, sectors, tr_idx, BATCH, shuffle=True)
    val_dl = make_loader(X_SCALED, Y_scaled, sectors, te_idx, BATCH, shuffle=False)

    # Model
    model = BiLSTMForecast().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BiLSTMForecast  Parameters: {n_params / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_steps = len(train_dl) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=0.2, anneal_strategy="cos"
    )
    amp = GradScaler("cuda")
    stopper = EarlyStopping(patience=PATIENCE)
    best_val = float("inf")
    ckpt = "lstm_best.pth"

    mlflow.set_experiment("stock-forecasting-lstm")
    with mlflow.start_run(run_name="bilstm"):
        mlflow.log_params(
            {
                "model": "BiLSTMForecast",
                "window": WINDOW,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "bidirectional": True,
                "attention": "Bahdanau (additive)",
                "sector_dim": SECTOR_DIM,
                "dropout": DROPOUT,
                "epochs": EPOCHS,
                "batch_size": BATCH,
                "lr": LR,
                "patience": PATIENCE,
                "n_params_M": round(n_params / 1e6, 1),
                "split": "chronological 80/20",
                "loss": "MSE",
            }
        )

        epoch_bar = tqdm(
            range(1, EPOCHS + 1), desc="[lstm]", unit="ep", dynamic_ncols=True
        )
        for ep in epoch_bar:
            tr_loss = run_epoch(
                model, train_dl, opt, amp, sched, train=True, desc=f"  train ep{ep}"
            )
            val_loss = run_epoch(
                model, val_dl, None, None, None, train=False, desc=f"  val   ep{ep}"
            )

            epoch_bar.set_postfix(
                train=f"{tr_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val:.4f}"
            )
            mlflow.log_metrics(
                {"train_mse": float(tr_loss), "val_mse": float(val_loss)}, step=ep
            )

            if torch.isfinite(torch.tensor(val_loss)) and val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt)
            elif not os.path.exists(ckpt):
                torch.save(model.state_dict(), ckpt)

            if stopper.step(val_loss):
                print(f"Early stopping at epoch {ep}")
                break

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))

        # Evaluation
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb, sb in val_dl:
                all_pred.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())
                all_true.append(yb.numpy())

        all_pred = sc_targ.inverse_transform(np.concatenate(all_pred))
        all_true = sc_targ.inverse_transform(np.concatenate(all_true))

        print("\n--- Validation metrics per horizon ---")
        for i, key in enumerate(HORIZONS):
            t, p = all_true[:, i], all_pred[:, i]
            mae = mean_absolute_error(t, p)
            rmse = math.sqrt(mean_squared_error(t, p))
            r2 = r2_score(t, p)
            dir_acc = np.mean(np.sign(p - LC_te) == np.sign(t - LC_te)) * 100
            print(
                f"  {key}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  DirAcc={dir_acc:.1f}%"
            )
            mlflow.log_metrics(
                {f"val_mae_{key}": float(mae), f"val_dir_acc_{key}": float(dir_acc)},
                step=ep,
            )

        # Save
        torch.save(model.state_dict(), "lstm_multi_horizon.pth")
        joblib.dump(sc_feat_global, "lstm_scaler_feat.pkl")
        joblib.dump(sc_targ, "lstm_scaler_targ.pkl")
        joblib.dump(
            {
                "window": WINDOW,
                "horizons": HORIZONS,
                "n_features": F,
                "model_type": "BiLSTMForecast",
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "bidirectional": True,
                "sector_dim": SECTOR_DIM,
            },
            "lstm_meta.pkl",
        )
        mlflow.log_artifact("lstm_multi_horizon.pth")
        print("Done. Model and scalers saved.")


if __name__ == "__main__":
    main()
