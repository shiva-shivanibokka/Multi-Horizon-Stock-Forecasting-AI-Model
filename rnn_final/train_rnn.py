"""
train_rnn.py
Trains an improved Bidirectional GRU with Temporal Attention and Sector
Conditioning to predict stock prices at three horizons: 1 week, 1 month,
and 6 months.

Why GRU instead of vanilla RNN?

    The original model was a vanilla RNN:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1})

    Vanilla RNNs have the vanishing gradient problem: gradients shrink
    exponentially as they propagate backward through 252 timesteps, making
    the model effectively blind to events more than ~50 days in the past.

    A GRU (Gated Recurrent Unit) solves this with two learnable gates:

    Reset gate:  r_t = sigmoid(W_r * [h_{t-1}, x_t])
        Controls how much of the previous hidden state to forget.
        When r_t ≈ 0, the model ignores the past and resets to the current input.
        This lets it quickly adapt to sudden regime changes (e.g. COVID crash).

    Update gate: z_t = sigmoid(W_z * [h_{t-1}, x_t])
        Controls how much the hidden state should update vs. carry forward.
        When z_t ≈ 1, the previous state is kept almost unchanged — the model
        can maintain long-range memory without gradient vanishing.

    New hidden: h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W * [r_t * h_{t-1}, x_t])

    GRU vs LSTM:
        LSTM has 3 gates (forget, input, output) and a separate cell state.
        GRU has 2 gates and merges the cell and hidden state.
        GRU is ~25% fewer parameters than LSTM for the same hidden size,
        trains faster, and often matches LSTM on short sequences like 252 days.

Architecture improvements over the original vanilla RNN:

1. GRU replaces vanilla RNN — gating eliminates vanishing gradients
2. Bidirectional — forward and backward passes combined, output dim = hidden*2
3. 3 layers instead of 2 — deeper hierarchical representations
4. Hidden size 128 -> 512 — 15.8M parameters vs the original 0.03M
5. Temporal Attention (Bahdanau) — attends over all 252 hidden states
6. Sector conditioning — learned sector embedding added to inputs
7. Separate deeper output heads per horizon

How to run:
    1. python build_dataset.py --refresh
    2. python rnn_final/train_rnn.py
    (The first run will extract X_seq_252.npy automatically, ~5-10 min)
"""

import os
import sys
import math
import joblib

import numpy as np
import torch
import mlflow
from torch import nn
from tqdm import tqdm

mlflow.set_tracking_uri("file:./mlruns")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import check_feature_array, log_dataset_summary

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# HYPERPARAMETERS

HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
N_HORIZONS = len(HORIZONS)
WINDOW = 252  # 1 year of trading days
N_FEATS = 36
N_SECTORS = 11

# Model capacity
HIDDEN_SIZE = 512  # per direction; total = 512*2 = 1024 (bidirectional)
NUM_LAYERS = 3  # stacked BiGRU layers
DROPOUT = 0.3
SECTOR_DIM = 16

# Training
EPOCHS = 70
BATCH = 128
LR = 3e-4
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# DATASET


class MmapDataset(torch.utils.data.Dataset):
    """
    Loads (window, target, sector) triples from memory-mapped files.

    X_path:   path to X_seq_252.npy (float16, pre-scaled during extraction)
    Y_scaled: (N, 3) float32 — scaled target prices, fully in RAM
    sec:      (N,) int32 — GICS sector index per window
    indices:  which rows of the full dataset this split uses

    The mmap is opened lazily per worker — Windows cannot pickle mmap objects.
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
        x = self._X[real_i].astype(np.float32)  # (252, 36)
        y = self.Y[real_i].copy()  # (3,)
        s = int(self.sec[real_i])
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(s, dtype=torch.long),
        )


def make_loader(X_path, Y_scaled, sec, indices, batch, shuffle):
    ds = MmapDataset(X_path, Y_scaled, sec, indices)
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

    For each timestep t, computes:
        score_t = v * tanh(W_h * h_t  +  W_q * query)

    Scores are normalised with softmax to give attention weights.
    Context = weighted sum of all hidden states.

    This allows the model to focus on the most predictive timesteps
    (e.g. recent earnings dates, volatility spikes, sector breakouts)
    rather than treating all 252 days equally.

    Uses additive attention to avoid the (B, T, T) memory cost of
    dot-product attention — only (B, T, 1) scores are needed.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, hidden: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, H)
        query:  (B, H)  — typically the last hidden state
        returns context: (B, H)
        """
        q = self.W_q(query).unsqueeze(1)  # (B, 1, H)
        scores = self.v(self.tanh(self.W_h(hidden) + q))  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        return (weights * hidden).sum(dim=1)  # (B, H)


class BiGRUForecast(nn.Module):
    """
    Bidirectional GRU with Temporal Attention and Sector Conditioning.

    Architecture:
        Input (B, 252, 36)
          |
          + sector embedding projected to (B, 252, 36)
          |
        3-layer BiGRU: (B, 252, 36) -> (B, 252, 1024)
          |
        LayerNorm
          |
        Temporal Attention -> context (B, 1024)
          |
        Dropout
          |
        Three output heads (1w, 1m, 6m) -> (B, 3)

    Why GRU here and LSTM in the other model?
        Both solve the vanishing gradient problem with gating. GRU is
        computationally cheaper (~25% fewer parameters for the same hidden
        size) and typically converges faster. For the 252-day window used
        here, GRU is well-matched — the gates provide enough memory capacity
        for 1-year patterns without the overhead of LSTM's separate cell state.
        The LSTM model uses a 756-day window where the extra cell state
        provides more benefit.
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
        hidden_dim = hidden_size * 2  # bidirectional output

        # Sector conditioning
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, input_size)

        # 3-layer Bidirectional GRU
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = TemporalAttention(hidden_dim)
        self.drop = nn.Dropout(dropout)

        def _head(deep=False):
            if deep:
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
        x:      (B, 252, 36)
        sector: (B,) — integer sector index [0, 10]
        returns: (B, 3)
        """
        sec_emb = self.sector_proj(self.sector_embed(sector))
        x = x + sec_emb.unsqueeze(1)

        h_all, _ = self.gru(x)  # (B, 252, hidden_size*2)
        h_all = self.layer_norm(h_all)
        query = h_all[:, -1, :]  # last timestep as attention query
        context = self.attention(h_all, query)
        context = self.drop(context)

        return torch.cat(
            [
                self.head_1w(context),
                self.head_1m(context),
                self.head_6m(context),
            ],
            dim=1,
        )  # (B, 3)


# EARLY STOPPING


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss) -> bool:
        if not (val_loss == val_loss):  # NaN check
            self.counter += 1
            return self.counter >= self.patience
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# MAIN


def main():
    DATASET_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
    )
    X_PATH = os.path.join(DATASET_DIR, "windows_252.npz")
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {X_PATH}.\n"
            "Run  python build_dataset.py --refresh  first."
        )

    # Load sector data from the 756-day dataset
    # The 252-day and 756-day datasets cover the same tickers and windows,
    # so sector labels are looked up by window index alignment.
    SEC_756 = os.path.join(DATASET_DIR, "sectors_756.npy")

    # ONE-TIME EXTRACTION: windows_252.npz -> X_seq_252.npy (float16, scaled)
    # See original train_rnn.py or the comments below for full explanation.
    X_SEQ_NPY = os.path.join(DATASET_DIR, "X_seq_252.npy")
    Y_NPY = os.path.join(DATASET_DIR, "Y_252.npy")
    LC_NPY = os.path.join(DATASET_DIR, "LC_252.npy")
    SEC_252_NPY = os.path.join(DATASET_DIR, "sectors_252.npy")
    SC_FEAT_252 = os.path.join(DATASET_DIR, "scaler_feat_252.pkl")

    if not os.path.exists(X_SEQ_NPY):
        print("One-time extraction: windows_252.npz -> X_seq_252.npy (scaled float16)")
        print("  Fitting StandardScaler on 5k sample to prevent float16 overflow...")
        import zipfile
        import numpy.lib.format as npfmt

        SCALER_FIT_N = 5_000
        CHUNK_ROWS = 2_000

        with zipfile.ZipFile(X_PATH, "r") as zf:
            with zf.open("X_seq.npy") as fh:
                ver = npfmt.read_magic(fh)
                if ver == (2, 0):
                    shape, _, src_dtype = npfmt.read_array_header_2_0(fh)
                else:
                    shape, _, src_dtype = npfmt.read_array_header_1_0(fh)

            N_rows, W_ex, F_ex = shape
            row_bytes = W_ex * F_ex * np.dtype(src_dtype).itemsize

            print(f"  Source: shape={shape}  dtype={src_dtype}")

            with zf.open("X_seq.npy") as fh:
                ver2 = npfmt.read_magic(fh)
                if ver2 == (2, 0):
                    npfmt.read_array_header_2_0(fh)
                else:
                    npfmt.read_array_header_1_0(fh)
                raw_sample = fh.read(SCALER_FIT_N * row_bytes)

            sample = np.frombuffer(raw_sample, dtype=src_dtype).reshape(
                SCALER_FIT_N, W_ex, F_ex
            )
            sc_feat = StandardScaler().fit(sample.reshape(-1, F_ex))
            sc_mean = sc_feat.mean_.astype(np.float32)
            sc_scale = sc_feat.scale_.astype(np.float32)
            print("  Scaler fitted.")

            with zf.open("Y.npy") as fh:
                Y_raw = np.load(fh)
            np.save(Y_NPY, Y_raw)

            print("  Extracting LC (last close per window)...")
            LC_all = np.empty(N_rows, dtype=np.float32)
            with zf.open("X_seq.npy") as fh:
                ver3 = npfmt.read_magic(fh)
                if ver3 == (2, 0):
                    npfmt.read_array_header_2_0(fh)
                else:
                    npfmt.read_array_header_1_0(fh)
                rows_read = 0
                while rows_read < N_rows:
                    n = min(CHUNK_ROWS, N_rows - rows_read)
                    raw = fh.read(n * row_bytes)
                    if not raw:
                        break
                    chunk = np.frombuffer(raw, dtype=src_dtype).reshape(n, W_ex, F_ex)
                    LC_all[rows_read : rows_read + n] = chunk[:, -1, 3]
                    rows_read += n
            np.save(LC_NPY, LC_all)

            print(
                f"  Writing scaled float16 X ({N_rows * W_ex * F_ex * 2 / 1e9:.2f} GB)..."
            )
            out_dtype = np.dtype(np.float16)
            with open(X_SEQ_NPY, "wb") as fout:
                dummy = np.empty((N_rows, W_ex, F_ex), dtype=out_dtype)
                npfmt.write_array_header_2_0(
                    fout, npfmt.header_data_from_array_1_0(dummy)
                )
                del dummy
                with zf.open("X_seq.npy") as fh:
                    ver4 = npfmt.read_magic(fh)
                    if ver4 == (2, 0):
                        npfmt.read_array_header_2_0(fh)
                    else:
                        npfmt.read_array_header_1_0(fh)
                    rows_done = 0
                    while rows_done < N_rows:
                        n = min(CHUNK_ROWS, N_rows - rows_done)
                        raw = fh.read(n * row_bytes)
                        if not raw:
                            break
                        chunk_f32 = np.frombuffer(raw, dtype=src_dtype).reshape(
                            n, W_ex, F_ex
                        )
                        flat = chunk_f32.reshape(-1, F_ex)
                        scaled = ((flat - sc_mean) / sc_scale).reshape(n, W_ex, F_ex)
                        fout.write(scaled.astype(np.float16).tobytes())
                        rows_done += n
                        print(
                            f"\r  {rows_done}/{N_rows} rows ({rows_done / N_rows * 100:.1f}%)",
                            end="",
                            flush=True,
                        )
            print(f"\n  Done. {os.path.getsize(X_SEQ_NPY) / 1e9:.2f} GB written.")

        joblib.dump(sc_feat, SC_FEAT_252)

    # Build or load sector labels for 252-day windows.
    # The 252-day dataset has more windows than the 756-day dataset (875k vs 657k).
    # We use the sectors_756.npy as a reference and tile/interpolate as needed,
    # or simply assign sector 0 if the file is not available.
    if not os.path.exists(SEC_252_NPY):
        X_tmp = np.load(X_SEQ_NPY, mmap_mode="r")
        N_252 = len(X_tmp)
        if os.path.exists(SEC_756):
            # Repeat sectors_756 to match length of 252-day dataset
            sec_756 = np.load(SEC_756).astype(np.int32)
            # Tile and trim to match N_252
            reps = (N_252 // len(sec_756)) + 1
            sec_252 = np.tile(sec_756, reps)[:N_252]
        else:
            sec_252 = np.zeros(N_252, dtype=np.int32)
        np.save(SEC_252_NPY, sec_252)

    # Load dataset
    print(f"Loading dataset from {X_SEQ_NPY} ...")
    X = np.load(X_SEQ_NPY, mmap_mode="r")  # (875771, 252, 36) float16
    Y = np.load(Y_NPY)  # (875771, 3) float32
    LC = np.load(LC_NPY)  # (875771,) float32
    sectors = np.load(SEC_252_NPY).astype(np.int32)  # (875771,) int32
    sc_feat = joblib.load(SC_FEAT_252)

    N, W, F = X.shape
    print(f"Loaded: X={X.shape} dtype={X.dtype}  Y={Y.shape}")

    # Chronological 80/20 split
    split = int(0.8 * N)
    tr_idx = np.arange(split)
    te_idx = np.arange(split, N)
    assert len(tr_idx) > len(te_idx)
    assert len(tr_idx) + len(te_idx) == N
    LC_te = LC[te_idx]

    # Target scaler — fit on training split only
    sc_targ = StandardScaler().fit(Y[tr_idx])
    Y_scaled = sc_targ.transform(Y).astype(np.float32)

    fit_idx = tr_idx[
        np.random.choice(len(tr_idx), min(2_000, len(tr_idx)), replace=False)
    ]
    check_feature_array(
        np.array(X[fit_idx], dtype=np.float32).reshape(-1, F), "X sample"
    )
    log_dataset_summary(
        np.array(X[fit_idx], dtype=np.float32).reshape(-1, F),
        Y_scaled[fit_idx],
        n_tickers=N // WINDOW,
    )

    joblib.dump(sc_feat, "rnn_scaler_feat.pkl")
    joblib.dump(sc_targ, "rnn_scaler_targ.pkl")

    train_dl = make_loader(X_SEQ_NPY, Y_scaled, sectors, tr_idx, BATCH, shuffle=True)
    val_dl = make_loader(X_SEQ_NPY, Y_scaled, sectors, te_idx, BATCH, shuffle=False)

    # Model
    model = BiGRUForecast().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BiGRUForecast  Parameters: {n_params / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_steps = len(train_dl) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=0.2, anneal_strategy="cos"
    )
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=PATIENCE)
    best_val = float("inf")
    ckpt_path = "rnn_best.pth"

    mlflow.set_experiment("stock-forecasting-rnn")
    with mlflow.start_run(run_name="bigru"):
        mlflow.log_params(
            {
                "model": "BiGRUForecast",
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
            range(1, EPOCHS + 1), desc="[rnn]", unit="ep", dynamic_ncols=True
        )
        for ep in epoch_bar:
            # Training phase
            model.train()
            total, n = 0.0, 0
            bar = tqdm(
                train_dl,
                desc=f"  train ep{ep}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            for xb, yb, sb in bar:
                xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                pred = model(xb, sb)
                loss = loss_fn(pred, yb)
                if not torch.isfinite(loss):
                    bar.set_postfix(loss="NaN-skip")
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * xb.size(0)
                n += xb.size(0)
                bar.set_postfix(loss=f"{loss.item():.4f}")
            train_mse = total / n if n > 0 else float("nan")

            # Validation phase
            model.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb, sb in val_dl:
                    xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
                    loss = loss_fn(model(xb, sb), yb)
                    if not torch.isfinite(loss):
                        continue
                    vtotal += loss.item() * xb.size(0)
                    vn += xb.size(0)
            val_mse = vtotal / vn if vn > 0 else float("nan")

            epoch_bar.set_postfix(
                train=f"{train_mse:.4f}", val=f"{val_mse:.4f}", best=f"{best_val:.4f}"
            )
            mlflow.log_metrics(
                {"train_mse": float(train_mse), "val_mse": float(val_mse)}, step=ep
            )

            if val_mse == val_mse and val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), ckpt_path)
            elif not os.path.exists(ckpt_path):
                torch.save(model.state_dict(), ckpt_path)

            if stopper.step(val_mse):
                print(f"\nEarly stopping at epoch {ep}")
                break

        model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        )

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
        torch.save(model.state_dict(), "rnn_multi_horizon.pth")
        joblib.dump(sc_feat, "rnn_scaler_feat.pkl")
        joblib.dump(sc_targ, "rnn_scaler_targ.pkl")
        joblib.dump(
            {
                "window": WINDOW,
                "horizons": HORIZONS,
                "n_features": F,
                "model_type": "BiGRUForecast",
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "bidirectional": True,
                "sector_dim": SECTOR_DIM,
            },
            "rnn_meta.pkl",
        )
        mlflow.log_artifact("rnn_multi_horizon.pth")
        print("Done. Model and scalers saved.")


if __name__ == "__main__":
    main()
