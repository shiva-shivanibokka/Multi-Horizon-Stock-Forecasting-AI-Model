"""
train_transformer.py
Trains a PatchTST Transformer to predict stock RETURNS at three future horizons:
1 week, 1 month, and 6 months — with uncertainty quantification via quantile loss.

What is a Transformer?
    A Transformer processes the entire input sequence simultaneously using
    "self-attention." At each position in the sequence, the model computes how
    much attention to pay to every other position. This is very different from
    RNNs and LSTMs, which process the sequence one step at a time.

    Self-attention allows the model to directly connect any two timesteps,
    regardless of how far apart they are. An LSTM processing 756 days must
    pass information through 756 intermediate hidden states to connect day 1 to
    day 756. A Transformer connects them in a single attention operation.

What is PatchTST?
    PatchTST (Patch Time Series Transformer) is a modification that divides the
    input sequence into non-overlapping or overlapping "patches" before feeding
    them to the Transformer. This is analogous to how Vision Transformers divide
    images into patches.

    Instead of processing 756 individual timesteps, PatchTST creates:
        n_patches = (756 - patch_len) / stride + 1
                  = (756 - 16) / 8 + 1 = 93 patches

    Self-attention is O(n²) in sequence length. The patch approach reduces
    attention complexity from 756² = 571,536 to 93² = 8,649 — a 66x reduction.
    Each patch captures local temporal patterns (like a 16-day price movement),
    and attention across patches captures long-range relationships.

Why quantile (pinball) loss instead of MSE?
    MSE trains the model to predict the single most likely future price.
    Pinball loss trains the model to predict THREE quantiles simultaneously:
      - p10 (10th percentile): a pessimistic scenario
      - p50 (50th percentile): the median / most likely scenario
      - p90 (90th percentile): an optimistic scenario

    This gives users an uncertainty range, not just a point estimate.
    A trader can see "the model says 50% chance the price is between $140-$160."

    The pinball loss for quantile q is:
        L_q(y, y_hat) = q * max(y - y_hat, 0) + (1 - q) * max(y_hat - y, 0)
    This asymmetrically penalises under/over-prediction to push the model
    toward the correct quantile.

Why walk-forward cross-validation (3 folds)?
    A single train/test split can give misleading results if the test period
    happens to be an unusually easy or hard market regime.

    Walk-forward CV trains and evaluates across three separate 1-year test periods:
      Fold 1: train 2016-2023  -> test 2023     (post-COVID bull run)
      Fold 2: train 2016-2024  -> test 2024     (election year, rate cuts)
      Fold 3: train 2016-2025  -> test 2025     (current regime)

    The CV metrics (averaged across folds) give an honest estimate of how well
    the model will perform on future unseen data across different market conditions.

    After CV, a final model is trained on ALL available data for deployment.

Prerequisites:
    1. python build_dataset.py --refresh
    2. python dataset/prescale.py

How to run:
    python transformer_final/train_transformer.py
"""

import os
import sys
import joblib
import logging

import numpy as np
import torch
import torch.utils.checkpoint as grad_ckpt
import mlflow
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# MLflow 3.x changed its default tracking store to SQLite, which requires a
# database file that does not exist unless explicitly created. Setting this to
# the flat file store means all metrics and params are written directly into
# the mlruns/ folder with no database setup required.
mlflow.set_tracking_uri("file:./mlruns")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import check_feature_array, check_train_test_split


# HYPERPARAMETERS

# Forecast horizons: label -> number of trading days ahead to predict
HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
N_HORIZONS = len(HORIZONS)
N_FEATS = 36  # must match the feature count in build_dataset.py

# WINDOW: lookback period in trading days. 756 days = 3 years.
# The Transformer uses the entire 3 years of history as its context window.
WINDOW = 756

# PATCH_LEN: number of days in each patch fed to the Transformer.
# 16 days captures about 3 weeks of price action per patch token.
PATCH_LEN = 16

# STRIDE: how many days to step between patches.
# Reduced from 8 to 4 (Option 4) — doubles the number of patch tokens from
# 93 to 186, giving the attention mechanism finer temporal resolution.
# Each patch still covers 16 days but consecutive patches now overlap by 12 days
# instead of 8, so recent price structure is represented at a finer grain.
# n_patches = (756 - 16) // 4 + 1 = 186 tokens.
STRIDE = 4

# Sector embedding: the model learns a vector representation of each of the
# 11 GICS sectors. This lets it learn sector-specific patterns
# (e.g. tech stocks behave differently from utilities).
N_SECTORS = 11
SECTOR_DIM = 8  # dimension of the sector embedding vector

# EPOCHS: maximum training passes. Early stopping usually stops earlier.
EPOCHS = 100

# BATCH: number of windows per gradient update.
# 128 is safe on a 34 GB machine with this dataset.
BATCH = 128

# PATIENCE: early stopping patience.
PATIENCE = 10

# LR: learning rate. Kept at 1.5e-4 — the wider model (d_model=256) has more
# parameters so it benefits from a slightly conservative rate during warmup.
# OneCycleLR will handle the warmup and decay automatically.
LR = 1.5e-4

# QUANTILES: the three probability levels to predict simultaneously.
# p10 = 10th percentile (pessimistic), p50 = median, p90 = 90th (optimistic)
QUANTILES = [0.1, 0.5, 0.9]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk-forward CV fold boundaries — (train_end_date, test_end_date)
# Each fold's test window is exactly 1 calendar year
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


# EARLY STOPPING


class EarlyStopping:
    """
    Stops training when validation loss has not improved for `patience` epochs.
    Prevents overfitting and saves compute by stopping unnecessary epochs.
    """

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        """Resets the stopper for a new training run (e.g. a new CV fold)."""
        self.best = float("inf")
        self.counter = 0


# MODEL ARCHITECTURE


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).

    Standard Transformers have no built-in sense of where each token sits in
    the sequence. Without positional encoding, patch token 1 (days 1-16) and
    patch token 186 (days 741-756) look identical to the attention mechanism —
    the model cannot tell recent patches from old ones.

    RoPE fixes this by rotating the query and key vectors in each attention head
    by an angle that depends on the token's position. Tokens that are close
    together in the sequence will have similar rotation angles, so their dot
    product (attention score) is naturally higher. Tokens far apart have
    different rotations, making it harder for them to attend to each other
    unless they have learned to do so deliberately.

    Why RoPE instead of learnable positional embeddings?
        Learnable embeddings add a fixed learned vector to each position. They
        work well but do not generalise to sequence lengths longer than those
        seen during training. RoPE encodes relative position directly into the
        attention computation using rotation matrices, which generalise better
        and add zero extra parameters to the model.

    Implementation note:
        RoPE is applied inside a custom attention wrapper (RoPETransformerLayer)
        that replaces the standard TransformerEncoderLayer. The rotation is
        applied to the Q and K projections before computing attention scores,
        leaving V (values) unchanged.
    """

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        # Compute inverse frequencies for each pair of dimensions.
        # theta_i = 1 / (10000 ^ (2i / dim)) for i in [0, dim/2)
        # This creates a geometric sequence of frequencies — low frequencies
        # for the first dimensions (capture slow, long-range position changes)
        # and high frequencies for the last dimensions (capture fine-grained
        # position differences between nearby tokens).
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute the cos and sin tables for all positions up to max_seq_len.
        # This avoids recomputing them every forward pass.
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        """Returns (cos, sin) tables for positions 0..seq_len-1."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """
    Helper for applying the rotation: splits x into two halves and rotates.
    For a vector [x1, x2, ..., xd/2, xd/2+1, ..., xd], this returns
    [-xd/2+1, ..., -xd, x1, ..., xd/2].
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """
    Applies rotary positional encoding to query and key tensors.

    q, k shape: (batch, n_heads, seq_len, head_dim)
    cos, sin shape: (seq_len, head_dim) — broadcast across batch and heads

    The rotation formula is:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin

    This encodes absolute position as a rotation angle, and the dot product
    q_rotated · k_rotated depends only on the RELATIVE angle between positions,
    which is what we want — the model learns "how far apart are these two tokens"
    rather than "what are their absolute positions."
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RoPETransformerLayer(nn.Module):
    """
    A Transformer encoder layer with Rotary Positional Encoding applied to
    the query and key projections in the self-attention sublayer.

    This is a drop-in replacement for nn.TransformerEncoderLayer that adds
    RoPE without changing any other part of the architecture.

    The feedforward sublayer, LayerNorm positions (Pre-LN), dropout, and
    residual connections are identical to the standard PyTorch implementation.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
        rope: RotaryEmbedding,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rope = rope

        # Q, K, V projections and output projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Feedforward sublayer: two linear layers with GELU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

        # Pre-LayerNorm: applied BEFORE attention and feedforward (more stable)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5  # 1/sqrt(head_dim) attention scaling

    def forward(self, x, src_key_padding_mask=None):
        B, T, D = x.shape

        # Pre-LN: normalise before attention
        x_norm = self.norm1(x)

        # Project to Q, K, V and reshape to (B, nhead, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x_norm))
        K = split_heads(self.k_proj(x_norm))
        V = split_heads(self.v_proj(x_norm))

        # Apply RoPE to Q and K
        cos, sin = self.rope(T)
        Q, K = apply_rope(Q, K, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        # Weighted sum of values
        out = torch.matmul(attn, V)  # (B, nhead, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, d_model)
        out = self.out_proj(out)

        # Residual connection after attention
        x = x + self.drop_ff(out)

        # Pre-LN: normalise before feedforward
        x = x + self.drop_ff(self.ff(self.norm2(x)))
        return x


class RoPETransformerEncoder(nn.Module):
    """Stacks multiple RoPETransformerLayer instances."""

    def __init__(self, layer: RoPETransformerLayer, num_layers: int):
        super().__init__()
        # Each layer gets the SAME rope instance — they share positional frequencies.
        # This is correct: all layers see the same positions in the sequence.
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    layer.q_proj.in_features,
                    layer.nhead,
                    layer.ff[0].out_features,
                    layer.drop_attn.p,
                    layer.rope,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Converts a raw time series into a sequence of patch embeddings.

    Input:  (batch, time=756, features=36)
    Step 1: Permute to (batch, features=36, time=756)
    Step 2: Apply 1D convolution with kernel=16, stride=4
            Each patch covers 16 consecutive days. With stride=4, consecutive
            patches overlap by 12 days (instead of the original 8), giving
            186 tokens instead of 93. This finer granularity means the model
            can distinguish between price patterns that differ by just 4 days.
    Step 3: Permute output back to (batch, n_patches=186, d_model=256)

    Why Conv1d for patching?
        A convolution with kernel_size=patch_len and stride=stride is exactly
        the operation of "slide a window, project to embedding space."
        It is efficient, learnable, and handles the entire sequence in one pass.
    """

    def __init__(
        self, n_features=N_FEATS, d_model=256, patch_len=PATCH_LEN, stride=STRIDE
    ):
        super().__init__()
        self.proj = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        # x: (B, T, F) — permute to (B, F, T) for Conv1d, then back
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class PatchTST(nn.Module):
    """
    PatchTST: Patch-based Transformer for multi-horizon stock forecasting.

    Architecture (improved version):
        1. PatchEmbedding: (B, 756, 36) -> (B, 186, 256) patch tokens
           STRIDE reduced 8->4: 186 tokens instead of 93 (finer resolution)
           d_model increased 128->256: 2x wider embedding space
        2. Sector embedding: learnable sector vector added to all patch tokens
        3. Dropout: regularises the combined embedding
        4. RoPETransformerEncoder (6 layers, up from 4):
           - Rotary Positional Encoding in every attention layer
           - Pre-LayerNorm for numerical stability
           - dim_ff=1024 (up from 512) — wider feedforward sublayers
        5. LayerNorm: final normalisation
        6. Mean pooling: (B, 186, 256) -> (B, 256) summary vector
        7. Three output heads (deeper than before, scaled to d_model=256)

    Changes from v1:
        d_model:    128  -> 256   (Option 1: 2x wider — 4x more encoder params)
        dim_ff:     512  -> 1024  (scales with d_model)
        num_layers: 4    -> 6     (Option 3: deeper encoder)
        STRIDE:     8    -> 4     (Option 4: 186 patches instead of 93)
        Positional: None -> RoPE  (Option 2: relative position in attention)

    Parameter count: ~0.89M (v1) -> ~11M (v2)

    Gradient checkpointing (during training only):
        Discards intermediate activations during the forward pass and
        recomputes them during backward, saving ~40% VRAM at the cost of
        ~30% more compute. Disabled during evaluation.
    """

    def __init__(
        self,
        n_features=N_FEATS,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_ff=1024,
        dropout=0.3,
        n_sectors=N_SECTORS,
        sector_dim=8,
        n_quantiles=3,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_horizons = N_HORIZONS

        # Number of patch tokens with new stride
        n_patches = (WINDOW - PATCH_LEN) // STRIDE + 1  # (756-16)//4+1 = 186

        # Patch embedding: (B, 756, 36) -> (B, 186, 256)
        self.patch_embed = PatchEmbedding(n_features, d_model)

        # Sector conditioning
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, d_model)
        self.drop = nn.Dropout(dropout)

        # Rotary Positional Encoding — shared across all encoder layers.
        # max_seq_len=256 is safely above our 186 tokens.
        rope = RotaryEmbedding(dim=d_model // nhead, max_seq_len=256)

        # Build the first layer spec; RoPETransformerEncoder will replicate it.
        first_layer = RoPETransformerLayer(d_model, nhead, dim_ff, dropout, rope)
        self.encoder = RoPETransformerEncoder(first_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        def _head(out_size, deep=False):
            """Output MLP head for one forecast horizon. All heads are deeper
            now that d_model=256 gives more signal to work with."""
            if deep:
                # 3-layer MLP for 6-month head: wider intermediate layers
                return nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, out_size),
                )
            # 2-layer MLP for 1-week and 1-month heads
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, out_size),
            )

        self.head_1w = _head(n_quantiles)
        self.head_1m = _head(n_quantiles)
        self.head_6m = _head(n_quantiles, deep=True)

    def encode(self, x, sector):
        """
        Shared encoder: (B, 756, 36) -> (B, 256) summary vector.

        1. Patch embedding -> (B, 186, 256)
        2. Add sector embedding (broadcast across all 186 patches)
        3. Dropout
        4. 6 RoPE Transformer encoder layers
        5. LayerNorm
        6. Mean pool across patches -> (B, 256)
        """
        patches = self.patch_embed(x)  # (B, 186, 256)
        sec_emb = self.sector_proj(self.sector_embed(sector)).unsqueeze(
            1
        )  # (B, 1, 256)
        patches = self.drop(patches + sec_emb)

        if self.training:
            out = grad_ckpt.checkpoint(self.encoder, patches, use_reentrant=False)
        else:
            out = self.encoder(patches)

        out = self.norm(out)
        return out.mean(dim=1)  # (B, 256)

    def forward(self, x, sector):
        """
        Returns (B, 3, 3): three horizons × three quantiles.
        output[i, 0, 1] = p50 for 1-week horizon, sample i.
        """
        enc = self.encode(x, sector)
        return torch.stack(
            [
                self.head_1w(enc),
                self.head_1m(enc),
                self.head_6m(enc),
            ],
            dim=1,
        )


# LOSS FUNCTION


def pinball_loss(pred, target, quantiles=QUANTILES):
    """
    Quantile (pinball) loss for simultaneous multi-quantile regression.

    For each quantile q in [0.1, 0.5, 0.9]:
        L_q = q * max(target - pred, 0) + (1 - q) * max(pred - target, 0)

    This is asymmetric: at q=0.9, under-predictions are penalised 9x more
    than over-predictions, pushing the model to output values where 90% of
    true values fall below the prediction.

    pred:   (B, 3 horizons, 3 quantiles)
    target: (B, 3 horizons) — the true return values
    """
    q = torch.tensor(quantiles, device=pred.device)
    # Expand target to match pred shape: (B, 3) -> (B, 3, 3)
    target = target.unsqueeze(-1).expand_as(pred)
    err = target - pred
    # torch.max takes the element-wise max of two tensors:
    # max(q*err, (q-1)*err) gives the asymmetric pinball loss for each quantile
    return torch.max(q * err, (q - 1) * err).mean()


# TRAINING LOOP (one epoch)


def run_epoch(model, dl, opt, amp, scheduler, train=True, desc=""):
    """
    Runs one pass through the dataset (training or validation).

    In TRAINING mode (train=True):
        - Enables gradients and dropout
        - Uses AMP (Automatic Mixed Precision) to compute in float16 where safe
        - Scales gradients, clips them, then updates weights

    In VALIDATION mode (train=False):
        - Disables gradients (saves memory, speeds up)
        - Computes loss but does not update weights

    NaN/Inf guard:
        Some batches may contain extreme feature values that cause the loss to
        be NaN or Inf. The guard detects this and skips the batch entirely,
        preventing corrupted gradients from being applied to the model weights.

    Returns: average pinball loss per sample for this epoch.
    """
    model.train() if train else model.eval()
    total = 0.0
    n_samples = 0
    n_skipped = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        bar = tqdm(dl, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
        for xb, yb, sb in bar:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)

            if train:
                opt.zero_grad()

                # autocast("cuda") automatically uses float16 for matrix
                # multiplications (where it is safe and fast) and float32
                # for operations that need higher precision.
                with autocast("cuda"):
                    pred = model(xb, sb)  # (B, 3, 3)
                    loss = pinball_loss(pred, yb)  # scalar

                # Skip batch if the loss is NaN or Inf.
                # NaN loss means a batch caused numerical overflow inside the
                # Transformer. Skipping prevents corrupted gradients from
                # being applied to model weights.
                if not torch.isfinite(loss):
                    n_skipped += 1
                    bar.set_postfix(loss="NaN-skip")
                    continue

                # Backpropagate with AMP gradient scaling.
                # Scaling prevents float16 underflow during backpropagation.
                amp.scale(loss).backward()
                amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
                amp.step(opt)
                amp.update()
                scheduler.step()

            else:
                # Validation: forward pass only
                with torch.no_grad():
                    pred = model(xb, sb)
                loss = pinball_loss(pred, yb)
                if not torch.isfinite(loss):
                    n_skipped += 1
                    bar.set_postfix(loss="NaN-skip")
                    continue

            batch_loss = loss.item()
            total += batch_loss * xb.size(0)
            n_samples += xb.size(0)
            bar.set_postfix(loss=f"{batch_loss:.4f}")

    if n_skipped > 0:
        logger.warning("Skipped %d NaN/Inf batches in this epoch", n_skipped)
    if n_samples == 0:
        return float("nan")
    return total / n_samples


# EVALUATION


def evaluate(model, dl, sc_ret, LC_arr, Y_px_arr):
    """
    Evaluates the model on a validation or test set and returns per-horizon metrics.

    Steps:
      1. Run all batches through the model to collect raw predictions
      2. Extract the p50 (median) quantile prediction — shape (N, 3)
      3. Reverse the StandardScaler transform to get predicted RETURNS
      4. Convert returns to predicted PRICES: price = last_close * (1 + return)
      5. Compute MAE and Directional Accuracy against true future prices

    NaN guard:
        If model weights contain NaN (from a bad training batch that slipped
        through the guard in run_epoch), all predictions will be NaN. This
        guard detects the problem and returns sentinel NaN metrics instead of
        crashing the sklearn metric functions.
    """
    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _, sb in dl:
            preds_list.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N, 3 horizons, 3 quantiles)

    # Guard against NaN predictions from corrupted model weights
    if not np.isfinite(preds).all():
        nan_pct = (~np.isfinite(preds)).mean() * 100
        logger.warning(
            "evaluate(): %.1f%% of predictions are NaN/Inf. Returning NaN metrics.",
            nan_pct,
        )
        metrics = {
            name: {"mae": float("nan"), "dir_acc": float("nan")} for name in HORIZONS
        }
        return metrics, preds

    # p50_s: the median predicted RETURN (scaled, not in dollars yet)
    p50_s = preds[:, :, 1]  # (N, 3) — index 1 is the p50 quantile
    # Reverse the StandardScaler to get returns in their original scale
    p50_ret = sc_ret.inverse_transform(p50_s)
    # Convert returns to prices: predicted_price = last_close * (1 + return)
    p50_px = LC_arr[:, None] * (1 + p50_ret)

    metrics = {}
    for idx, name in enumerate(HORIZONS):
        t = Y_px_arr[:, idx]  # true future prices
        p = p50_px[:, idx]  # predicted future prices (median)
        mae = mean_absolute_error(t, p)
        dir_acc = np.mean(np.sign(p - LC_arr) == np.sign(t - LC_arr)) * 100
        metrics[name] = {"mae": mae, "dir_acc": dir_acc}

    return metrics, preds


# DATASET: memory-mapped, Windows-compatible


class MmapDataset(torch.utils.data.Dataset):
    """
    Reads pre-scaled float16 windows from X_756_scaled.npy one item at a time.

    Why store a file PATH instead of the array?
        DataLoader spawns worker processes on Windows via multiprocessing.
        Each worker receives a pickled (serialised) copy of the Dataset.
        numpy memory-mapped arrays cannot be pickled on Windows.
        Storing only the file path (a plain string) avoids this — each worker
        opens its own mmap handle when __getitem__ is first called.

    Why float16?
        X_756_scaled.npy is 35.8 GB in float16 vs 71.5 GB in float32.
        Half the disk size means half the SSD reads per epoch and half the
        OS page-cache footprint. The precision error (< 0.03%) is negligible.

    Dataset contents:
        X_path   : str         — path to X_756_scaled.npy
        X_shape  : tuple       — (N, 756, 36), used for mmap reopening
        Y_scaled : float32 (N, 3)  — pre-scaled return targets
        sec      : int32   (N,)    — GICS sector index per window
        indices  : int64   (M,)    — which rows of the full dataset this split uses
    """

    def __init__(self, X_path, X_shape, Y_scaled, sec, indices):
        self.X_path = X_path  # str — pickleable
        self.X_shape = X_shape  # tuple — pickleable
        self.Y = Y_scaled  # float32 ndarray — pickleable
        self.sec = sec  # int32  ndarray  — pickleable
        self.idx = indices  # int64  ndarray  — pickleable
        self._X = None  # mmap opened lazily per worker process

    def _open_mmap(self):
        """Opens the memory-mapped file on first access in this worker."""
        if self._X is None:
            self._X = np.load(self.X_path, mmap_mode="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        """
        Returns one (input_window, target_returns, sector_index) tuple.

        x shape: (756, 36)  float32 — 3 years of 36 features, pre-scaled
        y shape: (3,)       float32 — 3 target returns (scaled)
        s:       scalar int — sector index [0, 10]
        """
        self._open_mmap()
        real_i = self.idx[i]

        # Cast float16 -> float32 for GPU computation (one small copy per item)
        x = self._X[real_i].astype(np.float32)  # (756, 36)
        y = self.Y[real_i].copy()  # (3,) float32
        s = int(self.sec[real_i])

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(s, dtype=torch.long),
        )


def make_loaders(X_path, X_shape, Y_scaled, sec, indices, batch, shuffle):
    """
    Creates a DataLoader around MmapDataset.

    num_workers=2: two background processes load and prefetch batches while
        the GPU processes the current batch. Without workers, the GPU waits
        idle for the CPU to load each batch.

    pin_memory=True: allocates batch tensors in pinned (page-locked) RAM.
        This enables direct DMA transfer to GPU, skipping an extra CPU copy.
        Only effective with num_workers > 0 (async prefetch).

    persistent_workers=True: worker processes stay alive between epochs.
        Avoids the overhead of killing and re-forking workers each epoch.
    """
    ds = MmapDataset(X_path, X_shape, Y_scaled, sec, indices)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


# TRAIN ONE CV FOLD


def train_fold(
    X_path,
    X_shape,
    tr_idx,
    te_idx,
    Y_scaled,
    sec,
    sc_ret,
    LC_te,
    Y_px_te,
    fold_name,
    n_steps,
):
    """
    Trains a fresh PatchTST model on one walk-forward CV fold.

    Each fold gets its own model (trained from random initialisation), its own
    optimiser, and its own learning rate scheduler. After training, the best
    checkpoint is loaded and evaluated on the fold's test period.

    Parameters:
        X_path   : path to X_756_scaled.npy (passed to MmapDataset as a string)
        X_shape  : shape tuple of X
        tr_idx   : training window indices for this fold
        te_idx   : test window indices for this fold
        Y_scaled : return targets scaled with this fold's sc_ret
        sc_ret   : per-fold StandardScaler for returns (for inverse_transform)
        LC_te    : last close prices for the test windows (for price conversion)
        Y_px_te  : true future prices for the test windows (for evaluation)
    """
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
    amp = GradScaler("cuda")

    tr_dl = make_loaders(X_path, X_shape, Y_scaled, sec, tr_idx, BATCH, shuffle=True)
    te_dl = make_loaders(X_path, X_shape, Y_scaled, sec, te_idx, BATCH, shuffle=False)

    best_val = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, f"_fold_{fold_name}.pth")

    epoch_bar = tqdm(
        range(1, EPOCHS + 1), desc=f"[{fold_name}]", unit="ep", dynamic_ncols=True
    )
    for ep in epoch_bar:
        tr_loss = run_epoch(
            model, tr_dl, opt, amp, sched, train=True, desc=f"  train ep{ep}"
        )
        val_loss = run_epoch(
            model, te_dl, None, None, None, train=False, desc=f"  val   ep{ep}"
        )

        epoch_bar.set_postfix(
            train=f"{tr_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val:.4f}"
        )
        logger.info(
            "[%s] Epoch %d/%d  train=%.4f  val=%.4f",
            fold_name,
            ep,
            EPOCHS,
            tr_loss,
            val_loss,
        )

        # Log per-epoch losses to MLflow so the loss curve is visible in the UI
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    f"{fold_name}_train_loss": float(tr_loss),
                    f"{fold_name}_val_loss": float(val_loss)
                    if val_loss == val_loss
                    else -1.0,
                },
                step=ep,
            )

        # Save checkpoint if validation loss improved and is a valid number.
        # If val_loss is NaN (all batches were skipped), the checkpoint is NOT
        # updated — but we save a fallback on epoch 1 so torch.load never fails.
        if torch.isfinite(torch.tensor(val_loss)) and val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        elif not os.path.exists(ckpt_path):
            torch.save(model.state_dict(), ckpt_path)

        if stopper.step(val_loss):
            logger.info("[%s] Early stopping at epoch %d", fold_name, ep)
            break

    # Load the best checkpoint (lowest validation loss) for final evaluation
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    )
    metrics, _ = evaluate(model, te_dl, sc_ret, LC_te, Y_px_te)

    logger.info("[%s] Results:", fold_name)
    for h, m in metrics.items():
        logger.info("  %s: MAE=%.2f  DirAcc=%.1f%%", h, m["mae"], m["dir_acc"])

    return model, metrics


# MAIN SCRIPT

if __name__ == "__main__":
    DATASET_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
    )

    # -------------------------------------------------------------------------
    # DATASET LOADING
    #
    # This script requires files produced by dataset/prescale.py:
    #
    #   X_756_scaled.npy  — float16, shape (657035, 756, 36), pre-scaled features
    #   Y_ret_756.npy     — float32, shape (657035, 3), return targets (pre-scaled)
    #   Y_px_756.npy      — float32, shape (657035, 3), raw future prices
    #   LC_756.npy        — float32, shape (657035,), last close per window
    #   sectors_756.npy   — int32,   shape (657035,), GICS sector index
    #   dates_756.npy     — object,  shape (657035,), window end date strings
    #   scaler_feat.pkl   — StandardScaler fitted on features
    #   scaler_ret.pkl    — StandardScaler fitted on returns
    #
    # Why these files instead of the raw windows_756.npz?
    #   windows_756.npz is a 71 GB ZIP archive. np.load(..., mmap_mode='r') on a
    #   ZIP does not create a real OS mmap — numpy reads the entire entry into RAM.
    #   On a 34 GB machine, this causes an immediate out-of-memory crash.
    #
    #   prescale.py extracts and scales X into a standalone float16 .npy file.
    #   np.load on a .npy file IS a true OS mmap: the OS loads only the pages
    #   (rows) that are actually accessed, keeping RAM usage near zero.
    # -------------------------------------------------------------------------
    X_SCALED = os.path.join(DATASET_DIR, "X_756_scaled.npy")
    Y_RET_NPY = os.path.join(DATASET_DIR, "Y_ret_756.npy")
    Y_PX_NPY = os.path.join(DATASET_DIR, "Y_px_756.npy")
    LC_NPY = os.path.join(DATASET_DIR, "LC_756.npy")
    SEC_NPY = os.path.join(DATASET_DIR, "sectors_756.npy")
    DATES_NPY = os.path.join(DATASET_DIR, "dates_756.npy")
    SC_FEAT_PKL = os.path.join(DATASET_DIR, "scaler_feat.pkl")
    SC_RET_PKL = os.path.join(DATASET_DIR, "scaler_ret.pkl")

    required = [X_SCALED, Y_RET_NPY, Y_PX_NPY, LC_NPY, SEC_NPY, SC_FEAT_PKL, SC_RET_PKL]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Pre-scaled dataset files not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun first:\n"
            "  python build_dataset.py --refresh\n"
            "  python dataset/prescale.py"
        )

    logger.info("Loading pre-scaled dataset (float16 mmap)...")

    # X stays on disk — only rows accessed by the DataLoader are pulled into RAM
    X = np.load(X_SCALED, mmap_mode="r")  # (657035, 756, 36) float16
    Y_ret = np.load(Y_RET_NPY)  # (657035, 3) float32 — scaled returns
    Y_px = np.load(Y_PX_NPY)  # (657035, 3) float32 — raw prices
    LC = np.load(LC_NPY)  # (657035,) float32
    sectors = np.load(SEC_NPY).astype(np.int32)  # (657035,) int32
    sc_feat = joblib.load(SC_FEAT_PKL)
    sc_ret = joblib.load(SC_RET_PKL)

    has_meta = os.path.exists(DATES_NPY)
    if has_meta:
        dates = np.load(DATES_NPY, allow_pickle=True)  # (657035,) date strings
        logger.info("Date and sector metadata available.")
    else:
        logger.warning("dates_756.npy not found — re-run dataset/prescale.py")
        dates = None

    assert X.shape[1] == WINDOW and X.shape[2] == N_FEATS, (
        f"Shape mismatch: X is {X.shape}, expected (N, {WINDOW}, {N_FEATS})."
    )
    assert X.dtype == np.float16, f"Expected float16 X, got {X.dtype}"

    logger.info(
        "Dataset: %d windows, %d features, %d horizons  [float16 mmap, %.1f GB]",
        X.shape[0],
        N_FEATS,
        N_HORIZONS,
        X.nbytes / 1e9,
    )

    # Y_ret from prescale.py is globally scaled (sc_ret fitted on all N windows).
    # For CV folds we need per-fold scaling to avoid leaking future data into the
    # training normalisation. We re-fit sc_ret on each fold's training targets below.
    Y_ret_raw = np.load(Y_RET_NPY)  # same file, loaded again for per-fold refitting

    # -------------------------------------------------------------------------
    # WALK-FORWARD CROSS-VALIDATION
    # -------------------------------------------------------------------------
    mlflow.set_experiment("stock-forecasting-transformer")
    fold_metrics = {h: {"mae": [], "dir_acc": []} for h in HORIZONS}

    if has_meta:
        for fold_idx, (test_start, test_end) in enumerate(WF_FOLDS, 1):
            fold_name = f"fold{fold_idx}"

            # Skip folds that are already trained — allows resuming a crashed run
            # without re-training completed folds from scratch.
            ckpt_path_check = os.path.join(SAVE_DIR, f"_fold_{fold_name}.pth")
            if os.path.exists(ckpt_path_check):
                logger.info(
                    "Fold %d checkpoint exists — skipping (already trained).", fold_idx
                )
                continue

            # Build boolean masks: which windows fall in the training vs test period
            tr_mask = dates < test_start
            te_mask = (dates >= test_start) & (dates < test_end)

            if tr_mask.sum() < 100 or te_mask.sum() < 10:
                logger.warning("Fold %d skipped — not enough data.", fold_idx)
                continue

            logger.info(
                "Fold %d: train=%d  test=%d  [%s -> %s]",
                fold_idx,
                tr_mask.sum(),
                te_mask.sum(),
                test_start,
                test_end,
            )

            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
            Px_te = Y_px[te_idx]
            LC_te = LC[te_idx]

            # Fit a fold-specific return scaler on TRAINING data only.
            # Using the global scaler would leak the validation period's return
            # distribution into the training normalisation.
            sc_ret_fold = StandardScaler().fit(Y_ret_raw[tr_idx])
            Y_scaled_fold = sc_ret_fold.transform(Y_ret_raw).astype(np.float32)

            n_steps = (len(tr_idx) // BATCH + 1) * EPOCHS

            with mlflow.start_run(run_name=f"patchtst_{fold_name}", nested=True):
                mlflow.log_params(
                    {
                        # Fold-specific data split info
                        "fold": fold_idx,
                        "train_size": int(tr_mask.sum()),
                        "test_size": int(te_mask.sum()),
                        "test_start": test_start,
                        "test_end": test_end,
                        # Architecture params (same for every fold)
                        "model": "PatchTST_v2",
                        "d_model": 256,
                        "nhead": 8,
                        "num_layers": 6,
                        "dim_ff": 1024,
                        "dropout": 0.3,
                        "positional_encoding": "RoPE",
                        "patch_len": PATCH_LEN,
                        "stride": STRIDE,
                        "window": WINDOW,
                        "n_features": N_FEATS,
                        "batch_size": BATCH,
                        "lr": LR,
                        "loss": "pinball_quantile",
                        "patience": PATIENCE,
                    }
                )
                _, metrics = train_fold(
                    X_SCALED,
                    X.shape,
                    tr_idx,
                    te_idx,
                    Y_scaled_fold,
                    sectors,
                    sc_ret_fold,
                    LC_te,
                    Px_te,
                    fold_name,
                    n_steps,
                )
                for h, m in metrics.items():
                    fold_metrics[h]["mae"].append(m["mae"])
                    fold_metrics[h]["dir_acc"].append(m["dir_acc"])
                    mlflow.log_metrics(
                        {
                            f"mae_{h}": float(m["mae"]),
                            f"dir_acc_{h}": float(m["dir_acc"]),
                        }
                    )

        # Print CV summary — average performance across all test periods
        logger.info("\n=== Walk-Forward CV Summary ===")
        for h in HORIZONS:
            maes = fold_metrics[h]["mae"]
            dir_accs = fold_metrics[h]["dir_acc"]
            if maes:
                logger.info(
                    "  %s: MAE=%.2f +/- %.2f  DirAcc=%.1f%% +/- %.1f%%",
                    h,
                    np.mean(maes),
                    np.std(maes),
                    np.mean(dir_accs),
                    np.std(dir_accs),
                )

    # -------------------------------------------------------------------------
    # FINAL PRODUCTION MODEL
    #
    # After CV proves the architecture works, we train one last model on ALL
    # available data (no holdout). This gives the model the most information
    # possible before deployment.
    #
    # The CV fold models are discarded — they existed only to produce honest
    # performance estimates. This final model is what app.py loads for inference.
    # -------------------------------------------------------------------------
    logger.info("\n=== Training final production model on all data ===")

    all_idx = np.arange(len(X))
    final_dl = make_loaders(
        X_SCALED, X.shape, Y_ret, sectors, all_idx, BATCH, shuffle=True
    )

    logger.info("Final model: %d windows, %d batches/epoch", len(X), len(final_dl))

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
    final_amp = GradScaler("cuda")
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
                "d_model": 256,
                "nhead": 8,
                "num_layers": 6,
                "dim_ff": 1024,
                "dropout": 0.3,
                "positional_encoding": "RoPE",
                "sector_dim": SECTOR_DIM,
                "batch_size": BATCH,
                "lr": LR,
                "loss": "pinball_quantile",
                "horizons": list(HORIZONS.keys()),
                "n_folds_cv": len(WF_FOLDS) if has_meta else 0,
            }
        )
        # Log CV summary metrics so we can compare against the final model
        for h in HORIZONS:
            if fold_metrics[h]["mae"]:
                mlflow.log_metric(f"cv_mae_{h}", float(np.mean(fold_metrics[h]["mae"])))
                mlflow.log_metric(
                    f"cv_dir_acc_{h}", float(np.mean(fold_metrics[h]["dir_acc"]))
                )

        epoch_bar = tqdm(
            range(1, EPOCHS + 1), desc="[final]", unit="ep", dynamic_ncols=True
        )
        for ep in epoch_bar:
            tr_loss = run_epoch(
                final_model,
                final_dl,
                final_opt,
                final_amp,
                final_sched,
                train=True,
                desc=f"  train ep{ep}",
            )
            epoch_bar.set_postfix(loss=f"{tr_loss:.4f}", best=f"{best_loss:.4f}")
            logger.info("Final  Epoch %d/%d  train=%.4f", ep, EPOCHS, tr_loss)
            mlflow.log_metric("train_loss", float(tr_loss), step=ep)

            if tr_loss < best_loss:
                best_loss = tr_loss
                torch.save(final_model.state_dict(), ckpt)

            if final_stopper.step(tr_loss):
                logger.info("Stopping final training at epoch %d", ep)
                break

        mlflow.log_artifact(ckpt)

    # Save scalers alongside model weights for inference
    joblib.dump(sc_feat, os.path.join(SAVE_DIR, "scaler_feat.pkl"))
    joblib.dump(sc_ret, os.path.join(SAVE_DIR, "scaler_ret.pkl"))

    # Metadata tells app.py how to prepare inputs and interpret outputs
    joblib.dump(
        {
            "window": WINDOW,
            "patch_len": PATCH_LEN,
            "stride": STRIDE,
            "horizons": HORIZONS,
            "n_features": N_FEATS,
            "model_type": "PatchTST_v2",
            "n_quantiles": 3,
            "quantiles": QUANTILES,
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_ff": 1024,
            "positional_encoding": "RoPE",
        },
        os.path.join(SAVE_DIR, "transformer_meta.pkl"),
    )

    logger.info("Production model saved to %s", ckpt)
    logger.info("Training complete.")
