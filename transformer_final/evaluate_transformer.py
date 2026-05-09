"""
evaluate_transformer.py

Loads the trained transformer_multi_horizon.pth and computes validation metrics
on the walk-forward fold test windows without retraining anything.

Run from the transformer_final/ folder:
    python evaluate_transformer.py

The model definition here must exactly match train_transformer.py.
Architecture (v2):
    d_model=256, nhead=8, num_layers=6, dim_ff=1024
    STRIDE=4 -> 186 patches, RoPE positional encoding
"""

import os
import sys
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SAVE_DIR), "dataset")

HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
N_HORIZONS = len(HORIZONS)
N_FEATS = 36
WINDOW = 756
PATCH_LEN = 16
STRIDE = 4  # reduced from 8 -> 186 patches
N_SECTORS = 11
SECTOR_DIM = 8
QUANTILES = [0.1, 0.5, 0.9]
BATCH = 256

WF_FOLDS = [
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
]


# Model definition — must match train_transformer.py exactly


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RoPETransformerLayer(nn.Module):
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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ff = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x, src_key_padding_mask=None):
        B, T, D = x.shape
        x_norm = self.norm1(x)

        def split_heads(t):
            return t.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x_norm))
        K = split_heads(self.k_proj(x_norm))
        V = split_heads(self.v_proj(x_norm))
        cos, sin = self.rope(T)
        Q, K = apply_rope(Q, K, cos, sin)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.drop_attn(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop_ff(self.out_proj(out))
        x = x + self.drop_ff(self.ff(self.norm2(x)))
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, layer: RoPETransformerLayer, num_layers: int):
        super().__init__()
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
    def __init__(
        self, n_features=N_FEATS, d_model=256, patch_len=PATCH_LEN, stride=STRIDE
    ):
        super().__init__()
        self.proj = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x):
        return self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class PatchTST(nn.Module):
    def __init__(
        self,
        n_features=N_FEATS,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_ff=1024,
        dropout=0.3,
        n_sectors=N_SECTORS,
        sector_dim=SECTOR_DIM,
        n_quantiles=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.sector_embed = nn.Embedding(n_sectors, sector_dim)
        self.sector_proj = nn.Linear(sector_dim, d_model)
        self.drop = nn.Dropout(dropout)
        rope = RotaryEmbedding(dim=d_model // nhead, max_seq_len=256)
        first_layer = RoPETransformerLayer(d_model, nhead, dim_ff, dropout, rope)
        self.encoder = RoPETransformerEncoder(first_layer, num_layers=num_layers)
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
        self.head_6m = _head(n_quantiles, deep=True)

    def encode(self, x, sector):
        patches = self.patch_embed(x)
        sec_emb = self.sector_proj(self.sector_embed(sector)).unsqueeze(1)
        patches = self.drop(patches + sec_emb)
        out = self.encoder(patches)
        return self.norm(out).mean(dim=1)

    def forward(self, x, sector):
        enc = self.encode(x, sector)
        return torch.stack(
            [self.head_1w(enc), self.head_1m(enc), self.head_6m(enc)], dim=1
        )


# Dataset


class MmapDataset(torch.utils.data.Dataset):
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
        x = self._X[real_i].astype(np.float32)
        y = self.Y[real_i].copy()
        s = int(self.sec[real_i])
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(s, dtype=torch.long),
        )


def run_fold_eval(model, te_idx, Y_scaled, sectors, sc_ret, LC, Y_px, fold_name):
    """Evaluate on one fold's test window and return per-horizon metrics."""
    ds = MmapDataset(
        os.path.join(DATASET_DIR, "X_756_scaled.npy"),
        Y_scaled,
        sectors,
        te_idx,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _, sb in tqdm(dl, desc=f"  {fold_name}", leave=False):
            preds_list.append(model(xb.to(DEVICE), sb.to(DEVICE)).cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N, 3, 3)

    # Exclude windows where predictions are NaN/Inf
    nan_mask = ~np.isfinite(preds).all(axis=(1, 2))
    if nan_mask.any():
        pct = nan_mask.mean() * 100
        print(
            f"  WARNING: {nan_mask.sum()} predictions ({pct:.1f}%) are NaN — excluded."
        )
        valid = ~nan_mask
        preds = preds[valid]
        te_idx = te_idx[valid]

    if len(preds) == 0:
        print(f"  ERROR: all predictions for {fold_name} are NaN.")
        return {
            name: {"mae": float("nan"), "dir_acc": float("nan")} for name in HORIZONS
        }

    p50_s = preds[:, :, 1]
    p50_ret = sc_ret.inverse_transform(p50_s)
    LC_te = LC[te_idx]
    p50_px = LC_te[:, None] * (1 + p50_ret)
    Y_px_te = Y_px[te_idx]

    results = {}
    for idx, name in enumerate(HORIZONS):
        t = Y_px_te[:, idx]
        p = p50_px[:, idx]
        mae = mean_absolute_error(t, p)
        dir_acc = np.mean(np.sign(p - LC_te) == np.sign(t - LC_te)) * 100
        results[name] = {"mae": mae, "dir_acc": dir_acc}
    return results


def main():
    print(f"Device: {DEVICE}")

    X_SCALED = os.path.join(DATASET_DIR, "X_756_scaled.npy")
    Y_RET = os.path.join(DATASET_DIR, "Y_ret_756.npy")
    Y_PX = os.path.join(DATASET_DIR, "Y_px_756.npy")
    LC_PATH = os.path.join(DATASET_DIR, "LC_756.npy")
    DATES = os.path.join(DATASET_DIR, "dates_756.npy")
    SC_RET = os.path.join(DATASET_DIR, "scaler_ret.pkl")
    CKPT = os.path.join(SAVE_DIR, "transformer_multi_horizon.pth")

    for p in [X_SCALED, Y_RET, Y_PX, LC_PATH, DATES, SC_RET, CKPT]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Loading dataset...")
    dates = np.load(DATES, allow_pickle=True)
    Y_ret = np.load(Y_RET)
    Y_px = np.load(Y_PX)
    LC = np.load(LC_PATH)
    sectors = np.load(os.path.join(DATASET_DIR, "sectors_756.npy")).astype(np.int32)
    sc_ret = joblib.load(SC_RET)

    print("Loading model weights...")
    model = PatchTST().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params / 1e6:.1f}M parameters")
    print()

    all_mae = {h: [] for h in HORIZONS}
    all_dir_acc = {h: [] for h in HORIZONS}

    for fold_idx, (test_start, test_end) in enumerate(WF_FOLDS, 1):
        fold_name = f"fold{fold_idx}"
        tr_mask = dates < test_start
        te_mask = (dates >= test_start) & (dates < test_end)

        if te_mask.sum() < 10:
            print(f"Fold {fold_idx}: not enough test windows — skipping")
            continue

        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        print(
            f"Fold {fold_idx}: [{test_start} -> {test_end}]  "
            f"train={len(tr_idx):,}  test={len(te_idx):,}"
        )

        sc_ret_fold = StandardScaler().fit(Y_ret[tr_idx])
        Y_scaled_fold = sc_ret_fold.transform(Y_ret).astype(np.float32)

        results = run_fold_eval(
            model,
            te_idx,
            Y_scaled_fold,
            sectors,
            sc_ret_fold,
            LC,
            Y_px,
            fold_name,
        )
        for h, m in results.items():
            all_mae[h].append(m["mae"])
            all_dir_acc[h].append(m["dir_acc"])
            print(f"  {h}: MAE={m['mae']:.2f}  DirAcc={m['dir_acc']:.1f}%")
        print()

    print("Transformer — Walk-Forward CV Summary")
    print("(evaluated using production model weights)")
    for h in HORIZONS:
        if all_mae[h]:
            print(
                f"  {h}:  MAE = {np.mean(all_mae[h]):.2f} +/- "
                f"{np.std(all_mae[h]):.2f}  "
                f"DirAcc = {np.mean(all_dir_acc[h]):.1f}% +/- "
                f"{np.std(all_dir_acc[h]):.1f}%"
            )


if __name__ == "__main__":
    main()
