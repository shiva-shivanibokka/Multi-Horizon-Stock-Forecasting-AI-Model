"""
train_transformer.py
Trains a Temporal Fusion Transformer (TFT) on S&P 500 stock data.

Why TFT instead of a plain Transformer?
----------------------------------------
A standard Transformer treats all inputs the same — every feature at every
time step goes through the same self-attention process. This works well for
text, where all tokens are of the same type.

Stock market data is different. You have three very different kinds of inputs:
  - Static information: the company ticker, sector, and region. These never
    change over time. A company being in the Technology sector is always true.
  - Past observations: the historical prices, volume, RSI, MACD etc. These are
    things we observed in the past and can use for training.
  - Future known inputs: things we know about the future in advance — like the
    day of week, month, quarter, and whether tomorrow is a market holiday.

The Temporal Fusion Transformer, developed by Google Research (Lim et al. 2020),
was designed specifically to handle this distinction. It:
  1. Uses a Variable Selection Network to learn which features actually matter
     (and ignores the ones that don't — automatically)
  2. Uses an LSTM encoder to capture short-term sequential patterns
  3. Uses sparse self-attention to find long-range patterns across the full window
  4. Produces calibrated quantile forecasts (p10, p50, p90) natively — not
     through MC Dropout approximation like the standard Transformer

This makes TFT the strongest architecture for multi-horizon financial forecasting.
It consistently outperforms standard Transformers, LSTMs, and ARIMA-family models
on the M5 forecasting competition and several financial benchmarks.

Run:
    python train_transformer.py
    (run python build_dataset.py from the project root first)
"""

import os
import sys
import math
import warnings
import joblib
import logging

import numpy as np
import pandas as pd
import mlflow
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

warnings.filterwarnings("ignore")
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
MAX_H = max(HORIZONS.values())
WINDOW = 756
BATCH_SIZE = 128  # increased from 64 — larger batches keep the GPU busier
MAX_EPOCHS = 30
PATIENCE = 5
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

# Number of CPU workers for data loading.
# On Windows, multiprocessing with DataLoader can cause issues so we use 0.
# On Linux/Mac you can set this to 4 or 8 for faster loading.
NUM_WORKERS = 0
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_raw_csvs(raw_dir: str) -> pd.DataFrame:
    """
    Loads the per-ticker CSV files saved by build_dataset.py and combines
    them into one long DataFrame in the format TFT expects.

    TFT needs a long-format DataFrame where each row is one day for one ticker.
    We add time-based features (day of week, month, quarter) as future known inputs
    because the calendar is always known in advance — even for future dates.
    """
    frames = []
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            "Run  python build_dataset.py  from the project root first."
        )

    for fname in csv_files:
        sym = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(raw_dir, fname), index_col=0, parse_dates=True)
        df = df.dropna()

        if len(df) < WINDOW + MAX_H:
            continue

        df["ticker"] = sym
        df["time_idx"] = range(len(df))

        # Calendar features — known in advance for future dates
        df["day_of_week"] = df.index.dayofweek.astype(str)
        df["month"] = df.index.month.astype(str)
        df["quarter"] = df.index.quarter.astype(str)

        # Log-transform Close for a more stationary target
        df["log_close"] = np.log(df["Close"])

        frames.append(df)

    if not frames:
        raise RuntimeError("No tickers had enough data after filtering.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d tickers, %d total rows.", len(frames), len(combined))
    return combined


def build_tft_datasets(df: pd.DataFrame):
    """
    Builds PyTorch Forecasting TimeSeriesDataSet objects for training and validation.

    We predict 'log_close' (log price) at each horizon. At inference time the
    app converts predictions back to actual prices with exp().

    The dataset object handles all the windowing, batching, and normalization
    internally — we just describe what the inputs are.
    """
    # Use the last MAX_H time steps of each ticker as the validation set
    max_time = df.groupby("ticker")["time_idx"].transform("max")
    df["is_val"] = df["time_idx"] > (max_time - MAX_H * 2)

    train_df = df[~df["is_val"]].copy()
    val_df = df.copy()

    # Features that don't change over time for a given ticker
    static_cats = ["ticker"]

    # Features we know about the future — calendar info
    future_known_cats = ["day_of_week", "month", "quarter"]

    # Features we observed in the past
    past_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "SMA_50",
        "SMA_200",
        "RSI_14",
        "MOM_1",
        "ROC_14",
        "MACD",
    ]

    training_ds = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="log_close",
        group_ids=["ticker"],
        min_encoder_length=WINDOW // 2,
        max_encoder_length=WINDOW,
        min_prediction_length=1,
        max_prediction_length=MAX_H,
        static_categoricals=static_cats,
        time_varying_known_categoricals=future_known_cats,
        time_varying_unknown_reals=past_features,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(
        training_ds, val_df, predict=True, stop_randomization=True
    )

    return training_ds, val_ds


def train():
    raw_dir = os.path.join(os.path.dirname(SAVE_DIR), "dataset", "raw")
    model_path = os.path.join(SAVE_DIR, "tft_best.ckpt")
    meta_path = os.path.join(SAVE_DIR, "transformer_meta.pkl")

    logger.info("Loading ticker data from %s ...", raw_dir)
    df = load_raw_csvs(raw_dir)

    logger.info("Building TFT datasets ...")
    training_ds, val_ds = build_tft_datasets(df)

    train_dl = training_ds.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    val_dl = val_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    logger.info("TFT parameter count: %d", sum(p.numel() for p in tft.parameters()))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        dirpath=SAVE_DIR,
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    progress = TQDMProgressBar(refresh_rate=10)

    mlflow.set_experiment("stock-forecasting-tft")
    with mlflow.start_run(run_name="tft"):
        mlflow.log_params(
            {
                "model": "TemporalFusionTransformer",
                "window": WINDOW,
                "max_prediction_length": MAX_H,
                "hidden_size": 64,
                "attention_heads": 4,
                "dropout": 0.2,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "loss": "QuantileLoss(p10, p50, p90)",
                "target": "log_close",
                "split": "chronological (last 2*MAX_H per ticker = val)",
            }
        )

        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=DEVICE,
            gradient_clip_val=0.1,
            callbacks=[early_stop, checkpoint, progress],
            enable_model_summary=True,
            log_every_n_steps=10,
        )
        trainer.fit(tft, train_dl, val_dl)

        best_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        best_model.eval()

        # Run validation predictions and log metrics
        preds = best_model.predict(val_dl, mode="quantiles", return_y=True)
        y_true = preds.y[0].cpu().numpy().flatten()
        y_pred = preds.output[:, :, 1].cpu().numpy().flatten()  # p50

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        dir_acc = (
            np.mean(np.sign(y_pred - y_true.mean()) == np.sign(y_true - y_true.mean()))
            * 100
        )

        logger.info(
            "Val MAE=%.4f  RMSE=%.4f  R²=%.4f  DirAcc=%.1f%%", mae, rmse, r2, dir_acc
        )
        mlflow.log_metrics(
            {
                "val_mae": mae,
                "val_rmse": rmse,
                "val_r2": r2,
                "val_dir_acc": dir_acc,
            }
        )
        mlflow.log_artifact(model_path)

    # Save metadata so app.py knows how to load and use the model
    joblib.dump(
        {
            "window": WINDOW,
            "horizons": HORIZONS,
            "max_prediction_length": MAX_H,
            "model_type": "TFT",
            "checkpoint": model_path,
        },
        meta_path,
    )

    logger.info("Training complete. Best checkpoint saved to %s", model_path)


if __name__ == "__main__":
    train()
