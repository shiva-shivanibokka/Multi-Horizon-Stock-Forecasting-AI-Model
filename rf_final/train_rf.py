"""
train_rf.py
Trains a Random Forest model to predict stock prices at three future horizons:
1 week, 1 month, and 6 months.

What is a Random Forest?
    A Random Forest is an ensemble of decision trees. Each tree learns a set of
    if-then-else rules that split the data into smaller and smaller groups until
    it can make a prediction. A single decision tree tends to overfit (memorise
    the training data), so a Random Forest builds many trees, each trained on a
    random subset of the data and features, and averages their predictions.

    The averaging effect cancels out individual tree errors, giving a much more
    robust and generalisable model. This technique is called "bagging"
    (Bootstrap Aggregating).

Why Random Forest for stock prediction?
    Unlike neural networks, Random Forests:
      - Require no scaling of inputs (trees split on thresholds, not magnitudes)
      - Are not sensitive to outliers
      - Are interpretable: you can see which features are most important
      - Train quickly on CPU with parallelism (n_jobs=-1 uses all cores)
      - Provide a strong non-deep-learning baseline to compare against

    The tradeoff: Random Forests cannot model sequential dependencies. Each
    window of 252 trading days is flattened to a 1260-element vector, losing
    the temporal order information. Neural networks (RNN, LSTM, Transformer)
    process the sequence day-by-day and can learn time-ordered patterns.

Why 252 days (1 year)?
    The RF uses a 252-day lookback window. Since it treats the window as a flat
    vector (ignoring sequence), longer windows just add more features without
    adding sequential understanding. 252 days provides enough price history for
    the tree splits to capture trend, momentum, and seasonality signals.

Feature representation:
    Each window is flattened to (252 days * 5 OHLCV columns) = 1260 features.
    Feature names follow the pattern "Close_t-252" (Close price 252 days ago),
    "Close_t-251", ..., "Close_t-1" (yesterday's Close). This naming convention
    makes feature importance analysis straightforward.

How to run:
    1. python build_dataset.py --refresh   (build the shared dataset)
    2. python rf_final/train_rf.py
"""

import os
import sys
import pandas as pd
import numpy as np
import math
import joblib
import mlflow

# MLflow 3.x defaults to a SQLite tracking store that requires a database file.
# The flat file store writes directly to mlruns/ with no setup needed.
mlflow.set_tracking_uri("file:./mlruns")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_guards import (
    check_price_data,
    check_feature_array,
    check_train_test_split,
    check_target_distribution,
    log_dataset_summary,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


# HYPERPARAMETERS

# Forecast horizons: label -> number of trading days ahead to predict
HORIZONS = {"1w": 5, "1m": 21, "6m": 126}
MAX_H = max(HORIZONS.values())  # largest horizon = 126 days

# WINDOW: how many consecutive trading days feed into the Random Forest.
# 252 days is approximately 1 trading year.
# These 252 days are flattened to a 1260-element input vector (252 * 5 OHLCV).
WINDOW = 252


# HELPER FUNCTIONS


def fetch_sp500_tickers():
    """
    Scrapes the current S&P 500 ticker list from Wikipedia.
    Returns a list of ticker symbols like ['AAPL', 'MSFT', 'GOOG', ...].

    Note: This function is only used if the dataset is built from scratch
    without the shared build_dataset.py. In normal operation, the model
    loads from the shared windows_252.npz dataset.
    """
    import requests, io

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research project; contact via GitHub)"}
    html = requests.get(url, headers=headers, timeout=15).text
    df0 = pd.read_html(io.StringIO(html), header=0)[0]
    return df0["Symbol"].tolist()


def _download_one(sym):
    """
    Downloads 5 years of daily OHLCV data for a single ticker from Yahoo Finance.

    Returns a tuple (sym, df) where df is a DataFrame, or (sym, None) if the
    download fails for any reason (network error, delisted ticker, etc.).

    The try/except ensures one failed ticker does not stop the entire download.
    """
    import yfinance as yf

    try:
        yf_sym = sym.replace(".", "-").upper()  # Yahoo uses BRK-B not BRK.B
        df = yf.download(yf_sym, period="5y", interval="1d", progress=False)
        if df is None or df.empty:
            return sym, None
        # Newer yfinance returns MultiIndex columns for single-ticker downloads.
        # Flatten to single-level so df["Close"] works normally.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            return sym, None
        return sym, df
    except Exception:
        return sym, None


def download_all(tickers, max_workers=32):
    """
    Downloads all tickers in parallel using a thread pool.

    Why threads and not processes?
        yfinance downloads are I/O-bound (waiting for Yahoo's servers).
        The Python GIL does not block threads during I/O, so threads
        give true parallelism here. max_workers=32 keeps concurrent
        requests within Yahoo Finance's rate limits.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, sym): sym for sym in tickers}
        done = 0
        for future in as_completed(futures):
            sym, df = future.result()
            done += 1
            if df is not None:
                results[sym] = df
            if done % 50 == 0:
                print(f"  Downloaded {done}/{len(tickers)} tickers...")
    print(f"Download complete: {len(results)}/{len(tickers)} tickers succeeded.")
    return results


def build_multi_horizon_dataset(tickers):
    """
    Builds the RF training dataset from raw price data.

    For each ticker, for each 252-day window:
      - X row: the 252 days of OHLCV flattened to a 1260-element vector
      - Y row: [future_price_1w, future_price_1m, future_price_6m]

    The "sliding window" approach means consecutive windows overlap heavily.
    For example, window[0] uses days 0-251 and window[1] uses days 1-252.
    This maximises the amount of training data extracted from each ticker.
    """
    X_rows, Y_rows = [], []
    print(f"Downloading {len(tickers)} tickers in parallel...")
    data = download_all(tickers)

    for sym, hist in data.items():
        try:
            hist = check_price_data(hist, sym)

            # Skip tickers with insufficient history for even one window + targets
            if len(hist) < WINDOW + MAX_H:
                print(f"Skip {sym}: only {len(hist)} rows")
                continue

            # Get only OHLCV columns as a numpy array, sorted by date
            vals = (
                hist[["Open", "High", "Low", "Close", "Volume"]].sort_index().to_numpy()
            )

            # Sliding window: extract one row per starting position
            for start in range(len(vals) - (WINDOW + MAX_H) + 1):
                # Flatten the 252-day OHLCV block to a single vector of length 1260
                window_feats = vals[start : start + WINDOW].flatten()

                # Targets: Close price at each forecast horizon after the window
                # h-1 because index 0 = first day AFTER the window (1 trading day ahead)
                targets = [vals[start + WINDOW + h - 1, 3] for h in HORIZONS.values()]

                X_rows.append(window_feats)
                Y_rows.append(targets)

        except Exception as e:
            print(f"Skip {sym}: {e}")
            continue

    # Build descriptive column names: "Open_t-252", "High_t-252", ..., "Volume_t-1"
    # t-252 means "252 days ago", t-1 means "yesterday"
    feat_names = [
        f"{c}_t-{t}"
        for t in range(WINDOW, 0, -1)  # 252, 251, ..., 1
        for c in ["Open", "High", "Low", "Close", "Volume"]
    ]
    X = pd.DataFrame(X_rows, columns=feat_names)
    Y = pd.DataFrame(Y_rows, columns=[f"next_{k}" for k in HORIZONS])
    return X, Y


def print_metrics(name, y_true, y_pred, last_close):
    """
    Prints evaluation metrics for all three forecast horizons.

    Metrics explained:
      MSE  (Mean Squared Error):    average of (predicted - actual)^2
      RMSE (Root MSE):              sqrt(MSE) — in the same units as price (dollars)
      MAE  (Mean Absolute Error):   average |predicted - actual| in dollars
      MAPE (Mean Abs % Error):      average |predicted - actual| / actual — scale-free
      R²   (R-squared):             1 = perfect fit, 0 = no better than predicting the mean
      DirAcc (Directional Accuracy): % of windows where the model predicts the correct
                                      direction (up/down) relative to the last close price

    Why last_close matters for DirAcc:
        DirAcc compares each prediction against its OWN last close, not a single
        global reference price. This is the correct metric for evaluating whether
        a trading signal (buy/sell) would have been correct for each window.
    """
    print(f"--- {name} ---")
    for i, key in enumerate(HORIZONS):
        true = y_true.iloc[:, i].values
        pred = y_pred[:, i]
        mse = mean_squared_error(true, pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        r2 = r2_score(true, pred)

        # np.sign() returns +1 for positive, -1 for negative, 0 for zero.
        # Comparing signs tells us if the model predicted the correct direction.
        dir_acc = (
            np.mean(np.sign(pred - last_close) == np.sign(true - last_close)) * 100
        )

        print(
            f" {key}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, "
            f"MAPE={mape:.2%}, R²={r2:.4f}, DirAcc={dir_acc:.1f}%"
        )
    print()


# MAIN TRAINING SCRIPT


def main():
    # Build the path to the shared dataset file
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "windows_252.npz",
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset cache not found at {dataset_path}.\n"
            "Run  python build_dataset.py  from the project root first."
        )

    # -------------------------------------------------------------------------
    # LOAD DATASET
    #
    # The RF uses X_flat — a pre-flattened version of the OHLCV data.
    # Shape: (N_windows, 1260) where 1260 = 252 days * 5 OHLCV columns.
    #
    # Unlike the neural network models, the RF does NOT need memory-mapping
    # because the full X_flat array is only 4.4 GB (float32) — manageable
    # to load directly into RAM on a 34 GB machine.
    # -------------------------------------------------------------------------
    print(f"Loading dataset from {dataset_path} ...")
    cache = np.load(dataset_path)
    X_np = cache["X_flat"]  # (N, 1260) — flattened OHLCV windows, float32
    Y_np = cache["Y"]  # (N, 3)    — future Close prices, float32
    print(f"Loaded: X={X_np.shape}  Y={Y_np.shape}")

    # Create descriptive column names for the RF's feature importance output
    feat_names = [
        f"{c}_t-{t}"
        for t in range(252, 0, -1)
        for c in ["Open", "High", "Low", "Close", "Volume"]
    ]
    X = pd.DataFrame(X_np, columns=feat_names)
    Y = pd.DataFrame(Y_np, columns=[f"next_{k}" for k in HORIZONS])

    # Sanity check: verify no NaN or Inf values in X
    check_feature_array(X_np, "X (raw)")

    # Compute 1-week implied returns for distribution check.
    # The last Close value is at position -5+3 = -2 in X_flat
    # (5 columns per day, Close is index 3, last day has offset -5 from end).
    current_close = X_np[:, -5 + 3]  # last row's Close column
    returns_1d = (Y_np[:, 0] - current_close) / (current_close + 1e-8)
    check_target_distribution(returns_1d, "1d returns")

    print("Training model...")

    # -------------------------------------------------------------------------
    # CHRONOLOGICAL 80/20 SPLIT
    #
    # Same principle as the neural network models: split by date order to
    # avoid look-ahead bias. The first 80% is training, last 20% is validation.
    # -------------------------------------------------------------------------
    split = int(0.8 * len(X))
    X_train = X.iloc[:split]
    Y_train = Y.iloc[:split]
    X_test = X.iloc[split:]
    Y_test = Y.iloc[split:]

    # Compute per-window last close for Directional Accuracy
    LC_all = current_close
    LC_train = LC_all[:split]
    LC_test = LC_all[split:]

    check_train_test_split(X_train.values, X_test.values)
    log_dataset_summary(
        X_train.values,
        Y_train.values,
        n_tickers=len(X_np) // 252,  # rough estimate: windows / days per year
    )

    # -------------------------------------------------------------------------
    # TRAIN THE RANDOM FOREST
    # -------------------------------------------------------------------------
    mlflow.set_experiment("stock-forecasting-rf")
    with mlflow.start_run(run_name="random-forest"):
        mlflow.log_params(
            {
                "model": "RandomForest + MultiOutputRegressor",
                "n_estimators": 50,
                "window": WINDOW,
                "split": "chronological 80/20",
            }
        )

        # RandomForestRegressor parameters explained:
        #   n_estimators=50:    50 decision trees. More trees = better accuracy
        #                       but slower training. 50 is a practical baseline.
        #   random_state=42:    fixed seed for reproducibility (same results each run)
        #   n_jobs=-1:          use ALL available CPU cores in parallel
        #   verbose=1:          print one line per tree so you can see progress
        #   max_features="sqrt": each split considers sqrt(1260) ≈ 35 features.
        #                       Using all 1260 features per split would be very slow
        #                       and would make trees too similar (defeating the
        #                       diversity benefit of an ensemble).
        base = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            max_features="sqrt",
        )

        # MultiOutputRegressor trains one separate RandomForest per output column.
        # Since we have 3 horizons (1w, 1m, 6m), this trains 3 independent forests.
        # n_jobs=-1 parallelises across the 3 output forests as well.
        model = MultiOutputRegressor(base, n_jobs=-1)
        print(
            f"Training RandomForest on {len(X_train)} samples x {X_train.shape[1]} features ..."
        )
        model.fit(X_train, Y_train)

        # Evaluate on both splits to check for overfitting
        # (A large gap between training and validation metrics indicates overfitting)
        print_metrics("Training", Y_train, model.predict(X_train), LC_train)
        print_metrics("Validation", Y_test, model.predict(X_test), LC_test)

        # Log validation MAE per horizon to mlflow for run comparison
        val_preds = model.predict(X_test)
        for idx, name in enumerate(HORIZONS):
            mae = mean_absolute_error(Y_test.iloc[:, idx], val_preds[:, idx])
            mlflow.log_metric(f"val_mae_{name}", mae)

        # Save the trained model and feature list to disk.
        # joblib serialises large sklearn models efficiently (better than pickle).
        joblib.dump(model, "rf_multi_horizon.pkl")
        joblib.dump(X.columns.tolist(), "feature_list_multi.pkl")
        mlflow.log_artifact("rf_multi_horizon.pkl")
        print("Model and features saved.")


if __name__ == "__main__":
    main()
