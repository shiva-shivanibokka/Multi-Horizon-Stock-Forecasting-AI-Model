import pandas as pd
import numpy as np
import math
import joblib
import mlflow
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# --- Forecast horizons (trading days) ---
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "6m": 126,
    "1y": 252,
}
MAX_H = max(HORIZONS.values())
WINDOW = 252


def fetch_sp500_tickers():
    import requests, io

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (research project; contact via GitHub)"}
    html = requests.get(url, headers=headers, timeout=15).text
    df0 = pd.read_html(io.StringIO(html), header=0)[0]
    return df0["Symbol"].tolist()


def build_multi_horizon_dataset(tickers):
    X_rows, Y_rows = [], []
    for sym in tickers:
        yf_sym = sym.replace(".", "-").upper()
        try:
            hist = yf.download(
                yf_sym, period="5y", interval="1d", progress=False, auto_adjust=False
            ).dropna()
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
            continue

        if len(hist) < WINDOW + MAX_H:
            print(f"Skip {sym}: only {len(hist)} rows")
            continue

        vals = hist[["Open", "High", "Low", "Close", "Volume"]].sort_index().to_numpy()
        for start in range(len(vals) - (WINDOW + MAX_H) + 1):
            window_feats = vals[start : start + WINDOW].flatten()
            targets = [vals[start + WINDOW + h - 1, 3] for h in HORIZONS.values()]
            X_rows.append(window_feats)
            Y_rows.append(targets)

    feat_names = [
        f"{c}_t-{t}"
        for t in range(WINDOW, 0, -1)
        for c in ["Open", "High", "Low", "Close", "Volume"]
    ]
    X = pd.DataFrame(X_rows, columns=feat_names)
    Y = pd.DataFrame(Y_rows, columns=[f"next_{k}" for k in HORIZONS])
    return X, Y


def print_metrics(name, y_true, y_pred):
    print(f"--- {name} ---")
    for i, key in enumerate(HORIZONS):
        true = y_true.iloc[:, i]
        pred = y_pred[:, i]
        mse = mean_squared_error(true, pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        r2 = r2_score(true, pred)
        dir_acc = (
            np.mean(
                np.sign(pred - float(true.iloc[0]))
                == np.sign(true.values - float(true.iloc[0]))
            )
            * 100
        )
        print(
            f" {key}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2%}, R²={r2:.4f}, DirAcc={dir_acc:.1f}%"
        )
    print()


def main():
    tickers = fetch_sp500_tickers()
    print("Downloading stock data...")
    X, Y = build_multi_horizon_dataset(tickers)

    print("Finished downloading. Training model...")
    # Chronological split — no shuffle. Random shuffle causes data leakage
    # in financial time series: future windows leak into the training set.
    split = int(0.8 * len(X))
    X_train, Y_train = X.iloc[:split], Y.iloc[:split]
    X_test, Y_test = X.iloc[split:], Y.iloc[split:]

    mlflow.set_experiment("stock-forecasting-rf")
    with mlflow.start_run(run_name="random-forest"):
        mlflow.log_params(
            {
                "model": "RandomForest + MultiOutputRegressor",
                "n_estimators": 100,
                "window": WINDOW,
                "split": "chronological 80/20",
            }
        )

        base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base)
        model.fit(X_train, Y_train)

        print_metrics("Training", Y_train, model.predict(X_train))
        print_metrics("Validation", Y_test, model.predict(X_test))

        val_preds = model.predict(X_test)
        for idx, name in enumerate(HORIZONS):
            mae = mean_absolute_error(Y_test.iloc[:, idx], val_preds[:, idx])
            mlflow.log_metric(f"val_mae_{name}", mae)

        joblib.dump(model, "rf_multi_horizon.pkl")
        joblib.dump(X.columns.tolist(), "feature_list_multi.pkl")
        mlflow.log_artifact("rf_multi_horizon.pkl")
        print("Model and features saved.")


if __name__ == "__main__":
    main()
