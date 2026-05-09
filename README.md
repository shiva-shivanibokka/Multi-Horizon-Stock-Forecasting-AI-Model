# Multi-Horizon Stock Price Forecasting

A full-stack AI system that predicts stock prices at three future time horizons — 1 week, 1 month, and 6 months — using four different machine learning models. It serves predictions through a Flask API and displays everything in a Next.js dashboard with confidence intervals, model comparisons, news sentiment, and company fundamentals.

The project was built with learning in mind. Each of the four models represents a different approach to sequence modelling, from the simplest (Random Forest) to the most advanced (PatchTST Transformer). Reading the training scripts alongside this README will give you a good grounding in how each technique works and why one outperforms another.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [The Four Models](#the-four-models)
4. [How the Data Pipeline Works](#how-the-data-pipeline-works)
5. [The 36 Features](#the-36-features)
6. [Data Quality Checks](#data-quality-checks)
7. [Training the Models](#training-the-models)
8. [How to Run It](#how-to-run-it)
9. [API Reference](#api-reference)
10. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
11. [Automated Weekly Retraining](#automated-weekly-retraining)
12. [Deployment](#deployment)
13. [Results](#results)

---

## What This Project Does

Given a stock ticker like `AAPL` or `NVDA`, the system fetches the last few years of that stock's daily price history, runs it through four trained machine learning models, and returns price predictions at three points in the future.

For every prediction, the system also returns a **confidence range** — a low estimate (p10), a middle estimate (p50), and a high estimate (p90). This tells you not just what the model thinks will happen, but how uncertain it is. A wide range between p10 and p90 means the model is unsure. A narrow range means it is more confident.

The dashboard shows five tabs:

- **Forecast** — price predictions with the confidence range, and a BUY / HOLD / SELL recommendation based on the predicted upside
- **Model Comparison** — all four models' predictions side by side, so you can see where they agree and disagree
- **Price Chart** — one year of actual price history with moving averages overlaid
- **Sentiment** — recent news headlines with an AI-generated sentiment score for each
- **Fundamentals** — key company data like P/E ratio, revenue, debt, and profit margin

---

## Project Structure

```
Multi-Horizon-Stock-Forecasting-AI-Model/
│
├── app.py                    Flask backend — loads all 4 models and serves predictions
├── build_dataset.py          Downloads S&P 500 data once, shared by all four models
├── data_guards.py            Data quality checks that run automatically during training
├── retrain.py                Orchestrates retraining all 4 models in sequence
├── requirements.txt          Python package dependencies
├── Dockerfile                Packages the Flask backend into a container
├── docker-compose.yml        Runs backend + frontend together with one command
│
├── transformer_final/
│   ├── train_transformer.py  PatchTST Transformer training script
│   ├── evaluate_transformer.py         Standalone evaluation script (run after training)
│   ├── transformer_multi_horizon.pth   Trained model weights
│   ├── scaler_feat.pkl       Feature scaler (used at inference time)
│   ├── scaler_ret.pkl        Target scaler (used at inference time)
│   └── transformer_meta.pkl  Model metadata (window size, horizons, etc.)
│
├── lstm_final_project/
│   ├── train_lstm.py         LSTM training script
│   ├── lstm_multi_horizon.pth
│   ├── lstm_scaler_feat.pkl
│   └── lstm_scaler_targ.pkl
│
├── rnn_final/
│   ├── train_rnn.py          RNN training script
│   ├── rnn_multi_horizon.pth
│   ├── rnn_scaler_feat.pkl
│   └── rnn_scaler_targ.pkl
│
├── rf_final/
│   ├── train_rf.py           Random Forest training script
│   ├── rf_multi_horizon.pkl  Trained model (sklearn, not PyTorch)
│   └── feature_list_multi.pkl
│
├── dataset/                  Shared data cache (created by build_dataset.py)
│   ├── windows_756.npz       3-year sliding windows for Transformer and LSTM
│   ├── windows_252.npz       1-year sliding windows for RNN and Random Forest
│   ├── X_756_scaled.npy      Pre-scaled float16 version of the Transformer/LSTM features
│   ├── X_seq_252.npy         Pre-scaled float16 version of the RNN features
│   ├── meta.json             Dataset summary (n_tickers, date range, feature count)
│   └── raw/                  One .npz per ticker with cleaned OHLCV + indicators
│
├── nextjs/
│   ├── src/app/page.jsx      Main dashboard (5 tabs)
│   └── src/components/       ForecastTable, CompareChart, SentimentPanel, etc.
│
└── .github/workflows/
    └── retrain.yml           Runs every Sunday to retrain all models on fresh data
```

---

## Key Files Explained

Here is a plain-language description of every important file in the project and what it does. You do not need to understand all of these to run the project, but if you are studying the code this is a useful reference.

**Root folder**

`app.py` — the Flask web server that powers the entire backend. When you run `python app.py`, it loads all four trained models into memory, then listens for prediction requests from the frontend. Every time someone searches for a ticker on the website, this file handles the request: it fetches the latest price history from Yahoo Finance, formats it into the right shape for each model, runs the predictions, and sends the results back as JSON. It also handles the sentiment and fundamentals endpoints, rate limiting, and the model reload endpoint used by the automated retraining workflow.

`build_dataset.py` — downloads 10 years of daily stock data for all S&P 500 companies from Yahoo Finance and builds the shared training dataset. All four models train from this shared cache rather than downloading data independently. Run this once before training any model. The `--refresh` flag forces a fresh download to update the data with the latest prices.

`data_guards.py` — a set of data quality checking functions that are called automatically during training. It checks for things like zero prices, missing values, extreme single-day moves (which usually indicate a data error from a stock split), and NaN or infinity values that would corrupt model weights. If a critical problem is found, training stops immediately with a clear error message rather than silently producing a broken model.

`retrain.py` — a convenience script that runs all four model training scripts one after another in the correct order. Used by the automated weekly retraining workflow and for local retraining. Accepts flags like `--model transformer` to retrain only one model, or `--refresh-data` to rebuild the dataset first.

`start_app_simple.py` — a simplified version of app.py for running the backend locally without production-level configuration. Useful for quick testing.

`start_app.bat` — a Windows batch file that starts the backend with a double-click. Equivalent to running `python app.py` from the terminal.

`requirements.txt` — lists every Python package the project depends on with pinned version numbers. Run `pip install -r requirements.txt` to install everything at once.

`Dockerfile` — instructions for packaging the Flask backend into a Docker container. Allows the backend to run in any environment without worrying about Python version or dependency conflicts.

`docker-compose.yml` — runs both the backend and frontend together with one command (`docker-compose up`). Handles the networking between the two containers automatically.

---

**dataset/ folder**

`build_dataset.py` (called from root) produces all the files in this folder.

`windows_756.npz` — the main training dataset for the Transformer and LSTM. Contains 657,035 training windows, each covering 756 consecutive trading days (3 years) for one stock. Also contains the matching target prices (what the stock was worth 1 week, 1 month, and 6 months after each window), sector labels, dates, and last close prices. This is a ZIP archive of numpy arrays.

`windows_252.npz` — the equivalent dataset for the RNN and Random Forest, using shorter 252-day (1 year) windows. Contains 875,771 training windows plus the same target and metadata arrays.

`X_756_scaled.npy` — a pre-processed version of the Transformer and LSTM feature data. The 657,035 windows have been run through a StandardScaler (normalised to zero mean and unit variance) and saved as float16 (half-precision) to halve the file size. The Transformer training script loads directly from this file. Created by `dataset/prescale.py`.

`X_seq_252.npy` — the equivalent pre-processed file for the RNN. Contains the 252-day feature windows, scaled and saved as float16. Created automatically the first time `train_rnn.py` runs.

`Y_ret_756.npy` — the return targets for the Transformer: how much the stock price changed (as a fraction) at each horizon after each window. The Transformer predicts returns rather than raw prices because returns are more stationary across different stocks and time periods.

`Y_px_756.npy` — the raw price targets for the Transformer evaluation: the actual dollar price at each horizon. Used only for computing evaluation metrics — not fed into the model during training.

`LC_756.npy` — the last closing price for each of the 657,035 training windows. Used to convert predicted returns back to predicted prices at inference time, and to compute directional accuracy during evaluation.

`LC_252.npy` — same as above but for the 252-day RNN dataset.

`Y_252.npy` — the price targets for the RNN dataset.

`scaler_feat.pkl` — the StandardScaler fitted on the Transformer/LSTM feature data. Saved so inference code can apply the same normalisation to new data at prediction time.

`scaler_ret.pkl` — the StandardScaler fitted on the Transformer's return targets. Saved so inference code can reverse the normalisation and convert model outputs back to interpretable return values.

`scaler_feat_252.pkl` — the equivalent feature scaler for the RNN dataset.

`meta.json` — a small JSON file summarising the dataset: how many tickers were included, the date range covered, how many training windows were produced, and what features were used. Useful for quickly checking what a dataset contains without loading the full arrays.

`prescale.py` — the one-time script that reads `windows_756.npz`, fits a StandardScaler, applies it to all 657,035 windows, and writes `X_756_scaled.npy` in float16. Also extracts all the small arrays (`Y_ret_756.npy`, `Y_px_756.npy`, `LC_756.npy`, etc.) from the zip into separate files. Run this once after `build_dataset.py` and before training the Transformer or LSTM.

`convert_to_npy.py` — an earlier version of prescale.py that extracted arrays without applying scaling. Superseded by prescale.py and kept only as a reference.

---

**transformer_final/ folder**

`train_transformer.py` — trains the PatchTST Transformer. Runs three walk-forward cross-validation folds to produce honest performance metrics, then trains a final production model on the full dataset. At the end of each fold, it automatically evaluates the fold model on its test window and logs MAE and directional accuracy. The most complex training script in the project. Heavily commented to explain every design decision.

`evaluate_transformer.py` — a standalone evaluation script that can be run at any time after training, without retraining the model. It loads the saved `transformer_multi_horizon.pth` weights and evaluates them against the three walk-forward fold test windows, printing MAE and directional accuracy per horizon. This is useful if you want to measure the model's performance after the fact, check how it holds up on a specific time period, or get metrics without running a full training job again. The training script already evaluates automatically during training, so this script is not required — it is simply a convenience tool for post-training analysis.

`transformer_multi_horizon.pth` — the trained Transformer weights. This is what `app.py` loads. A `.pth` file is a standard PyTorch format for saving model parameters.

`_fold_fold1.pth`, `_fold_fold2.pth`, `_fold_fold3.pth` — the trained weights from each cross-validation fold. These are intermediate checkpoints used only during the CV evaluation phase. The production model loaded by app.py is `transformer_multi_horizon.pth`, not these.

`scaler_feat.pkl` — the feature scaler saved alongside the production model for use during inference.

`scaler_ret.pkl` — the return target scaler saved alongside the production model.

`transformer_meta.pkl` — metadata the inference code needs to reconstruct and use the model: window size, patch length, stride, which horizons the model was trained for, how many features it expects, and how many quantiles it outputs.

---

**lstm_final_project/ folder**

`train_lstm.py` — trains the LSTM model. Uses the same pre-scaled dataset as the Transformer (`X_756_scaled.npy`). Includes early stopping, AMP, gradient clipping, and per-horizon evaluation metrics.

`lstm_multi_horizon.pth` — the trained LSTM weights loaded by `app.py`.

`lstm_best.pth` — a temporary checkpoint written during training to save the best epoch's weights. After training completes, the best weights are copied to `lstm_multi_horizon.pth`. This file can be ignored.

`lstm_scaler_feat.pkl` — the feature scaler for LSTM inference.

`lstm_scaler_targ.pkl` — the target scaler for LSTM inference (used to convert scaled predictions back to dollar values).

`lstm_meta.pkl` — metadata: window size, horizons, feature count.

`infer_lstm.py` — a standalone script for running the LSTM on a single ticker and printing its predictions. Useful for testing the model outside the full app.

---

**rnn_final/ folder**

`train_rnn.py` — trains the RNN model. On first run, extracts and scales the RNN feature data from `windows_252.npz` into `X_seq_252.npy` (this takes about 10 minutes). Subsequent runs skip the extraction and load directly from the `.npy` file.

`rnn_multi_horizon.pth` — the trained RNN weights loaded by `app.py`.

`rnn_best.pth` — the best-epoch checkpoint written during training. Can be ignored after training completes.

`rnn_scaler_feat.pkl`, `rnn_scaler_targ.pkl` — scalers for inference.

`rnn_meta.pkl` — metadata: window size, horizons, feature count.

`infer_rnn.py` — standalone inference script for testing the RNN on a single ticker.

---

**rf_final/ folder**

`train_rf.py` — trains the Random Forest. Unlike the neural network training scripts, this does not use GPU, does not have epochs, and does not have early stopping. It loads `windows_252.npz` directly (the full 4.4 GB flat feature array fits in RAM), splits it chronologically, and calls sklearn's `model.fit()` once. Training takes 2–4 hours because fitting 50 decision trees on 700,000 samples with 1,260 features is compute-intensive even with all CPU cores.

`rf_multi_horizon.pkl` — the trained Random Forest model. Unlike neural networks which save weight tensors as `.pth` files, sklearn models are saved with `joblib` as `.pkl` files. This file contains all 50 trees for all 3 horizons.

`feature_list_multi.pkl` — a list of the 1,260 feature names in the order the model expects them. Required at inference time to ensure the input columns are in the right order.

`infer_rf.py` — standalone inference script for testing the Random Forest on a single ticker.

---

## The Four Models

This project deliberately uses four different models rather than just one. Each represents a distinct approach to sequence prediction, and understanding why they perform differently is as valuable as the predictions themselves.

### Random Forest

The Random Forest is the simplest model in the project and the easiest to understand. It is not a neural network — it is an ensemble of decision trees.

A decision tree is a series of yes/no questions that splits data into smaller and smaller groups. For example: "Is the 50-day moving average above the 200-day moving average? If yes, go left. If no, go right." At the end of each branch, the tree makes a prediction based on the training examples that ended up in that group.

A single decision tree tends to overfit — it memorises the training data rather than learning general rules. A Random Forest solves this by building 50 trees, each trained on a random sample of the data and using only a random subset of features for each split. The final prediction is the average across all 50 trees. The diversity between trees cancels out individual mistakes.

For this project, the Random Forest receives a 252-day window flattened into a single vector of 1,260 numbers (252 days × 5 OHLCV columns). It has no concept of time order — it simply sees 1,260 numbers and produces three price predictions. This limitation means it cannot learn patterns like "momentum that builds over 3 months tends to reverse," but it is fast, requires no GPU, and produces surprisingly competitive short-term predictions.

- Input: 252 days of OHLCV, flattened to 1,260 numbers
- Output: one point estimate per horizon (no confidence intervals)
- Training time: 2–4 hours on CPU (n_jobs=-1 uses all cores)
- File: `rf_final/train_rf.py`

### RNN (Recurrent Neural Network)

The RNN is the simplest neural network designed for sequential data. Unlike the Random Forest, it processes data one day at a time and carries a "hidden state" — a fixed-size vector that summarises everything it has seen so far — from one day to the next.

At each timestep, the RNN runs this calculation:

```
h_t = tanh( W_ih × x_t  +  W_hh × h_{t-1} )
```

Where `x_t` is today's feature vector, `h_{t-1}` is the hidden state from yesterday, and `W_ih`, `W_hh` are learned weight matrices. The `tanh` function squashes the result to the range [-1, 1].

After processing all 252 days, the final hidden state `h_252` is passed to a linear layer that produces the price predictions.

The RNN has a well-known limitation: the vanishing gradient problem. During training, gradients must flow backwards through all 252 timesteps to update the weights. At each step, the gradient is multiplied by `W_hh`. If the entries of `W_hh` are slightly less than 1, repeated multiplication makes the gradient shrink exponentially. By the time it reaches day 1, it is effectively zero. This means the model cannot learn relationships between events that are far apart in time — what happened 200 days ago has almost no influence on the learned weights.

This is why the RNN uses a 252-day window rather than the 756-day window used by the LSTM and Transformer. Longer windows give no benefit because the vanishing gradient makes the model blind to anything more than ~50 days back.

- Input: 252 days of 36 features
- Output: one point estimate per horizon (no confidence intervals)
- Training time: 20–35 minutes on GPU
- File: `rnn_final/train_rnn.py`

### LSTM (Long Short-Term Memory)

The LSTM was invented specifically to fix the vanishing gradient problem of vanilla RNNs. It introduces three learnable "gates" that control information flow:

- **Forget gate**: decides what fraction of the previous cell state to keep
- **Input gate**: decides what new information to write into the cell state
- **Output gate**: decides what to read from the cell state as the hidden state

The cell state `C_t` is a separate memory vector that passes through the sequence with only additive updates — gradients can flow through it over many timesteps without shrinking. This is the key innovation that lets LSTMs learn dependencies across hundreds of timesteps.

In practice, this means the LSTM can learn things like "a drop in price following a strong earnings beat tends to recover over the next 6 months" — a pattern that requires connecting events separated by a large number of days.

This project uses a two-layer LSTM with hidden size 128. The first layer learns local patterns (day-to-day momentum, short-term moving average crossovers). The second layer learns patterns of patterns (e.g. what configurations of the first layer's output signal a trend reversal).

The LSTM uses the same 756-day window as the Transformer, making their comparison fair.

- Input: 756 days of 36 features
- Output: one point estimate per horizon (no confidence intervals)
- Training time: 20–30 minutes on GPU
- File: `lstm_final_project/train_lstm.py`

### PatchTST Transformer (Primary Model)

The Transformer is the most recent and most accurate model in the project. It does not process sequences step by step — instead, it looks at the entire 756-day history simultaneously using a mechanism called **self-attention**.

Self-attention computes, for every pair of positions in the sequence, how much each position should "attend to" every other position. This is done in parallel across all positions at once. The result is that any two days can directly influence each other regardless of how far apart they are — day 1 and day 756 are connected in a single computation step, unlike the LSTM which must pass information through 756 hidden states.

This project uses a specific variant called **PatchTST**. Instead of applying attention to each of the 756 individual days, it first groups consecutive days into "patches" of 16 days each, with a stride of 8 (meaning patches overlap). This produces 93 patch tokens instead of 756 day tokens.

Self-attention has a computational cost that grows with the square of the sequence length. Comparing all pairs of 756 individual days costs 756² = 571,536 operations. Comparing all pairs of 93 patches costs 93² = 8,649 — a 66-times reduction. Each patch also captures local temporal structure (a 16-day price movement), so the model can reason about both local and global patterns.

The model adds **sector conditioning**: a learnable embedding vector for each of the 11 GICS sector categories (technology, healthcare, energy, etc.) is added to every patch token. This allows the model to learn that technology stocks and utility stocks respond differently to the same market conditions.

Unlike the other three models which predict absolute prices, the Transformer predicts **returns** (percentage changes) and uses **pinball loss** to simultaneously predict three quantiles: p10, p50, and p90. This gives a calibrated confidence range without any approximation — the model is directly trained to produce accurate uncertainty estimates.

- Input: 756 days of 36 features, sector label
- Output: p10, p50, p90 return predictions for each horizon (3 quantiles × 3 horizons)
- Training time: 32–56 minutes on GPU (3 CV folds + final model)
- File: `transformer_final/train_transformer.py`

### Why Different Window Sizes

| Model | Window | Reason |
|---|---|---|
| Transformer | 756 days (3 years) | Self-attention uses the full window with equal access to all positions |
| LSTM | 756 days (3 years) | Gates preserve long-range signals — same window as Transformer for a fair comparison |
| RNN | 252 days (1 year) | Vanishing gradients make anything beyond ~50 steps invisible regardless of window length |
| Random Forest | 252 days (1 year) | Flattened input grows linearly with window size — longer windows add noise without adding temporal understanding |

---

## How the Data Pipeline Works

### Step 1: Downloading the data

`build_dataset.py` downloads 10 years of daily stock data for all S&P 500 companies from Yahoo Finance using the `yfinance` library. The S&P 500 ticker list is fetched fresh from Wikipedia each time the script runs, so the dataset automatically includes newly added companies.

The download uses 32 parallel threads. Downloading all ~500 stocks sequentially would take over 30 minutes. In parallel it takes under 3 minutes.

After downloading, each ticker goes through a cleaning step that removes bad rows (zero prices, corrupted data, missing values) and computes 36 technical indicators from the raw OHLCV data.

### Step 2: Building the sliding windows

The cleaned data is converted into overlapping training windows. For a 756-day window, every consecutive stretch of 756 trading days in a ticker's history becomes one training sample. The target (what the model is trying to predict) is the stock's closing price 5, 21, and 126 trading days after the window ends.

This produces:
- `windows_756.npz`: 657,035 windows, shape (657035, 756, 36), for the Transformer and LSTM
- `windows_252.npz`: 875,771 windows, shape (875771, 252, 36), for the RNN and Random Forest

### Step 3: Pre-scaling (Transformer and LSTM only)

The Transformer and LSTM training scripts require a one-time pre-processing step run by `dataset/prescale.py`. This script fits a StandardScaler on a sample of the training data, applies it to all 657,035 windows, and saves the result as `X_756_scaled.npy` in float16 format (half-precision).

This step exists for two reasons. First, normalising features to zero mean and unit standard deviation is essential for neural networks — features on very different scales (Volume in the millions vs RSI between 0 and 100) cause unstable training. Second, doing this scaling inside the DataLoader on every batch would mean the CPU is applying sklearn transforms to 657,000 windows per epoch, keeping the GPU waiting idle. Pre-scaling eliminates this bottleneck entirely.

The RNN's extraction script (`train_rnn.py`) does the same thing automatically on first run, creating `X_seq_252.npy`.

### Train / validation split

All models use a strict **chronological split** — the first 80% of windows (by date) are for training, the last 20% are for validation.

Random splitting is never used in financial machine learning because it causes **data leakage**. If windows from 2024 end up in the training set while windows from 2022 are in the validation set, the model effectively learns from the future. It will report excellent validation metrics but fail entirely on real forward-looking predictions. Chronological splitting correctly simulates real-world conditions where you always train on the past and test on the future.

### Walk-forward cross-validation (Transformer only)

The Transformer goes further than a single train/validation split. It uses **walk-forward cross-validation** with three folds:

- Fold 1: train on all data before 2023, validate on 2023
- Fold 2: train on all data before 2024, validate on 2024
- Fold 3: train on all data before 2025, validate on 2025

Each fold represents a different market regime (post-COVID recovery, election year and rate cuts, current period). Averaging metrics across all three folds gives a much more reliable estimate of real-world accuracy than a single split would.

After all three folds complete, a final model is trained on the full dataset for deployment.

---

## The 36 Features

Every trading day in the dataset is described by 36 numbers. These are computed for each stock independently by `build_dataset.py`.

| Category | Features |
|---|---|
| Price (OHLCV) | Open, High, Low, Close, Volume |
| Trend | SMA 10, SMA 50, SMA 200, MACD, MACD Signal |
| Momentum | RSI 14, Momentum 5, Rate of Change 21, Williams %R |
| Volatility | ATR 14, Bollinger Upper, Bollinger Lower, Bollinger Width |
| Volume | OBV (normalised), Volume SMA 20, Volume Ratio |
| Candlestick patterns | Body size, upper shadow, lower shadow, body percentage, doji flag, hammer flag, shooting star flag, engulfing flag |
| Price structure | % from 52-week high, % from 52-week low, price range % |
| Market context | VIX (fear index), S&P 500 21-day return, S&P 500 63-day return, relative strength vs sector |

The market context features (VIX, S&P 500 returns, relative sector strength) are particularly important for the longer horizons. A stock might look bullish in isolation but if the broader market is in a high-volatility regime, that changes the outlook significantly.

---

## Data Quality Checks

Every ticker and every feature matrix goes through automatic quality checks in `data_guards.py`. Training stops immediately if a critical problem is found.

| Check | What it catches |
|---|---|
| Fewer than 252 rows of history | Not enough data for even one training window — ticker is skipped |
| Closing price of zero or negative | Corrupted data, often from stock splits or delistings |
| More than 5% missing values in a column | Too many gaps — ticker is skipped |
| Single-day price move above 50% | Almost always a data error from unadjusted stock splits |
| NaN or Infinity in the feature matrix | Would silently corrupt model weights — training stops with a clear error message |
| Train set smaller than 10% of total data | Almost all tickers failed to download — catches a broken data build |
| More than 70% of returns pointing in one direction | Means the dataset only covers a bull or bear run, not a balanced market |

---

## Training the Models

Each model has its own training script. They all load from the shared dataset cache and save their weights to their own folder.

The training scripts are extensively commented to explain the concepts behind every design decision. If you are learning about machine learning, reading the scripts is as valuable as running them.

### What happens during training

**Neural network models (Transformer, LSTM, RNN):**

Each script loops through the training data in batches of 128 windows. For each batch:
1. The batch goes through the model (forward pass) to produce predictions
2. The loss function measures how wrong the predictions are
3. Backpropagation computes the gradient of the loss with respect to every weight in the model
4. The optimiser (AdamW) adjusts each weight slightly in the direction that reduces the loss
5. The learning rate scheduler adjusts how large the weight updates are

This repeats for every batch in the dataset. One full pass through all batches is called an epoch. All three neural network models use **early stopping** — if the validation loss does not improve for a set number of epochs (the patience), training stops automatically to prevent overfitting.

**Random Forest:**

The Random Forest does not use iterative training. It builds all 50 trees in one call to `model.fit()`. Each tree is built independently (in parallel using all CPU cores), so the training time depends almost entirely on CPU speed.

### Automatic Mixed Precision (AMP)

The Transformer and LSTM use AMP during training. Modern GPUs have dedicated hardware (Tensor Cores) that perform float16 matrix multiplications 2–3x faster than float32. AMP automatically runs eligible operations in float16 and keeps the rest in float32. The result is faster training with no meaningful loss in accuracy.

### Gradient clipping

All three neural network models clip gradients to a maximum magnitude of 1.0 before applying the weight update. Occasionally, a batch of data with extreme feature values can produce a very large gradient that would cause the model's weights to jump to a bad region of the loss landscape. Clipping prevents this by capping the gradient at a safe size.

### Saved files after training

| Model | Saved files | What they are |
|---|---|---|
| Transformer | `transformer_multi_horizon.pth`, `scaler_feat.pkl`, `scaler_ret.pkl`, `transformer_meta.pkl` | Model weights, feature scaler, target scaler, metadata |
| LSTM | `lstm_multi_horizon.pth`, `lstm_scaler_feat.pkl`, `lstm_scaler_targ.pkl`, `lstm_meta.pkl` | Model weights, feature scaler, target scaler, metadata |
| RNN | `rnn_multi_horizon.pth`, `rnn_scaler_feat.pkl`, `rnn_scaler_targ.pkl`, `rnn_meta.pkl` | Model weights, feature scaler, target scaler, metadata |
| Random Forest | `rf_multi_horizon.pkl`, `feature_list_multi.pkl` | Trained model, feature name list |

---

## How to Run It

### What you need

- Python 3.10 or higher
- Node.js 18 or higher (for the frontend)
- A GPU is strongly recommended for the Transformer and LSTM. Training on CPU is possible but will take many hours.
- About 120 GB of free disk space during training (the datasets are large)
- A good internet connection for the initial data download

### Step 1 — Clone and install

```bash
git clone https://github.com/shiva-shivanibokka/Multi-Horizon-Stock-Forecasting-AI-Model.git
cd Multi-Horizon-Stock-Forecasting-AI-Model
pip install -r requirements.txt
```

### Step 2 — Build the dataset

Run this once from the project root. It downloads 10 years of S&P 500 data for all ~500 companies and builds the shared training datasets. It takes 3–10 minutes depending on your internet connection.

```bash
python build_dataset.py
```

You will see progress as it downloads tickers in parallel. When it finishes you will see a summary showing how many tickers succeeded and the date range covered.

To force a fresh download (e.g. to update with the latest prices):

```bash
python build_dataset.py --refresh
```

### Step 3 — Pre-scale the Transformer and LSTM dataset

This is a one-time step that scales the 756-day dataset and saves it as a memory-mapped float16 file. It takes 30–60 minutes because it processes 71 GB of data.

```bash
python dataset/prescale.py
```

After it completes you can delete `dataset/windows_756.npz` to recover ~35 GB of disk space (the prescaled file is half the size):

```bash
del dataset\windows_756.npz
```

### Step 4 — Train the models

Train the Transformer and LSTM first (they use the GPU). The RNN extraction step runs automatically on first launch and takes about 10 minutes before training begins. Train the Random Forest last since it uses all CPU cores and will slow down everything else.

```bash
# Transformer (~32–56 minutes on GPU, includes 3 CV folds + final model)
cd transformer_final
python train_transformer.py

# LSTM (~20–30 minutes on GPU)
cd lstm_final_project
python train_lstm.py

# RNN (~10 minute extraction + 20–35 minutes training on GPU)
cd rnn_final
python train_rnn.py

# Random Forest (~2–4 hours on CPU, uses all CPU cores)
cd rf_final
python train_rf.py
```

You do not have to wait for all four to finish before starting the app. The backend loads whichever models are ready.

### Step 5 — Start the backend

```bash
python app.py
```

You should see each model load in sequence, followed by:

```
Startup complete.
Running on http://0.0.0.0:5000
```

### Step 6 — Start the frontend

In a new terminal:

```bash
cd nextjs
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

### Optional — View training metrics in MLflow

```bash
mlflow ui --port 5001
```

Then open `http://localhost:5001`. You can see every training run, compare hyperparameters, and inspect loss curves epoch by epoch.

### Optional — Run with Docker

```bash
docker-compose up
```

This starts both the backend (port 5000) and frontend (port 3000) in containers with one command.

---

## API Reference

### GET /api/health

Returns the server status and which models are currently loaded.

```json
{
  "status": "ok",
  "models": ["transformer", "lstm", "rnn", "rf"],
  "device": "cuda"
}
```

### GET /api/predict/`<ticker>`?model=`<model>`

Returns price predictions from one model for all three horizons, with a confidence range and a recommendation.

- `ticker` — any S&P 500 ticker, e.g. `AAPL`, `NVDA`, `MSFT`
- `model` — one of `transformer`, `lstm`, `rnn`, `rf` (default: `transformer`)
- Rate limit: 20 requests per minute per IP address

```json
{
  "ticker": "AAPL",
  "model": "transformer",
  "current_price": 189.30,
  "predictions": {
    "1w": 191.45,
    "1m": 194.20,
    "6m": 205.80
  },
  "p10": {
    "1w": 188.20,
    "1m": 189.00,
    "6m": 192.10
  },
  "p90": {
    "1w": 195.10,
    "1m": 200.40,
    "6m": 221.50
  },
  "recommendation": {
    "recommendation": "BUY",
    "confidence": "High",
    "score": 4,
    "reasons": ["Strong long-term upside (15.4%)"]
  },
  "warnings": []
}
```

### GET /api/predict/all/`<ticker>`

Runs all four models and returns their predictions side by side. Powers the Model Comparison tab.

Rate limit: 5 requests per minute per IP (stricter because it runs all 4 models at once).

### GET /api/sentiment/`<ticker>`

Fetches the last 20 news headlines for the stock and returns an AI sentiment score for each, from -1 (very negative) to +1 (very positive), along with an overall score.

### GET /api/fundamentals/`<ticker>`

Returns key company data: P/E ratio, revenue, net income, profit margin, total debt, beta, and 52-week high and low.

### POST /api/reload

Reloads all model weights from disk without restarting the server. Requires the `X-Reload-Token` header set to the value of the `RELOAD_TOKEN` environment variable. Used by the automated retraining workflow.

---

## Experiment Tracking with MLflow

All four training scripts are wired up to log to MLflow. Every time you run a training script, it creates a new run inside an experiment and logs the following:

- All hyperparameters (batch size, learning rate, window size, number of layers, etc.)
- Training loss and validation loss after every epoch
- MAE and directional accuracy per forecast horizon at the end of training
- The trained model file as a downloadable artifact

To view what has been logged, run:

```bash
mlflow ui --port 5001
```

Then open `http://localhost:5001` in your browser.

| Model | MLflow experiment name |
|---|---|
| Transformer | `stock-forecasting-transformer` |
| LSTM | `stock-forecasting-lstm` |
| RNN | `stock-forecasting-rnn` |
| Random Forest | `stock-forecasting-rf` |

A note on what you will see right now: the mlflow folders currently contain only the model artifact files (the `.pth` and `.pkl` weights), not full metric logs. The reason is a breaking change in MLflow 3.x — it changed its default tracking store from a flat file system to SQLite, which requires a database file to exist before any metrics can be written. Since that file was never created, every `log_params` and `log_metrics` call during training silently failed. Only `log_artifact` worked because it bypasses the tracking store and copies files directly.

This has been fixed in all four training scripts by adding `mlflow.set_tracking_uri("file:./mlruns")` before any mlflow calls, which forces the flat file store that works without any database setup. The next time each model trains, params and metrics will be logged correctly and visible in the MLflow UI.

---

## Automated Weekly Retraining

The models retrain every Sunday at 2am UTC using GitHub Actions, completely automatically and at no cost.

The workflow:
1. GitHub starts a fresh machine and checks out the code
2. `build_dataset.py --refresh` downloads the latest 10 years of S&P 500 data
3. All four models are retrained in sequence on the new data
4. The new model checkpoint files are committed back to the repository
5. The production backend is sent a signal to reload the new weights without restarting

You can trigger a manual retrain at any time from the GitHub Actions tab without waiting for Sunday.

To retrain locally:

```bash
python retrain.py                   # retrain all 4 models
python retrain.py --model transformer  # retrain just the Transformer
python retrain.py --refresh-data    # download fresh data first, then retrain all
```

---

## Deployment

The full stack can be hosted for free.

| Service | What it runs | Free tier |
|---|---|---|
| Render | Flask backend (Python) | Yes — sleeps after 15 min of inactivity |
| Vercel | Next.js frontend | Yes — no sleep |
| GitHub Actions | Weekly retraining job | Yes — 2,000 minutes per month |

### Backend on Render

1. Create a new Web Service at render.com and connect your GitHub repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `python app.py`
4. Add these environment variables in Render Settings:
   - `RELOAD_TOKEN` — any secret string you choose (protects the reload endpoint)
   - `RENDER_DEPLOY_HOOK` — your Render deploy hook URL (found in Render Settings)

### Frontend on Vercel

1. Go to vercel.com, create a new project, and import your GitHub repo
2. Set the Root Directory to `nextjs`
3. Add environment variable: `NEXT_PUBLIC_API_URL` = your Render backend URL (e.g. `https://your-app.onrender.com`)
4. Deploy — Vercel redeploys automatically on every push to `main`

### Retraining secret for GitHub Actions

Go to your GitHub repo → Settings → Secrets and variables → Actions → New repository secret:

- Name: `RENDER_DEPLOY_HOOK`
- Value: your Render deploy hook URL

The retraining workflow uses this to signal the backend to reload models after training completes.

---

## Results

**What the metric means:** directional accuracy measures whether the model correctly predicted whether the stock price would go up or down relative to the last known close price. This is the metric that matters most for actual trading decisions — a model that predicts the right magnitude but the wrong direction is not useful. 50% would mean the model is no better than a coin flip. In financial machine learning using only historical price data, anything consistently above 55–60% is considered a meaningful signal.

### Transformer (PatchTST)

Evaluated using `evaluate_transformer.py` against the three walk-forward CV fold test windows (2023, 2024, 2025). The +/- values show variance across the three test periods — a narrow range means the model performs consistently across different market regimes.

| Horizon | MAE | Directional Accuracy |
|---|---|---|
| 1 week | 5.22 +/- 1.01 | 56.9% +/- 1.1% |
| 1 month | 9.95 +/- 1.59 | 59.7% +/- 1.1% |
| 6 months | 21.34 +/- 2.71 | 66.1% +/- 6.9% |

Note: these numbers use the production model weights, which were trained on all available data including the test windows. They are therefore slightly optimistic compared to a true holdout evaluation. The per-fold CV models would give more conservative numbers but those runs had numerical instability issues during training.

The 6-month directional accuracy (66.1%) being higher than 1-week (56.9%) is expected. Short-term price moves are dominated by random noise and news events that cannot be predicted from price history alone. Longer-horizon moves are more influenced by fundamental trends and sector cycles that the model's 3-year lookback window can detect.

### LSTM, RNN, Random Forest

Results will be added here once those models finish their current training runs.

---

## License

This code is available for viewing and educational purposes only. You may not use, copy, modify, or distribute it without written permission.
