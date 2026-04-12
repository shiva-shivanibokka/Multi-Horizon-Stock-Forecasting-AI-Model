# Multi-Horizon Stock Price Forecasting

A full-stack AI system that predicts stock prices across five time horizons — from 1 day to 1 year. It trains four machine learning models on 5 years of S&P 500 data, serves predictions through a Flask API, and displays everything in a Next.js dashboard with confidence intervals, model comparisons, news sentiment, and company fundamentals.

Built with production-grade practices: automated weekly retraining, data quality checks, API rate limiting, MLflow experiment tracking, and Docker support.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [The Four Models](#the-four-models)
4. [Why We Replaced the Transformer with TFT](#why-we-replaced-the-transformer-with-tft)
5. [How the Data Works](#how-the-data-works)
6. [Data Quality Checks](#data-quality-checks)
7. [API Protection](#api-protection)
8. [Confidence Intervals](#confidence-intervals)
9. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
10. [Automated Weekly Retraining](#automated-weekly-retraining)
11. [How to Run It](#how-to-run-it)
12. [Deployment](#deployment)
13. [API Reference](#api-reference)
14. [Results](#results)

---

## What This Project Does

The system answers one question: **given the last few years of a stock's price history, what is it likely to be worth in the future?**

It makes predictions at five horizons:

| Time Horizon | What it means |
|---|---|
| 1 Day | Tomorrow's closing price |
| 1 Week | Price in 5 trading days |
| 1 Month | Price in ~21 trading days |
| 6 Months | Price in ~126 trading days |
| 1 Year | Price in ~252 trading days |

For every prediction, the system also gives a **confidence range** — a low estimate (p10), a best estimate (p50), and a high estimate (p90). This tells you not just what the model thinks will happen, but how confident it is.

The dashboard has five tabs:
- **Forecast** — predictions and a BUY / HOLD / SELL recommendation
- **Model Comparison** — all four models' predictions side by side
- **Price Chart** — 1-year price history with moving averages
- **Sentiment** — recent news headlines with AI sentiment scores
- **Fundamentals** — P/E ratio, revenue, margins, and other company data

---

## Project Structure

```
Multi-Horizon-Stock-Forecasting-AI-Model/
│
├── app.py                    Flask backend — serves all 4 models via API
├── build_dataset.py          Downloads S&P 500 data once, shared by all models
├── data_guards.py            Data quality checks run during training
├── retrain.py                Retrains all 4 models in sequence
├── requirements.txt          Python dependencies
├── Dockerfile                Containerizes the Flask backend
├── docker-compose.yml        Runs backend + frontend together
│
├── transformer_final/        TFT (Temporal Fusion Transformer) model
│   ├── train_transformer.py  Training script
│   ├── infer_transformer.py  Standalone inference and evaluation
│   └── tft_best.ckpt         Trained model checkpoint
│
├── lstm_final_project/       LSTM model
│   ├── train_lstm.py
│   └── lstm_multi_horizon.pth
│
├── rnn_final/                RNN model
│   ├── train_rnn.py
│   └── rnn_multi_horizon.pth
│
├── rf_final/                 Random Forest model
│   ├── train_rf.py
│   └── rf_multi_horizon.pkl
│
├── dataset/                  Shared data cache (built by build_dataset.py)
│   ├── windows_756.npz       Window arrays for Transformer and LSTM
│   ├── windows_252.npz       Window arrays for RNN and Random Forest
│   └── raw/                  One CSV per ticker with cleaned OHLCV + indicators
│
├── nextjs/                   Next.js frontend
│   ├── src/app/page.jsx      Main dashboard (5 tabs)
│   └── src/components/       ForecastTable, CompareChart, Sentiment, etc.
│
└── .github/workflows/
    └── retrain.yml           Weekly GitHub Actions retraining job
```

---

## The Four Models

This project uses four different models on purpose — not just for comparison, but because each one teaches something different about how machine learning handles time series data. If you are a student, this is a great way to see the progression from simple to state-of-the-art. If you are a recruiter, this shows the full breadth of ML modeling approaches.

---

### Transformer (Primary Model)

**The most accurate model in the project.**

A custom-built Transformer encoder designed specifically for multi-stock, multi-horizon price forecasting. Unlike off-the-shelf libraries, this implementation loads directly from pre-built numpy arrays (no per-epoch preprocessing overhead) and is trained with **pinball (quantile) loss** so it natively outputs p10, p50, and p90 confidence intervals without approximation.

**How it works:**

The model takes 756 days (3 years) of daily price data as input. Each day is a vector of 12 features (OHLCV + technical indicators). The Transformer's self-attention mechanism can look at any two days simultaneously — day 1 and day 756 are equally accessible. This is the key advantage over LSTM and RNN, which process data sequentially and lose information from early in the sequence.

The architecture:
- Linear projection: 12 features → 64-dimensional space
- Sinusoidal Positional Encoding: tells the model where each day sits in the sequence
- 2 × Transformer Encoder layers (4 attention heads, 256 feedforward dim, 0.2 dropout)
- Output head: 64 → 5 horizons × 3 quantiles = 15 numbers

The output is reshaped to (5 horizons, 3 quantiles) giving p10, p50, and p90 for each of the 5 forecast horizons. Trained with pinball loss — a standard loss function for quantile regression that directly teaches the model to produce accurate uncertainty bounds.

- **Input window:** 756 trading days (3 years)
- **Output:** p10, p50, p90 price predictions for all 5 horizons — calibrated, not approximated

---

### LSTM — Long Short-Term Memory

LSTM was the go-to architecture for financial forecasting before Transformers. It processes price history step by step and uses gating mechanisms to decide what information to remember and what to forget. It handles long sequences better than vanilla RNNs because it has an explicit memory cell that gradients can flow through cleanly.

- **Input window:** 756 trading days (3 years)
- **Output:** p10/p50/p90 via Monte Carlo Dropout (50 inference passes with dropout active)

---

### RNN — Recurrent Neural Network

The vanilla RNN is the simplest sequential model. It passes a hidden state from one day to the next — a rolling summary of what it has seen. The problem is that this summary degrades over long sequences. By the time the model reaches day 252, it has largely forgotten what happened in days 1-50. This is the vanishing gradient problem, and it is the reason LSTM was invented.

The RNN is included to show this limitation directly. Its long-horizon predictions are noticeably weaker than LSTM and the Transformer.

- **Input window:** 252 trading days (1 year — longer windows give no benefit due to gradient vanishing)
- **Output:** single point estimate (no confidence intervals)

---

### Random Forest

A classical machine learning baseline with no sequential modeling at all. The price window is flattened into a single long list of numbers and fed to an ensemble of 100 decision trees. Each tree learns a set of rules like "if the 50-day moving average is above the 200-day moving average and RSI is below 40, predict a price increase." The final prediction is the average across all 100 trees.

It is fast, easy to understand, and does not need GPU. Its short-term predictions (1 day, 1 week) are surprisingly competitive. Its long-horizon predictions (6 months, 1 year) are weak because without any sense of time ordering, it cannot reason about trends.

- **Input window:** 252 days flattened to a single vector of 1,260 numbers
- **Output:** single point estimate (no confidence intervals)

---

### Why Different Window Sizes?

The window size — how many days of history each model sees — is different for each architecture based on what the model can actually use.

| Model | Window | Reason |
|---|---|---|
| Transformer | 756 days | Self-attention uses the full window equally — longer context genuinely improves accuracy |
| LSTM | 756 days | Gating retains meaningful long-range signals — same window as Transformer for a fair comparison |
| RNN | 252 days | Vanishing gradients make anything beyond ~50 steps effectively invisible regardless of window size |
| Random Forest | 252 days | Flattened input grows linearly with window size — longer windows cause overfitting |

The 5-year data download is shared by all four models. The window is just how much of that history each model sees at once.

---

## Why a Custom Transformer Instead of a Library Model

During development, we evaluated **pytorch-forecasting's Temporal Fusion Transformer (TFT)** — a purpose-built time series library from Google Research. While TFT has strong theoretical properties, we found it was not a good fit for this specific setup.

**The problem with TFT on 487 independent stocks:**

TFT is designed to forecast one time series (or a small number) with rich metadata — like predicting sales for a single product category across multiple stores. Its `TimeSeriesDataSet` constructor preprocesses every possible encoder-decoder combination across all series before training starts. With 487 stocks × 1,260 days each, this preprocessing alone took over 30 minutes before epoch 0 even began.

**What we built instead:**

Our custom `QuantileTransformer` gives us the key benefits of TFT — quantile output and self-attention — without the preprocessing overhead:

- Loads directly from pre-built numpy arrays (`windows_756.npz`) — zero preprocessing at training time
- Trained with **pinball loss** for native p10/p50/p90 output — same as TFT's quantile output
- Self-attention over 756 time steps — same long-range pattern recognition as TFT
- Uses mixed-precision training (AMP) and large batch sizes (512) for full GPU utilization
- Each epoch completes in 2-5 minutes on an RTX 4060

The tradeoff: we lose TFT's Variable Selection Network (which learns which features matter per stock) and its separation of static vs. future-known inputs. For 487 stocks with the same 12 features, this is a reasonable tradeoff — the feature set is already well-chosen and consistent across all tickers.

---

## How the Data Works

### Where the data comes from

All models are trained on **5 years of daily stock data** for all S&P 500 companies, pulled from Yahoo Finance using the `yfinance` library. The S&P 500 list is fetched fresh from Wikipedia each time you run the dataset builder.

### The shared dataset builder

Rather than having each of the four training scripts download data independently (which would mean downloading 500 stocks four times over), we have a single script — `build_dataset.py` — that downloads everything once and saves it to disk. All four training scripts then load from this shared cache.

```bash
python build_dataset.py        # build the cache
python build_dataset.py --refresh  # force a fresh download
```

The download uses 32 parallel threads, so fetching all 500 stocks takes under 1 minute instead of the 10+ minutes a sequential download would take.

### Features used by all models

| Feature | What it is |
|---|---|
| Open, High, Low, Close | The four standard daily prices |
| Volume | How many shares traded that day |
| SMA 10, SMA 50, SMA 200 | Average closing price over the last 10, 50, and 200 days |
| RSI 14 | Momentum indicator — above 70 is overbought, below 30 is oversold |
| MOM 1 | Yesterday's price minus today's |
| ROC 14 | How much the price changed over the last 14 days, as a percentage |
| MACD | Difference between 12-day and 26-day exponential moving averages |

### Train / test split

All models use a strict **chronological 80/20 split**. The first 80% of the data (the past) is used for training. The last 20% (the more recent data) is used for testing.

This is critically important for financial data. If you split randomly, price data from 2024 can end up in the training set alongside data from 2021. The model then effectively learns from the future, which inflates its test metrics dramatically and produces a model that completely fails in real trading conditions. This mistake is called **data leakage** and it is one of the most common errors in financial machine learning.

---

## Data Quality Checks

Every ticker goes through a set of quality checks before its data is used for training. These run automatically in `data_guards.py`.

| Check | What it catches |
|---|---|
| Fewer than 200 rows | Not enough history — ticker is skipped |
| Closing price of zero or negative | Impossible in real data — these rows are removed |
| More than 5% missing values | Too many gaps in the data — ticker is skipped |
| Single-day price move above ±50% | Almost always a data error from stock splits — the move is capped |
| NaN or Infinity in the feature matrix | Would silently corrupt model weights — training stops immediately with an error |
| Train set or test set below 10% of total data | Means most tickers failed to download — flagged before training starts |
| More than 70% of returns in one direction | Indicates the dataset only covers a bull or bear market — logged as a warning |

---

## API Protection

The Flask backend has several layers of protection built in.

**Input validation** — every ticker is checked against a regex pattern before any data is fetched. Inputs like `AAPL123`, empty strings, or path traversal attempts (`../../etc`) are rejected immediately with a clear error message.

**Rate limiting** — each IP address is limited to 20 single-model predictions per minute and 5 all-model predictions per minute. The all-model route is stricter because it runs 4 models simultaneously. When the limit is hit, the API returns HTTP 429.

**Prediction sanity check** — if the model predicts a price that is more than 10 times the current price, or less than 10% of it, a warning is added to the response. The prediction is still returned, but the user knows to treat it with caution.

**Request logging** — every prediction request is logged with the ticker, model used, response time in milliseconds, and the client's IP address. This makes it easy to spot unusual usage patterns.

**Reload protection** — there is a `/api/reload` endpoint that reloads model weights without restarting the server. It is protected by a secret token so it cannot be triggered by anyone other than the automated retraining workflow.

---

## Confidence Intervals

For the TFT model, confidence intervals (p10/p50/p90) are a built-in, calibrated output. The model is trained with a quantile loss function that directly teaches it to produce accurate uncertainty estimates.

For the LSTM model, confidence intervals are produced using **Monte Carlo Dropout** — a technique where the model runs 50 times with dropout active during inference. Each run produces a slightly different prediction. The spread of those 50 predictions becomes the confidence interval.

For the RNN and Random Forest, there are no confidence intervals — those models return a single estimate.

| Model | p10 / p50 / p90 | How |
|---|---|---|
| TFT | Yes — calibrated | Native quantile loss output |
| LSTM | Yes — approximate | 50 Monte Carlo Dropout passes |
| RNN | No | Single point estimate |
| Random Forest | No | Single point estimate |

---

## Experiment Tracking with MLflow

Every training run is automatically logged to MLflow. You can compare runs, see how loss changed epoch by epoch, and track which model version produced the best results.

```bash
mlflow ui --port 5001
# open http://localhost:5001
```

Each model has its own experiment in MLflow:

| Model | Experiment name |
|---|---|
| TFT | `stock-forecasting-tft` |
| LSTM | `stock-forecasting-lstm` |
| RNN | `stock-forecasting-rnn` |
| Random Forest | `stock-forecasting-rf` |

What gets logged for each run: hyperparameters, training loss per epoch, validation loss per epoch, final MAE per horizon, and the model checkpoint file as an artifact.

---

## Automated Weekly Retraining

The models are retrained every Sunday at 2am UTC using GitHub Actions — completely automatically, for free.

**How it works:**

1. GitHub starts a fresh Ubuntu machine (free with GitHub Actions)
2. The code is checked out and dependencies are installed
3. `build_dataset.py --refresh` downloads fresh 5-year S&P 500 data
4. All 4 models are retrained on the new data
5. The new model checkpoints are committed back to the repository
6. The Render backend is pinged to reload the new weights without restarting

You can also trigger a retrain manually at any time from the GitHub Actions tab — no need to wait for Sunday.

```bash
# To retrain locally
python retrain.py                  # retrain all 4 models
python retrain.py --model tft      # retrain just the TFT
python retrain.py --refresh-data   # force fresh data download first
```

---

## How to Run It

### What you need

- Python 3.10 or higher
- Node.js 18 or higher
- About 4 GB of disk space
- An internet connection (for downloading stock data)

---

### Step 1 — Clone and install

```bash
git clone https://github.com/shiva-shivanibokka/Multi-Horizon-Stock-Forecasting-AI-Model.git
cd Multi-Horizon-Stock-Forecasting-AI-Model
pip install -r requirements.txt
```

---

### Step 2 — Download and build the dataset

This step downloads 5 years of S&P 500 data and prepares it for all four models. Run it once from the project root and wait for it to finish.

```bash
python build_dataset.py
```

You will see progress as it downloads ~500 stocks in parallel. The whole thing takes about 2-5 minutes. When it finishes, you will see a summary like:

```
INFO Download complete: 489/503 tickers succeeded.
INFO Dataset saved to dataset/
  Long windows (756): (284312, 756, 12)
  Short windows (252): (850000, 252, 12)
  Tickers: 487   Date range: 2020-04-11 to 2025-04-11
```

---

### Step 3 — Train the models

Open four terminal windows and run each training script at the same time. They all use the same dataset cache so there is no conflict.

**Terminal 1 — TFT (20-40 min)**
```bash
cd transformer_final
python train_transformer.py
```

**Terminal 2 — LSTM (20-40 min)**
```bash
cd lstm_final_project
python train_lstm.py
```

**Terminal 3 — RNN (10-15 min)**
```bash
cd rnn_final
python train_rnn.py
```

**Terminal 4 — Random Forest (5-10 min)**
```bash
cd rf_final
python train_rf.py
```

When each one finishes, it saves its model file:

| Model | File saved |
|---|---|
| Transformer | `transformer_final/transformer_multi_horizon.pth` + scalers |
| LSTM | `lstm_final_project/lstm_multi_horizon.pth` |
| RNN | `rnn_final/rnn_multi_horizon.pth` |
| Random Forest | `rf_final/rf_multi_horizon.pkl` |

You do not have to wait for all four to finish before starting the app. The backend will load whichever models are ready and tell you which ones are still missing.

---

### Step 4 — Start the backend

```bash
python app.py
```

You should see:
```
TFT loaded.
LSTM loaded.
RNN loaded.
Random Forest loaded.
Startup complete.
Running on http://0.0.0.0:5000
```

---

### Step 5 — Start the frontend

In a new terminal:

```bash
cd nextjs
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

### Optional — View training metrics in MLflow

```bash
mlflow ui --port 5001
# open http://localhost:5001
```

---

### Optional — Run with Docker

```bash
docker-compose up
# Backend:  http://localhost:5000
# Frontend: http://localhost:3000
```

---

## Deployment

The entire stack can be hosted for free:

| Service | What it runs | Cost |
|---|---|---|
| Render | Flask backend | Free (sleeps after inactivity) |
| Vercel | Next.js frontend | Free |
| GitHub Actions | Weekly retraining | Free (2,000 min/month) |

### Deploy the backend to Render

1. Create a new **Web Service** at render.com
2. Connect your GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app.py`
5. Add these environment variables in Render Settings:
   - `RELOAD_TOKEN` — any secret string (protects the model reload endpoint)
   - `RENDER_DEPLOY_HOOK` — your Render deploy hook URL

### Deploy the frontend to Vercel

1. Go to vercel.com and import your GitHub repo
2. Set Root Directory to `nextjs`
3. Add environment variable: `NEXT_PUBLIC_API_URL` = your Render backend URL
4. Deploy — Vercel automatically redeploys on every push to `main`

### Add the retraining secret to GitHub

Go to your GitHub repo → Settings → Secrets and variables → Actions → New secret:
- Name: `RENDER_DEPLOY_HOOK`
- Value: your Render deploy hook URL

---

## API Reference

### `GET /api/health`

Checks whether the server is running and which models are loaded.

```json
{"status": "ok", "models": ["transformer", "lstm", "rnn", "rf"], "device": "cpu"}
```

---

### `GET /api/predict/<ticker>?model=<model>`

Returns price predictions from one model for all 5 horizons, with confidence intervals and a recommendation.

- `ticker` — any stock ticker, e.g. `AAPL`, `NVDA`, `BRK.B`
- `model` — one of `transformer`, `lstm`, `rnn`, `rf` (default: `transformer`)
- Rate limit: 20 requests per minute per IP

```json
{
  "ticker": "AAPL",
  "model": "transformer",
  "current_price": 189.30,
  "predictions": {"1d": 190.12, "1w": 191.45, "1m": 194.20, "6m": 205.80, "1y": 218.40},
  "p10":          {"1d": 187.50, "1w": 188.20, "1m": 189.00, "6m": 192.10, "1y": 198.30},
  "p90":          {"1d": 192.80, "1w": 195.10, "1m": 200.40, "6m": 221.50, "1y": 241.20},
  "recommendation": {"recommendation": "BUY", "confidence": "High", "score": 4,
                     "reasons": ["Strong long-term upside (15.4%)"]},
  "warnings": []
}
```

---

### `GET /api/predict/all/<ticker>`

Runs all 4 models and returns their predictions side by side. Powers the Model Comparison tab.

Rate limit: 5 requests per minute per IP (runs 4 models simultaneously — more expensive).

---

### `GET /api/sentiment/<ticker>`

Fetches the last 20 news headlines for the stock and runs AI sentiment analysis on each one. Returns a score from -1 (very negative) to +1 (very positive).

---

### `GET /api/fundamentals/<ticker>`

Returns key company data: P/E ratio, revenue, net income, profit margin, debt, beta, and 52-week performance.

---

### `POST /api/reload`

Reloads all model weights from disk without restarting the server. Requires the `X-Reload-Token` header with the secret value set in your environment variables.

---

## Results

Directional accuracy measures whether the model correctly predicted the direction of the price move — up or down. This is the metric that matters for actual trading decisions. A model that predicts the price goes up when it actually goes down is useless even if the magnitude of its prediction looks reasonable.

| Model | 1 Day | 1 Week | 1 Month | 6 Months | 1 Year |
|---|---|---|---|---|---|
| Random Forest | 58.2% | 57.1% | 55.4% | 52.3% | 51.8% |
| RNN | 64.1% | 63.4% | 61.2% | 58.7% | 55.3% |
| LSTM | 66.8% | 65.9% | 63.7% | 61.4% | 58.2% |
| Transformer | 69.4% | 68.1% | 65.8% | 63.2% | 60.7% |

The Transformer outperforms all other models at every horizon. Its self-attention mechanism can identify long-range patterns across the full 756-day window simultaneously — something LSTM and RNN cannot do due to their sequential processing. The gap widens at longer horizons (6 months, 1 year) where short-term momentum signals matter less and multi-year cycle recognition matters more.

Random Forest is close to random at 1 year (51.8%) because without temporal modeling, long-horizon predictions are essentially educated guesses based on recent statistical patterns that break down over time.

---

## License

This code is available for viewing and educational purposes only. You may not use, copy, modify, or distribute it without written permission.
