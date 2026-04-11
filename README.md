# Multi-Horizon Stock Price Forecasting

An end-to-end stock price forecasting system that trains four machine learning models (Transformer, LSTM, RNN, Random Forest) on S&P 500 data and serves predictions through a Flask API consumed by a Next.js frontend. The system is designed to professional production standards — with data quality guardrails, API rate limiting, uncertainty quantification, MLflow experiment tracking, and weekly automated retraining via GitHub Actions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Models](#models)
4. [Data Pipeline](#data-pipeline)
5. [Data Guardrails](#data-guardrails)
6. [API Guardrails](#api-guardrails)
7. [Uncertainty Quantification](#uncertainty-quantification)
8. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
9. [Automated Retraining](#automated-retraining)
10. [Running Locally](#running-locally)
11. [Deployment](#deployment)
12. [API Reference](#api-reference)
13. [Results](#results)

---

## Project Overview

The system predicts stock prices across five time horizons:

| Horizon | Trading Days |
|---|---|
| 1 Day | 1 |
| 1 Week | 5 |
| 1 Month | 21 |
| 6 Months | 126 |
| 1 Year | 252 |

Four models are trained on the same dataset and feature set, allowing direct comparison of classical machine learning vs. deep learning approaches for financial time series forecasting.

All four models are served from a single Flask backend. The Next.js frontend provides a 5-tab dashboard: forecast with confidence intervals, model comparison, price chart, news sentiment, and company fundamentals.

---

## Repository Structure

```
Multi-Horizon-Stock-Forecasting-AI-Model/
│
├── app.py                          Flask backend (all 4 models + guardrails)
├── data_guards.py                  Shared training-time data quality checks
├── retrain.py                      Standalone retraining script (all 4 models)
├── requirements.txt                Python dependencies
├── Dockerfile                      Container for Flask backend
├── docker-compose.yml              Runs backend + Next.js frontend together
│
├── transformer_final/
│   ├── train_transformer.py        Training script (5y S&P 500, parallel download)
│   ├── infer_transformer.py        Standalone inference and evaluation
│   ├── transformer_multi_horizon.pth   Trained model weights
│   ├── scaler_feat.pkl             Feature scaler
│   ├── scaler_ret.pkl              Return scaler
│   └── transformer_meta.pkl        Window size and horizon metadata
│
├── lstm_final_project/
│   ├── train_lstm.py
│   ├── infer_lstm.py
│   ├── lstm_multi_horizon.pth
│   ├── lstm_scaler_feat.pkl
│   ├── lstm_scaler_targ.pkl
│   └── lstm_meta.pkl
│
├── rnn_final/
│   ├── train_rnn.py
│   ├── infer_rnn.py
│   ├── rnn_multi_horizon.pth
│   ├── rnn_scaler_feat.pkl
│   ├── rnn_scaler_targ.pkl
│   └── rnn_meta.pkl
│
├── rf_final/
│   ├── train_rf.py
│   ├── infer_rf.py
│   ├── rf_multi_horizon.pkl
│   └── feature_list_multi.pkl
│
├── nextjs/                         Next.js frontend
│   ├── src/app/page.jsx            Main page (5-tab dashboard)
│   ├── src/components/
│   │   ├── ForecastTable.jsx       p10/p50/p90 prediction table
│   │   ├── CompareChart.jsx        All 4 models side-by-side bar chart
│   │   ├── PriceChart.jsx          1-year price history with SMA lines
│   │   ├── Sentiment.jsx           VADER news sentiment with article list
│   │   ├── Recommendation.jsx      BUY/HOLD/SELL card with reasoning
│   │   └── Fundamentals.jsx        Company valuation and financial data
│   ├── next.config.js              API proxy to Flask backend
│   └── package.json
│
├── frontend/                       Original React frontend (legacy)
│
└── .github/workflows/
    └── retrain.yml                 Weekly GitHub Actions retraining cron job
```

---

## Models

This project uses four different models covering the full spectrum from classical machine learning to state-of-the-art deep learning. This is intentional — not just for comparison, but because each model reveals something different about the data, and no single model is optimal for every stock, every horizon, or every market condition.

---

### Why these four models specifically

Stock price forecasting is a multi-horizon, multi-asset sequence prediction problem. The challenge is that:

- Different stocks have different volatility regimes, liquidity profiles, and sector dynamics
- Different horizons require different types of pattern recognition — short-term momentum vs. long-term fundamental trends
- The signal-to-noise ratio in financial data is extremely low compared to other domains (speech, images)
- The data distribution is non-stationary — patterns that held in 2020 may not hold in 2025

No single model handles all of these challenges well. Using four models across the classical-to-deep learning spectrum lets you:

1. Use the Random Forest as a **sanity check baseline** — if the deep learning models can't beat it, the pipeline is broken
2. Use the RNN to understand **how much sequential modeling matters** vs. the baseline
3. Use the LSTM to understand **how much gated memory matters** vs. vanilla recurrence
4. Use the Transformer as the **production-grade model** for the best predictions

Together they give a complete picture of model capability vs. complexity tradeoffs for this specific problem.

---

### Transformer

**Why we use it:** The Transformer is the primary production model because it is the only architecture that can simultaneously attend to any two time steps in the 3-year input window regardless of how far apart they are. Financial time series have long-range dependencies — a price pattern from 6 months ago may be more informative than yesterday's price, and the Transformer can learn this directly through self-attention. Recurrent models like LSTM and RNN process data sequentially and accumulate information in a fixed-size hidden state, which degrades over long sequences even with gating.

The Transformer was originally designed for NLP (the paper "Attention Is All You Need", Vaswani et al. 2017) but has since proven highly effective for time series forecasting. It is now the dominant architecture in production forecasting systems at major financial institutions.

**Architecture:**
- Input: `(756 days, 12 features)` — 3 years of daily OHLCV + technical indicators
- Linear embedding layer → 64-dimensional feature space per time step
- Sinusoidal Positional Encoding — tells the model where each time step sits in the sequence (unlike RNNs, the Transformer has no inherent sense of order)
- 2 × TransformerEncoder layers (4 attention heads, 128-dim feedforward, 0.2 dropout)
- Dense regression head → 5 outputs (one per horizon)
- Trained on **log returns** rather than absolute prices. Log returns are stationary and have consistent scale across different-priced stocks — a $1 move in a $10 stock is very different from a $1 move in a $500 stock. Training on returns removes this scale problem.

**Training details:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4) — weight decay acts as L2 regularization to prevent overfitting
- Mixed-precision training (AMP) for faster GPU training with no accuracy loss
- EarlyStopping with patience=5 on validation MSE — stops when the model stops improving
- ReduceLROnPlateau scheduler — reduces learning rate when training stalls

**Pros:**
- Self-attention can learn which specific historical time steps are most relevant for each prediction horizon — this is impossible for RNNs and LSTMs
- No sequential bottleneck — information from day 1 and day 756 are equally accessible to every layer
- Scales well with more data — performance consistently improves as training set size grows
- 4 attention heads learn different types of patterns simultaneously (e.g. one head may learn weekly seasonality while another learns sector momentum)
- Monte Carlo Dropout provides calibrated uncertainty estimates (p10/p50/p90) because dropout is applied inside the TransformerEncoder layers

**Cons:**
- Computationally expensive — self-attention is O(n²) in sequence length. A 756-step window means 756² = 571,536 attention computations per layer per sample
- Requires significantly more training data than RNN/LSTM to generalize well — works here because we train on the full S&P 500 across 5 years
- No built-in inductive bias for sequential data — the positional encoding is a workaround, not a native property
- Harder to interpret than Random Forest — it is not obvious which attention heads are learning what
- More hyperparameters to tune (number of heads, layers, d_model, feedforward dim)

---

### LSTM (Long Short-Term Memory)

**Why we use it:** The LSTM was chosen as the second deep learning model because it was, for most of the 2010s, the standard architecture for financial time series forecasting before Transformers. It is still widely used in production systems that were built before Transformers became practical. Including it here allows a direct apples-to-apples comparison: does the additional complexity of self-attention actually improve forecasting accuracy over a well-tuned LSTM? The answer from our results is yes — but not by as much as you might expect for shorter horizons, which is itself an important finding.

The LSTM extends the vanilla RNN with three gating mechanisms: an **input gate** (how much new information to store), a **forget gate** (how much past information to discard), and an **output gate** (how much of the cell state to expose as output). This gives the LSTM an explicit mechanism for long-term memory, unlike the RNN which relies entirely on gradient backpropagation to retain information.

**Architecture:**
- Input: `(756 days, 12 features)` — same 3-year window as the Transformer for a fair comparison
- 2 stacked LSTM layers (128 hidden units each) — stacking allows the second layer to learn higher-level temporal abstractions from the first layer's output
- Dropout (0.2) between layers — regularization to prevent the model from memorizing specific training sequences
- Final hidden state → Dense(5) — the last hidden state is a 128-dimensional summary of the entire 756-step sequence
- Trained on **absolute prices** (not returns) — a deliberate contrast with the Transformer to explore whether the target representation affects final accuracy

**Pros:**
- Better long-range memory than vanilla RNN — the forget gate allows the model to selectively retain information from hundreds of steps ago
- Well-understood and well-studied — decades of research on optimal hyperparameters for financial time series
- Faster to train than Transformer on CPU (O(n) vs O(n²) in sequence length)
- Naturally handles variable-length sequences
- Monte Carlo Dropout works well because dropout is applied between LSTM layers

**Cons:**
- Information bottleneck — the entire 756-step history is compressed into a single 128-dimensional hidden state. Some information is inevitably lost, especially for very long-range patterns
- Still sequential processing — cannot parallelize across the time dimension during training (unlike Transformer), making it slower on GPU than it could be
- Gradient flow is better than RNN but still degrades over very long sequences — the forget gate can "forget too much" on 756-step windows
- Gating mechanism adds complexity but the gates themselves can be hard to interpret
- Sensitive to the scale of inputs — requires careful normalization (StandardScaler is critical here)

---

### RNN (Recurrent Neural Network)

**Why we use it:** The vanilla RNN is included as a middle ground between the non-sequential Random Forest and the gated LSTM. It demonstrates concretely what happens when you add temporal modeling but without long-term memory management. The RNN's performance gap vs. the LSTM shows exactly how much the gating mechanism is worth — and the RNN's performance gap vs. Random Forest shows how much sequential modeling is worth at all.

In practice, vanilla RNNs are rarely used for financial forecasting in production anymore. They are included here for completeness of the model hierarchy and for educational value in the model comparison tab.

The RNN processes the input sequence step by step, passing a hidden state from one time step to the next. The hidden state is updated at each step using the formula:

```
h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
```

The gradient of this operation, when backpropagated across 252 steps, is multiplied by the same weight matrix 252 times. If the largest eigenvalue of this matrix is less than 1, gradients shrink exponentially toward zero (vanishing gradient). If it is greater than 1, they explode. This is the fundamental limitation that LSTM was designed to solve.

**Architecture:**
- Input: `(252 days, 12 features)` — 1-year window, shorter than LSTM/Transformer because the vanishing gradient problem makes longer windows counterproductive
- 2 stacked RNN layers (128 hidden units each, tanh activation)
- Final hidden state → Dense(5)
- No dropout in the recurrent stack — dropout in RNNs requires special handling (Zoneout or variational dropout) that was not implemented here

**Pros:**
- Fastest deep learning model to train — simple recurrent operation, no gating overhead
- Lowest memory footprint of the three neural models
- Good short-term accuracy — for 1-day and 1-week horizons where the relevant signal is in recent data, the RNN performs close to LSTM
- Simplest to understand conceptually — the hidden state is a rolling summary of recent history

**Cons:**
- Vanishing gradient problem — information from early in the 252-step window effectively disappears by the time it reaches the output. The model is functionally using only the last 30-50 time steps regardless of the stated window size
- No long-term memory — cannot retain signals from more than a few weeks ago, making it unsuitable for 6-month and 1-year horizon forecasting
- No uncertainty quantification — no dropout means no Monte Carlo Dropout confidence intervals. All p10/p50/p90 values are identical (single-point estimate)
- Performance degrades sharply for longer horizons compared to LSTM and Transformer
- Cannot be parallelized during training

---

### Random Forest

**Why we use it:** The Random Forest is included as a classical machine learning baseline for three specific reasons:

1. **Sanity check** — if Random Forest outperforms the neural models, something is wrong with the neural training pipeline (data leakage, bad normalization, insufficient data). This actually happened in an earlier version of this project where random shuffling caused data leakage — the RF's surprisingly good performance flagged the problem.

2. **Interpretability** — Random Forest feature importance scores reveal which time steps and OHLCV features drive predictions, providing insight that the neural models cannot easily offer.

3. **Non-stationarity robustness** — tree-based models can sometimes outperform neural models on financial data during regime changes (e.g. sudden market crashes) because they are less reliant on learned gradient-based patterns that may not transfer across market regimes.

The Random Forest treats the entire 1-year price window as a flat feature vector, with no awareness of time ordering. Each of the 100 decision trees in the ensemble independently learns a set of feature thresholds to split the data. The final prediction is the mean across all 100 trees, which reduces variance significantly compared to a single tree.

**Architecture:**
- Input: `(252 days × 5 OHLCV features)` flattened to a `(1260,)` vector — time ordering is discarded
- `MultiOutputRegressor` wrapper — fits a separate Random Forest for each of the 5 horizons. This allows each horizon to use a different set of features and split thresholds, rather than sharing a single model across all horizons
- 100 trees, trained in parallel across all available CPU cores (`n_jobs=-1`)

**Pros:**
- No temporal assumptions — makes no assumption about the ordering of the input, which is sometimes an advantage when patterns are more statistical than sequential
- Fast training — 100 trees on 1260 features trains in minutes vs. hours for neural models
- No gradient issues — no vanishing/exploding gradients, no learning rate tuning required
- Naturally handles missing values and outliers better than neural models
- Feature importance scores provide interpretability that neural models lack
- No normalization required — tree splits are scale-invariant
- No risk of overfitting from training too long — trees have a hard maximum depth and the ensemble averages out individual tree overfitting

**Cons:**
- No temporal modeling — flattening the sequence destroys all ordering information. A price of $150 on day 1 and $150 on day 252 are treated identically. This is the fundamental limitation.
- Performance collapses at longer horizons — without temporal patterns, 6-month and 1-year predictions degrade toward the historical mean
- Cannot extrapolate beyond the training distribution — if the market enters a regime it has never seen (e.g. a pandemic), the RF has no mechanism to adapt its predictions
- High memory usage — 100 trees × 1260 features × 500 tickers × 5 years of windows produces a very large model file
- No uncertainty quantification — there is no principled way to produce confidence intervals from a Random Forest without additional calibration

---

### Model comparison summary

| Property | Random Forest | RNN | LSTM | Transformer |
|---|---|---|---|---|
| Temporal modeling | None | Sequential, short-range | Sequential, long-range | Global attention |
| Input window | 252 days | 252 days | 756 days | 756 days |
| Training speed (CPU) | Fast | Medium | Slow | Slowest |
| 1-day accuracy | Moderate | Good | Good | Best |
| 1-year accuracy | Poor | Poor | Good | Best |
| Uncertainty (p10/p90) | No | No | Yes (MC Dropout) | Yes (MC Dropout) |
| Interpretability | High | Low | Low | Very low |
| Data needed | Less | Medium | More | Most |
| Production suitability | Baseline only | Limited | Good | Primary |

---

### Input Features (all models)

| Feature | Type | Why it is included |
|---|---|---|
| Open, High, Low | Price | Intraday range reveals volatility and sentiment |
| Close | Price | Primary target variable and most reliable daily signal |
| Volume | Market activity | Volume confirms or contradicts price moves — a price spike on low volume is less reliable than one on high volume |
| SMA 10 | Short-term trend | 2-week moving average smooths daily noise to reveal near-term direction |
| SMA 50 | Medium-term trend | Standard institutional reference for medium-term trend direction |
| SMA 200 | Long-term trend | The most widely watched moving average on Wall Street — crossovers with SMA 50 (golden/death cross) are major signals |
| RSI 14 | Momentum oscillator | Measures overbought (>70) and oversold (<30) conditions. Values above 70 often precede pullbacks; below 30 often precede bounces |
| MOM 1 | Short momentum | Raw 1-day price change captures immediate market direction |
| ROC 14 | Rate of change | 14-day percentage change standardized across stocks — useful for cross-asset comparison |
| MACD | Trend + momentum | Difference between 12-day and 26-day EMA — captures trend direction changes and momentum shifts |

---

## Data Pipeline

### Training data source

All models are trained on 5 years of daily OHLCV data for all S&P 500 companies, downloaded from Yahoo Finance via `yfinance`. Tickers are fetched from the Wikipedia S&P 500 list at training time.

### Parallel download

Fetching 500 tickers sequentially takes 8-10 minutes. The training scripts use `ThreadPoolExecutor` with 32 concurrent workers to fetch all 500 tickers in parallel — reducing download time to under 1 minute. `yfinance` is I/O-bound (network requests) so thread parallelism works well here.

```python
with ThreadPoolExecutor(max_workers=32) as pool:
    futures = {pool.submit(_download_one, sym): sym for sym in tickers}
```

### Chronological train/test split

**All four models use a strict chronological 80/20 split — no shuffling.**

```python
split = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]   # past trains, future tests
```

Random shuffling causes **data leakage** in financial time series: windows from the future (e.g. 2024 data) can end up in the training set alongside windows from the past (2020 data). The model effectively "sees the future" during training, producing artificially inflated validation metrics that collapse at deployment.

### Windowing

Each ticker's historical data is converted into overlapping fixed-length windows. Each window becomes one training sample:

```
[day 1 → day 756]  →  predict prices at day 757, 761, 777, 882, 1008
[day 2 → day 757]  →  predict prices at day 758, 762, 778, 883, 1009
...
```

---

## Data Guardrails

All training-time quality checks live in `data_guards.py` and are called by all four training scripts. These prevent bad data from silently corrupting the model.

### 1. Per-ticker price validation (`check_price_data`)

Applied to every ticker before any windows are created:

| Check | Action |
|---|---|
| < 200 rows of data | Skip ticker |
| Close price ≤ 0 | Drop those rows |
| > 5% of Close values are NaN | Skip ticker |
| Single-day return > ±50% | Clip the return and reconstruct the Close series |

The ±50% daily return clip is important. Stock splits, reverse splits, and yfinance data errors occasionally produce price jumps of 100-10000% in one day. These are not real market moves — they are data artifacts. If left in, they dominate the loss function during training and severely distort model weights.

### 2. Feature array integrity check (`check_feature_array`)

Called twice: once on the raw feature array and once after StandardScaler normalization.

```python
check_feature_array(X_tr_s, "X_tr (scaled)")
```

If any NaN or Inf values are found, training stops immediately with a clear error message. NaN/Inf values in the input array can propagate silently through the network and corrupt all weight updates without raising any obvious error during training.

### 3. Split sanity check (`check_train_test_split`)

Verifies that both the training and test sets are non-trivial (each must be at least 10% of total data). If most tickers failed to download, the dataset would be too small and the split degenerate.

### 4. Target distribution monitoring (`check_target_distribution`)

Logs a warning if more than 70% of target values are in one direction (all positive or all negative). A model trained only on a bull-market period will be biased toward predicting price increases and will underperform in bear markets.

### 5. Dataset summary logging

Before training begins, logs the full dataset shape, memory usage in MB, and how many tickers successfully contributed data.

```
Dataset ready: tickers=487  windows=284,312  X_shape=(284312, 756, 12)  Y_shape=(284312, 5)  memory=2451.3 MB
```

---

## API Guardrails

All guardrails in the Flask API live in `app.py` before any model inference is triggered.

### 1. Ticker format validation

Every request is validated against `^[A-Z]{1,5}(\.[A-Z]{1,2})?$` before any network calls are made.

```
GET /api/predict/AAPL       → OK
GET /api/predict/BRK.B      → OK
GET /api/predict/AAPL123    → 400: Invalid ticker format
GET /api/predict/../../etc  → 400: Invalid ticker format
GET /api/predict/           → 400: Ticker is required
```

### 2. Rate limiting (per IP, in-memory sliding window)

| Route | Limit | Reason |
|---|---|---|
| `/api/predict/<ticker>` | 20 requests / 60s | Standard inference cost |
| `/api/predict/all/<ticker>` | 5 requests / 60s | 4 models × 50 MC passes = ~200x more expensive |

Returns HTTP 429 with a clear message when exceeded. Uses a sliding-window counter stored in memory per IP. For production at scale, replace with a Redis-backed implementation.

### 3. Model availability check

If a model's checkpoint file was not found at startup (e.g. the RF model has never been trained), the predict function raises a clear error instead of crashing:

```json
{"error": "Random Forest not available. Run: python rf_final/train_rf.py"}
```

### 4. Prediction sanity check

After inference, checks whether any predicted price is more than 10x or less than 10% of the current price. These are almost always caused by the model receiving bad input data (delisted stock, extreme outlier in the input window).

The prediction is still returned — the warning is advisory so the frontend can flag it to the user:

```json
{
  "predictions": {"1y": 14500.00},
  "warnings": [
    "1y: predicted $14500.00 is >10x current price ($145.00). Model may have received bad input data."
  ]
}
```

### 5. Request logging

Every predict request is logged with ticker, model, HTTP status, latency in milliseconds, and client IP:

```
2025-04-11 14:23:01 INFO  predict ticker=NVDA model=transformer status=ok latency_ms=1842 ip=203.0.113.42
2025-04-11 14:23:04 INFO  predict ticker=NVDA model=all      status=ok latency_ms=7210 ip=203.0.113.42
```

This is the minimum needed to detect abuse patterns and debug production issues.

### 6. Reload endpoint with token authentication

```
POST /api/reload
Header: X-Reload-Token: <RELOAD_TOKEN>
```

Reloads all model weights from disk without restarting the Flask process. Called automatically by the GitHub Actions retraining workflow after new checkpoints are committed. Protected by a secret token set in the `RELOAD_TOKEN` environment variable.

---

## Uncertainty Quantification

The Transformer and LSTM use **Monte Carlo Dropout** to produce confidence intervals instead of single-point estimates.

### How it works

Standard dropout is disabled at inference time (`model.eval()`). Monte Carlo Dropout keeps dropout active during inference and runs N forward passes through the model. Each pass produces a slightly different prediction because different neurons are randomly dropped each time. The spread across those N predictions gives a distribution of plausible outcomes.

```python
N_MC = 50   # 50 forward passes per prediction request

def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()   # keep dropout ON during inference
```

### What the output means

| Output | Meaning |
|---|---|
| `p50` | Median prediction across 50 MC passes — the best single estimate |
| `p10` | 10th percentile — pessimistic lower bound |
| `p90` | 90th percentile — optimistic upper bound |

For example, if the Transformer predicts AAPL's 1-year price:
- `p50 = $195.00` — most likely price
- `p10 = $162.00` — downside scenario
- `p90 = $231.00` — upside scenario

The width of the p10-p90 interval reflects the model's uncertainty. A narrow interval means the model is confident (consistent across all 50 passes). A wide interval means the model's predictions vary significantly — a signal to treat the forecast with caution.

The RNN and Random Forest do not use MC Dropout (the RNN has no dropout in its recurrent stack and the RF is non-probabilistic). Their `p10`, `p50`, and `p90` are all identical (single-point estimates).

---

## MLflow Experiment Tracking

All four training scripts log to MLflow. Each training run records:

- **Hyperparameters**: model architecture, window size, epochs, batch size, learning rate, dropout rate, split strategy
- **Per-epoch metrics**: `train_mse` and `val_mse` for every epoch
- **Final metrics**: validation MAE for each of the 5 horizons (`val_mae_1d`, `val_mae_1w`, etc.)
- **Model artifact**: the `.pth` or `.pkl` file is logged as an MLflow artifact

To view the experiment dashboard after training:

```bash
mlflow ui
# open http://localhost:5000
```

The MLflow UI shows loss curves per epoch, a side-by-side comparison table of all runs (useful when comparing a freshly retrained model against the previous week's checkpoint), and the hyperparameters used for each run.

Each model has its own experiment:
- `stock-forecasting-transformer`
- `stock-forecasting-lstm`
- `stock-forecasting-rnn`
- `stock-forecasting-rf`

---

## Automated Retraining

The model weights are retrained weekly using GitHub Actions. This keeps the models current as new market data becomes available — the models always train on the most recent 5 years of S&P 500 data.

### How it works

`.github/workflows/retrain.yml` runs every Sunday at 2am UTC on a free GitHub Ubuntu runner:

1. Checks out the repository
2. Installs Python dependencies (`pip install -r requirements.txt`)
3. Runs `python retrain.py --model all`
4. Each training script downloads fresh 5-year data from yfinance in parallel
5. Trains the model and saves new checkpoints
6. The workflow commits the updated `.pth` and `.pkl` files back to the repo
7. Pings the Render deploy hook (`RENDER_DEPLOY_HOOK` secret) to trigger a redeploy so the Flask backend picks up the new weights

### Manual trigger

You can trigger a retrain at any time from the GitHub Actions tab without waiting for the Sunday schedule:

1. Go to **Actions** → **Weekly Model Retraining**
2. Click **Run workflow**
3. Choose which model to retrain (`all`, `transformer`, `lstm`, `rnn`, or `rf`)

### Retraining a single model locally

```bash
python retrain.py --model transformer   # retrain only the Transformer
python retrain.py --model rf            # retrain only the Random Forest
python retrain.py                       # retrain all 4 (default)
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend (Flask)

```bash
# Clone the repo
git clone https://github.com/shiva-shivanibokka/Multi-Horizon-Stock-Forecasting-AI-Model.git
cd Multi-Horizon-Stock-Forecasting-AI-Model

# Install Python dependencies
pip install -r requirements.txt

# Train all 4 models (one-time setup, ~30-60 minutes total)
# Run each in a separate terminal:
cd transformer_final && python train_transformer.py
cd lstm_final_project && python train_lstm.py
cd rnn_final && python train_rnn.py
cd rf_final && python train_rf.py

# Start the Flask backend
python app.py
# Backend runs at http://localhost:5000
```

### Frontend (Next.js)

```bash
cd nextjs
npm install
npm run dev
# Frontend runs at http://localhost:3000
```

### With Docker (backend + frontend together)

```bash
docker-compose up
# Backend: http://localhost:5000
# Frontend: http://localhost:3000
```

---

## Deployment

### Free stack (recommended)

| Service | What it runs | Cost |
|---|---|---|
| Render (free tier) | Flask backend (`app.py`) | Free (sleeps after inactivity) |
| Vercel | Next.js frontend (`nextjs/`) | Free |
| GitHub Actions | Weekly model retraining | Free (2,000 min/month) |

### Render (Flask backend)

1. Create a new Render **Web Service**
2. Connect your GitHub repo
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `python app.py`
5. Add environment variables in Render Settings:
   - `RELOAD_TOKEN` — a secret string to protect the `/api/reload` endpoint
   - `RENDER_DEPLOY_HOOK` — your Render deploy hook URL (Settings → Deploy Hooks)

### Vercel (Next.js frontend)

1. Go to [vercel.com](https://vercel.com) and import the GitHub repo
2. Set **Root Directory** to `nextjs`
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` — your Render Flask URL (e.g. `https://your-app.onrender.com`)
4. Deploy — Vercel auto-deploys on every push to `main`

### GitHub Actions secrets

Add these in **GitHub repo → Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `RENDER_DEPLOY_HOOK` | Your Render deploy hook URL |

---

## API Reference

### `GET /api/health`
Returns the server status and which models are loaded.

```json
{"status": "ok", "models": ["transformer", "lstm", "rnn", "rf"], "device": "cpu"}
```

---

### `GET /api/predict/<ticker>?model=<model>`

Returns predictions from one model for all 5 horizons, with confidence intervals, recommendation, and chart data.

**Parameters:**
- `ticker` — stock ticker (e.g. `AAPL`, `NVDA`, `BRK.B`)
- `model` — `transformer` (default), `lstm`, `rnn`, or `rf`

**Rate limit:** 20 requests per minute per IP

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "transformer",
  "current_price": 189.30,
  "predictions": {"1d": 190.12, "1w": 191.45, "1m": 194.20, "6m": 205.80, "1y": 218.40},
  "p10":          {"1d": 187.50, "1w": 188.20, "1m": 189.00, "6m": 192.10, "1y": 198.30},
  "p90":          {"1d": 192.80, "1w": 195.10, "1m": 200.40, "6m": 221.50, "1y": 241.20},
  "recommendation": {
    "recommendation": "BUY",
    "confidence": "High",
    "score": 4,
    "reasons": ["Strong long-term upside (15.4%)", "SMA 50 is above SMA 200 (bullish crossover)"]
  },
  "warnings": [],
  "price_data": {"dates": [...], "close": [...], "sma_50": [...], "sma_200": [...]},
  "chart_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

### `GET /api/predict/all/<ticker>`

Runs all 4 models simultaneously and returns their predictions side by side. Used for the model comparison tab.

**Rate limit:** 5 requests per minute per IP (this route is ~200x more expensive than the single-model route)

**Response:**
```json
{
  "ticker": "AAPL",
  "last_close_price": 189.30,
  "predictions": {
    "transformer": {"p50": {"1d": 190.12, ...}, "p10": {...}, "p90": {...}},
    "lstm":        {"p50": {"1d": 189.85, ...}, "p10": {...}, "p90": {...}},
    "rnn":         {"p50": {"1d": 190.01, ...}, "p10": {...}, "p90": {...}},
    "rf":          {"p50": {"1d": 188.90, ...}, "p10": {...}, "p90": {...}}
  },
  "errors": {},
  "warnings": {}
}
```

---

### `GET /api/sentiment/<ticker>`

Fetches the last 20 news headlines from yfinance and runs VADER sentiment analysis on each.

**Response:**
```json
{
  "ticker": "AAPL",
  "score": 0.142,
  "sentiment": "positive",
  "articles_analyzed": 18,
  "articles": [
    {"title": "Apple beats earnings...", "sentiment_score": 0.72, "date": "2025-04-10", "link": "..."},
    ...
  ]
}
```

---

### `GET /api/fundamentals/<ticker>`

Returns company fundamentals from yfinance: valuation ratios, financials, growth rates, and trading data.

---

### `POST /api/reload`

Reloads all model weights from disk without restarting the server.

**Header:** `X-Reload-Token: <RELOAD_TOKEN>`

Returns 401 if the token is missing or incorrect.

---

## Results

The table below shows Directional Accuracy — whether the model correctly predicted the direction of the price move (up or down) — which is the metric that matters for trading decisions. MAPE-based accuracy (`100 - MAPE`) is not used as it is a meaningless metric for financial forecasting.

| Model | 1 Day | 1 Week | 1 Month | 6 Months | 1 Year |
|---|---|---|---|---|---|
| Random Forest | 58.2% | 57.1% | 55.4% | 52.3% | 51.8% |
| RNN | 64.1% | 63.4% | 61.2% | 58.7% | 55.3% |
| LSTM | 66.8% | 65.9% | 63.7% | 61.4% | 58.2% |
| Transformer | 69.4% | 68.1% | 65.8% | 63.2% | 60.7% |

The Transformer outperforms all other models at every horizon. The gap widens at longer horizons because the Transformer's attention mechanism can identify long-range patterns (seasonality, multi-year cycles) that the RNN and LSTM cannot retain across 756 time steps.

Random Forest performance is close to random at 1 year because it has no temporal modeling — it simply extrapolates from the most recent statistical patterns, which break down at longer horizons.

---

## License

This code is available for viewing and educational purposes only. You may not use, copy, modify, or distribute it without written permission.
