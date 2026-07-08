# Multi-Horizon Probabilistic Equity Forecasting — v2 Design

**Date:** 2026-07-07
**Status:** Approved (design), pending implementation plan
**Supersedes:** v1 (four-model college project — removed for disqualifying data leakage)

---

## 1. Purpose & honest claim

A leakage-free, **probabilistic** multi-horizon equity-return forecaster built by **fine-tuning a
time-series foundation model**, honestly benchmarked against strong baselines, and deployed as a
production service on free-tier infrastructure.

**The claim is rigor and method, not alpha.** The project does not assert it beats the market. Its
value — and the thing that makes it defensible in an interview — is that it is evaluated correctly
(no leakage), quantifies its own uncertainty, compares honestly against dumb baselines, and ships
as a real, monitored, versioned system.

### Why this framing
v1 reported 56–66% directional accuracy from a pipeline with selection leakage, no embargo, a
global scaler, and a production model trained on 100% of the data. Those numbers were an optimistic
ceiling, not an out-of-sample estimate. Any credible reviewer would dismantle them. v2's
differentiator is that it makes the opposite mistake impossible and proves it with tests.

---

## 2. What the model predicts

One model per configuration outputs a **predictive distribution of forward returns** at each horizon
**{1w = 5d, 1m = 21d, 6m = 126d}**, expressed as quantiles **p10 / p50 / p90**. From that single
output we derive three honest views:

- **Return band** — p10/p50/p90 return (and implied price band).
- **Directional probability** — P(return > 0), from the fitted distribution.
- **Volatility** — from the distribution spread; benchmarked against GARCH.

No separate models for the three views. No point-estimate BUY/SELL score.

---

## 3. Constraints

- **Compute:** training + evaluation run **locally on the user's GPU laptop**. Cloud only serves.
- **Free tier only:** Hugging Face (Hub + Spaces) and Vercel. **No** Supabase, Render, or Fly.io.
- **Serving must be CPU-friendly** (HF free Space = CPU). This rules out running two large foundation
  models at inference — hence one heavy + one cheap model.
- **Data:** S&P 500, ~15–20 years daily, via yfinance (free, rate-limited → needs retry/caching).

---

## 4. System architecture

```
Local GPU laptop                          Cloud (free tier)
─────────────────                         ─────────────────
data pipeline  ─┐
train + eval   ─┼─► weights + model card ──► HF Hub  (registry, versioned)
                │                                    │ loaded at startup
                └─► W&B (experiment tracking)        ▼
                                            FastAPI service ── HF Space (Docker, CPU)
                                                    ▲ REST/JSON
                                            Next.js dashboard ── Vercel
GitHub Actions:  ruff lint + pytest (incl. leakage guards) + scheduled drift check
                 (NOT model training)
```

Training/eval is entirely local. The cloud serves only. **HF Hub is the model registry** (versioned
weights + model card) — free, native, and a real MLOps artifact.

---

## 5. Components

### 5.1 Data — `src/data/`
- `ingest.py` — fetch S&P 500 constituent list + 15–20y daily OHLCV per ticker + market series
  (VIX `^VIX`, S&P 500 `^GSPC`, sector series). Retry/backoff, cache raw to **Parquet** under
  `data/raw/`. Point-in-time: store each ticker's series as downloaded; document survivorship-bias
  caveat (constituent list is current-membership).
- `features.py` — **causal** feature engineering. **The v1 `.bfill()` future-leak is removed**
  (forward-fill only; leading NaNs dropped). Feature set: returns/technicals/market-context (the v1
  36-feature set, audited for causality). Every feature at date *t* uses only data ≤ *t*.
- `windows.py` — build input windows + **forward-return targets** at 5/21/126d. **Each sample is
  tagged with its point-in-time window-end date** so all splits are date-based, never index-based.
- `quality.py` — data-quality guards (from v1 `data_guards.py`, cleaned) — and **actually called**
  in the pipeline (v1 imported the split-check guard but never invoked it).

### 5.2 Models — `src/models/`
- `baselines.py` — random-walk (0 forward return), seasonal-naive, historical-quantile, and
  **GARCH(1,1)** for the volatility comparison. Baselines are non-negotiable; the star model must
  be shown to beat them or the honest finding is reported.
- `chronos_ft.py` — fine-tune **Chronos-Bolt** on the cross-section of ticker return series →
  native quantile output. The transfer-learning centerpiece. Trains on local GPU.
- `gbm.py` — **LightGBM** quantile regressors (per horizon, per quantile via `quantile` objective /
  `alpha`) on the engineered covariates. The cheap model that carries the multivariate signal
  (VIX, sector strength, technicals) that a univariate foundation model can't see.
- `ensemble.py` — stack/blend Chronos + GBM into final p10/p50/p90 per horizon. Blend weights are
  fit on the validation split only (never on test).

### 5.3 Evaluation — `src/eval/` — **the centerpiece**
- **Purged + embargoed walk-forward CV.** Embargo = **126 trading days** (= max horizon) so no
  6-month label ever resolves inside a later fold's test window. Purge removes train samples whose
  label window overlaps the test window.
- **Three-way split discipline:** train (fit) / validation (checkpoint selection, early stopping,
  ensemble weights) / test (touched **once**, for the reported number). v1's fatal bug was using the
  test set for selection *and* reporting — structurally impossible here.
- **Metrics:** pinball (quantile) loss; **coverage/calibration** (does the p10–p90 band cover ~80%
  empirically?); directional hit-rate vs baseline; **information coefficient** (rank corr of p50 vs
  realized return); and a **cost-aware backtest** (simple long/short on the directional signal, with
  transaction costs, reporting Sharpe — heavily caveated, not presented as a trading strategy).
- **Leakage-guard unit tests** (in `tests/`): assert no train/test date overlap; assert embargo gap
  ≥ max horizon; assert scalers/normalizers are fit on train indices only; assert features are
  causal (a feature at *t* is unchanged when future rows are perturbed). **These tests are a primary
  deliverable** — they demonstrate the candidate understands what killed v1.

### 5.4 Serving — `src/serve/`
- **FastAPI** (replaces Flask). Async, pydantic request validation, **auto OpenAPI docs at `/docs`**.
- Endpoints: `GET /health`; `GET /predict/{ticker}` → per-horizon quantiles + P(up) + vol + baseline
  comparison; `GET /models` → loaded version info.
- Loads **versioned weights from HF Hub** at startup. In-memory cache for yfinance calls. Rate
  limiting. **Structured logging** (JSON, levels). **Prediction logging** to feed drift monitoring.
- No base64 charts (frontend renders). Input validation at the boundary (ticker format, allowlist).

### 5.5 Frontend — `frontend/` (Next.js on Vercel, rebuilt)
- **Fan chart** of the return distribution per horizon; **P(up)** and **vol** tiles;
  **model-vs-baseline** and **calibration** panels.
- Prominent banner: *"Research project — methodology transparent, not investment advice."*
  Honesty is a UI feature, not fine print.
- Fetches from the HF Space API via `NEXT_PUBLIC_API_URL`.

### 5.6 MLOps
- **Experiment tracking:** Weights & Biases (free). Public run pages linkable from the README.
- **Model registry:** HF Hub model repo — versioned weights + model card (metrics, training data,
  intended use, limitations).
- **CI:** GitHub Actions on every PR — `ruff` lint + `pytest` (including the leakage guards). Pinned
  dependencies + lockfile for reproducible builds.
- **Monitoring:** prediction logging + a **scheduled data-drift check** (feature distribution vs a
  saved training reference) that **opens a GitHub issue** on drift. No dashboards to host.
- **Retraining (honest):** documented manual/local retrain → push new version to HF Hub → Space
  reloads on new version. **The v1 fantasy — GPU training on free Actions + Render reload hook — is
  deleted.** A scheduled Action only checks data freshness/drift, never trains.
- **Containerization:** Dockerfile for the HF Space — slim base, non-root user, pinned deps.

### 5.7 Config & docs
- `pydantic-settings` config with **fail-fast validation at startup** (missing required env → crash
  with a clear message, not a silent later failure). `.env.example` documents every variable.
- Rewritten, honest **README** (no "results will be added later", no admitted-broken telemetry).
- **`docs/adr/`** — architecture decision records: why Chronos-Bolt, why HF + Vercel, why the
  ensemble, why purged/embargoed CV, **why Moirai was considered and rejected** (CPU-serving cost).
- **Model card** on the HF Hub repo.

---

## 6. Explicitly NOT building (YAGNI + free tier)

Each has a documented upgrade path in an ADR:
- Orchestration (Airflow/Dagster) → scripts + `Makefile` + GitHub Actions.
- Feature store (Feast) → Parquet + point-in-time discipline.
- Redis cache → in-memory (note the swap point).
- Kubernetes → single HF Docker Space.
- A second foundation model (Moirai) → LightGBM carries covariates instead.
- Any live-trading / brokerage integration → this is a forecasting research system.

---

## 7. Tech stack

Python 3.11 · PyTorch + Chronos-Bolt · LightGBM · `arch` (GARCH) · pandas/numpy/pyarrow ·
scikit-learn · FastAPI + uvicorn · pydantic-settings · Weights & Biases · Hugging Face Hub + Spaces ·
Next.js + Recharts on Vercel · Docker · GitHub Actions · ruff · pytest.

---

## 8. Repository shape

```
src/
  data/    ingest.py  features.py  windows.py  quality.py
  models/  baselines.py  chronos_ft.py  gbm.py  ensemble.py
  eval/    cv.py  metrics.py  backtest.py
  serve/   app.py  schemas.py  config.py  registry.py
tests/     test_leakage.py  test_features.py  test_serve.py  ...
frontend/  (Next.js — rebuilt)
docs/adr/  0001-*.md ...
notebooks/ 01-eda-and-results.ipynb   (one clean notebook, not per-model sprawl)
Dockerfile  ·  pyproject.toml  ·  .github/workflows/{ci.yml, drift-check.yml}  ·  .env.example
```

---

## 9. Success criteria

1. `pytest` green, including leakage guards; CI runs them on every PR.
2. Purged/embargoed walk-forward CV produces **out-of-sample** metrics with **calibration** reported;
   the star model is compared against every baseline, honestly (win or lose).
3. A cloner can reproduce: `pip install` → run pipeline → train → eval, from the README.
4. Live: Vercel dashboard → HF Space API → real prediction with uncertainty for any S&P 500 ticker.
5. Weights versioned on HF Hub with a model card; experiments visible on a public W&B project.
6. README + ADRs make every design decision and every limitation explicit.

---

## 10. Risks & honesty notes

- **Low signal ceiling:** equity returns from OHLCV are near-unpredictable at short horizons. The
  project succeeds by *measuring this honestly*, not by manufacturing accuracy. If the foundation
  model doesn't beat baselines, that is a legitimate, defensible result and is reported as such.
- **Survivorship bias:** the S&P 500 constituent list is current membership. Documented, not hidden.
- **yfinance reliability:** free and rate-limited; ingestion needs retry/backoff and caching.
- **Free-tier cold starts:** HF Space may cold-start; acceptable for a portfolio demo, documented.
