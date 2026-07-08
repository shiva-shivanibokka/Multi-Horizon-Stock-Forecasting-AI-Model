# Local Training Pipeline Implementation Plan (Plan 02)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete leakage-free local training pipeline — dataset assembly, baselines, LightGBM quantile models, a fine-tuned Chronos-Bolt model, their stacked ensemble, and the purged/embargoed walk-forward evaluation that scores them — culminating in one runnable entrypoint that produces honest out-of-sample metrics plus saved artifacts for later cloud serving.

**Architecture:** The data layer (Plan 01) already emits per-ticker `WindowSet(X[n,252,35], y_ret[n,3], end_dates, base_close)` via `build_ticker`. This plan (a) assembles those per-ticker windows into one long *panel* table (one row per ticker/date, the window's last-timestep 35 features + realized forward returns) that drives the tabular models and the evaluation, and (b) reads the cached raw price series for the univariate Chronos-Bolt model. A purged + embargoed walk-forward CV harness is the outer loop: cheap models (baselines, LightGBM) refit per fold; the GPU-heavy Chronos-Bolt is fine-tuned once on the earliest train span and forecast forward across all test anchors (honest, conservative, feasible overnight). All models emit p10/p50/p90 forward-return quantiles at the three horizons; the ensemble is a convex blend whose weights are fit on validation only.

**Tech Stack:** Python 3.11+ · pandas/numpy/pyarrow · LightGBM (quantile objective) · `arch` (GARCH) · scipy/scikit-learn (metrics) · AutoGluon-TimeSeries (Chronos-Bolt fine-tuning) · Weights & Biases (optional tracking) · pytest · ruff.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the approved design spec (`docs/superpowers/specs/2026-07-07-multi-horizon-forecasting-overhaul-design.md`) and Plan 01.

- **Causality is absolute.** No feature, target, split, scaler, or forecast at date *t* may use any data from after *t*. Forward-fill only, never `.bfill()`. This is the project's entire reason for existing.
- **Splits are date-based, never index-based.** Every train/test boundary is a calendar date. Every sample carries its point-in-time window-end date (`end_dates` from Plan 01).
- **Embargo = 126 trading days = `settings.max_horizon`.** No label window may straddle a train/test boundary: purge train samples whose forward-return window overlaps the test span; embargo test samples within 126 trading days of the train boundary.
- **Horizons:** `{"1w": 5, "1m": 21, "6m": 126}` — exactly `settings.horizons`. Column order for `y_ret` is `[1w, 1m, 6m]` (insertion order of the dict), matching `build_windows`.
- **Quantiles:** `QUANTILES = [0.1, 0.5, 0.9]` (p10/p50/p90). p10 ≤ p50 ≤ p90 must always hold in emitted output (sort if a model violates it).
- **Test set is touched once.** Validation split (a slice of train, before the embargo) selects hyperparameters, early stopping, and ensemble weights. The test span is scored once for the reported number.
- **Feature contract:** `mhf.data.features.FEATURES` is the 35-name ordered list. The panel's feature columns are exactly `FEATURES`, in order. Never hardcode the list elsewhere — import it.
- **Free tier / local only.** Training and evaluation run on the user's local GPU laptop. No new cloud dependency. Heavy deps (autogluon, torch) live in an optional `train` extra, not the base install.
- **No attribution noise.** No "Claude", "Anthropic", "Co-Authored-By", or AI-assistant references anywhere in code, comments, commit messages, or docs.

## Interfaces from Plan 01 (do not reimplement — import)

- `from mhf.config import settings` → `settings.horizons` (dict), `settings.max_horizon` (126), `settings.window_short` (252), `settings.raw_dir` (Path `data/raw`), `settings.data_dir` (Path `data`).
- `from mhf.data.features import FEATURES, compute_features` — `FEATURES` is 35 names; `compute_features(df, market)` returns a causal feature frame.
- `from mhf.data.ingest import fetch_sp500, download_ohlcv, cache_path` — `fetch_sp500() -> (tickers, sectors)`; `download_ohlcv(ticker, refresh=False) -> DataFrame|None` (OHLCV, cached parquet under `raw_dir`); `cache_path(ticker) -> Path`.
- `from mhf.data.build import build_ticker` — `build_ticker(ticker, market) -> WindowSet|None`.
- `from mhf.data.windows import WindowSet` — dataclass `X, y_ret, end_dates, base_close`; `X` is `(n, 252, 35)` float32, `y_ret` is `(n, 3)` float32, `end_dates` is `(n,)` datetime64, `base_close` is `(n,)` float32.

---

## Execution note (read before starting Task 1)

After all tasks are implemented and reviewed, and BEFORE any real training run, the controller runs the **repo debugger skill** end-to-end on this pipeline (static + a `--smoke` dry run on 2–3 tickers with `fine_tune_steps=1`), fixes every bug it surfaces, and only then hands the user the full training command. Do not kick off the real fine-tune until the smoke path is green. This is an explicit user requirement.

---

## File Structure

- Create `src/mhf/models/__init__.py`, `src/mhf/eval/__init__.py` — package markers.
- Create `src/mhf/data/assemble.py` — panel assembly from per-ticker windows.
- Extend `src/mhf/data/ingest.py` — add `fetch_market()`.
- Create `src/mhf/eval/cv.py` — purged/embargoed walk-forward folds (centerpiece).
- Create `src/mhf/eval/metrics.py` — pinball, coverage, hit-rate, information coefficient.
- Create `src/mhf/models/baselines.py` — historical-quantile + GARCH volatility.
- Create `src/mhf/models/gbm.py` — LightGBM quantile regressors.
- Create `src/mhf/models/chronos_ft.py` — Chronos-Bolt fine-tune wrapper + pure transforms.
- Create `src/mhf/models/ensemble.py` — convex quantile blend.
- Create `src/mhf/train.py` — orchestration entrypoint (CLI, `--smoke`).
- Tests mirror under `tests/` (`tests/data/`, `tests/eval/`, `tests/models/`).

Each model exposes the same tiny contract so the entrypoint and ensemble treat them uniformly:
```
fit(panel_train: pd.DataFrame) -> self
predict_quantiles(panel_rows: pd.DataFrame) -> np.ndarray  # shape (len(rows), n_horizons, n_quantiles)
```
(Chronos also needs raw series and anchor dates — see Task 8. Its `predict_quantiles` accepts the same panel rows and looks up the raw series internally.)

---

## Task 1: Training dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mhf/models/__init__.py` (empty)
- Create: `src/mhf/eval/__init__.py` (empty)
- Create: `src/mhf/constants.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces: `from mhf.constants import QUANTILES` → `[0.1, 0.5, 0.9]`. A single source of truth for quantile levels, imported by every model and metric.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants.py
from mhf.constants import QUANTILES


def test_quantiles_are_p10_p50_p90():
    assert QUANTILES == [0.1, 0.5, 0.9]
    assert QUANTILES == sorted(QUANTILES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_constants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.constants'`

- [ ] **Step 3: Create the constants module and package markers**

```python
# src/mhf/constants.py
"""Cross-cutting constants shared by models, eval, and serving."""

QUANTILES: list[float] = [0.1, 0.5, 0.9]
```

Create empty `src/mhf/models/__init__.py` and `src/mhf/eval/__init__.py`.

- [ ] **Step 4: Add the `train` optional-dependency group to `pyproject.toml`**

Insert into the existing `[project.optional-dependencies]` table (keep the existing `dev` line):

```toml
[project.optional-dependencies]
dev = ["pytest>=8.2", "ruff>=0.5"]
train = [
    "lightgbm>=4.3",
    "arch>=7.0",
    "scikit-learn>=1.5",
    "scipy>=1.13",
    "autogluon.timeseries>=1.2",
    "wandb>=0.17",
]
```

Note in a comment above the `train` group: `# heavy local-GPU training deps; installed with: pip install -e ".[train]"`. Do NOT add these to the base `dependencies` list — base install must stay light for CPU serving.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_constants.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/mhf/constants.py src/mhf/models/__init__.py src/mhf/eval/__init__.py tests/test_constants.py
git commit -m "feat: add training deps extra, shared QUANTILES, models/eval packages"
```

---

## Task 2: `fetch_market()` — causal market-context series

**Files:**
- Modify: `src/mhf/data/ingest.py`
- Test: `tests/data/test_market.py`

**Interfaces:**
- Consumes: `download`-style yfinance access (same `yf.download` already used by `_download_one`).
- Produces: `fetch_market(refresh=False) -> pd.DataFrame` indexed by date with columns exactly `["vix_close", "sp500_ret_21d", "sp500_ret_63d"]` — the frame `compute_features` expects for its `market` argument. Cached to `settings.data_dir / "market.parquet"`.

**Context:** `compute_features` does `market.reindex(d.index).ffill()` and reads those three columns. `sp500_ret_21d/63d` are trailing percentage changes of `^GSPC` close (`pct_change(21)` / `pct_change(63)`), `vix_close` is `^VIX` close. All trailing → causal. Leading NaNs (first 63 rows) are fine; the reindex+ffill+downstream dropna handles them.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_market.py
import numpy as np
import pandas as pd

import mhf.data.ingest as ingest


def _fake_download(symbol, **kwargs):
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    close = 100 + np.cumsum(rng.normal(0, 1, size=300))
    return pd.DataFrame({"Close": np.clip(close, 5, None)}, index=idx)


def test_fetch_market_columns_and_causality(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.settings, "data_dir", tmp_path)
    monkeypatch.setattr(ingest.yf, "download", _fake_download)

    m = ingest.fetch_market(refresh=True)
    assert list(m.columns) == ["vix_close", "sp500_ret_21d", "sp500_ret_63d"]
    # Trailing returns: row t must equal close[t]/close[t-21]-1 -> depends only on the past.
    assert m["sp500_ret_21d"].isna().sum() == 21  # exactly the warmup, no bfill
    assert (ingest.settings.data_dir / "market.parquet").exists()


def test_fetch_market_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.settings, "data_dir", tmp_path)
    calls = {"n": 0}

    def counting_download(symbol, **kwargs):
        calls["n"] += 1
        return _fake_download(symbol, **kwargs)

    monkeypatch.setattr(ingest.yf, "download", counting_download)
    ingest.fetch_market(refresh=True)
    n_after_first = calls["n"]
    ingest.fetch_market()  # cache hit -> no new downloads
    assert calls["n"] == n_after_first
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/data/test_market.py -v`
Expected: FAIL — `AttributeError: module 'mhf.data.ingest' has no attribute 'fetch_market'`

- [ ] **Step 3: Implement `fetch_market`**

Add to `src/mhf/data/ingest.py` (reuse the existing `yf`, `settings`, `_download_one` machinery; add a small close-only fetch):

```python
def _download_close(symbol: str) -> pd.Series:
    df = yf.download(symbol, period=settings.history_period, interval="1d", progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna().sort_index()


def market_cache_path() -> Path:
    return settings.data_dir / "market.parquet"


def fetch_market(refresh: bool = False) -> pd.DataFrame:
    path = market_cache_path()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    vix = _download_close("^VIX")
    gspc = _download_close("^GSPC")
    idx = gspc.index
    out = pd.DataFrame(
        {
            "vix_close": vix.reindex(idx).ffill(),
            "sp500_ret_21d": gspc.pct_change(21, fill_method=None),
            "sp500_ret_63d": gspc.pct_change(63, fill_method=None),
        },
        index=idx,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/data/test_market.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/mhf/data/ingest.py tests/data/test_market.py
git commit -m "feat: fetch_market() causal VIX + S&P context series with parquet cache"
```

---

## Task 3: Panel assembly

**Files:**
- Create: `src/mhf/data/assemble.py`
- Test: `tests/data/test_assemble.py`

**Interfaces:**
- Consumes: `build_ticker(ticker, market) -> WindowSet|None` (Task interfaces), `FEATURES`, `settings`.
- Produces:
  - `build_panel(tickers: list[str], market: pd.DataFrame, downloader=build_ticker) -> pd.DataFrame` — one row per (ticker, window-end date). Columns: `ticker` (str), `end_date` (datetime64), the 35 `FEATURES` names (float32, the window's last timestep = `X[:, -1, :]`), `y_1w`,`y_1m`,`y_6m` (float32, from `y_ret`), `base_close` (float32). Rows with any NaN target dropped. Tickers where `build_ticker` returns None are skipped.
  - `write_panel(panel: pd.DataFrame, path: Path|None=None) -> Path` — writes parquet (default `settings.data_dir / "processed" / "panel.parquet"`), returns the path.
  - `Y_COLS = ["y_1w", "y_1m", "y_6m"]` module constant (order matches `settings.horizons` / `y_ret` columns).

**Context:** The panel's last-timestep feature vector is what LightGBM/baselines consume, and `end_date` is what the CV harness splits on. `X[:, -1, :]` is the feature snapshot AT the window-end date — strictly causal (it is row `end_idx` of the causal feature frame). The `downloader` parameter is an injection seam for testing (defaults to the real `build_ticker`).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_assemble.py
import numpy as np
import pandas as pd

from mhf.data.assemble import Y_COLS, build_panel, write_panel
from mhf.data.features import FEATURES
from mhf.data.windows import WindowSet


def _fake_windowset(n=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 252, len(FEATURES))).astype(np.float32)
    y = rng.normal(0, 0.05, size=(n, 3)).astype(np.float32)
    dates = pd.bdate_range("2021-01-01", periods=n).to_numpy()
    base = rng.uniform(50, 150, size=n).astype(np.float32)
    return WindowSet(X=X, y_ret=y, end_dates=dates, base_close=base)


def test_build_panel_shape_and_columns():
    def fake_downloader(ticker, market):
        return None if ticker == "BAD" else _fake_windowset(seed=abs(hash(ticker)) % 1000)

    panel = build_panel(["AAA", "BAD", "BBB"], market=pd.DataFrame(), downloader=fake_downloader)

    assert list(panel.columns) == ["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"]
    assert set(panel["ticker"]) == {"AAA", "BBB"}  # BAD skipped
    assert len(panel) == 20  # 10 rows each
    # Last-timestep feature copied verbatim.
    assert not panel[FEATURES].isna().any().any()


def test_build_panel_last_timestep_is_used():
    ws = _fake_windowset(n=3, seed=42)

    def one(ticker, market):
        return ws

    panel = build_panel(["AAA"], market=pd.DataFrame(), downloader=one)
    np.testing.assert_allclose(
        panel[FEATURES].to_numpy(), ws.X[:, -1, :], rtol=1e-6
    )
    np.testing.assert_allclose(panel[Y_COLS].to_numpy(), ws.y_ret, rtol=1e-6)


def test_write_panel_roundtrip(tmp_path):
    panel = build_panel(["AAA"], market=pd.DataFrame(),
                        downloader=lambda t, m: _fake_windowset(n=4))
    path = write_panel(panel, tmp_path / "p.parquet")
    assert path.exists()
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, panel)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/data/test_assemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.data.assemble'`

- [ ] **Step 3: Implement `assemble.py`**

```python
# src/mhf/data/assemble.py
import logging
from pathlib import Path

import pandas as pd

from mhf.config import settings
from mhf.data.build import build_ticker
from mhf.data.features import FEATURES

logger = logging.getLogger(__name__)

Y_COLS = ["y_1w", "y_1m", "y_6m"]


def build_panel(tickers, market: pd.DataFrame, downloader=build_ticker) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        ws = downloader(ticker, market)
        if ws is None:
            logger.info("skip %s: no windows", ticker)
            continue
        feats = pd.DataFrame(ws.X[:, -1, :], columns=FEATURES)
        feats.insert(0, "end_date", pd.to_datetime(ws.end_dates))
        feats.insert(0, "ticker", ticker)
        for i, col in enumerate(Y_COLS):
            feats[col] = ws.y_ret[:, i]
        feats["base_close"] = ws.base_close
        frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=Y_COLS).reset_index(drop=True)
    return panel


def write_panel(panel: pd.DataFrame, path: Path | None = None) -> Path:
    if path is None:
        path = settings.data_dir / "processed" / "panel.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/data/test_assemble.py -v`
Expected: PASS (3 tests). If `assert_frame_equal` trips on dtype after parquet roundtrip, cast feature/target/`base_close` columns to `float32` explicitly at the end of `build_panel` before returning.

- [ ] **Step 5: Commit**

```bash
git add src/mhf/data/assemble.py tests/data/test_assemble.py
git commit -m "feat: assemble per-ticker windows into a long panel table"
```

---

## Task 4: Purged + embargoed walk-forward CV (centerpiece)

**Files:**
- Create: `src/mhf/eval/cv.py`
- Test: `tests/eval/test_cv.py`

**Interfaces:**
- Produces: `walk_forward_folds(end_dates, n_folds=4, embargo=None, min_train=252) -> list[Fold]` where `Fold` is a `@dataclass` with boolean numpy masks `train` and `test` over the input rows. `embargo` defaults to `settings.max_horizon` (126). Splits are computed on the sorted unique dates; within a fold, train = rows whose `end_date` is on/before the train-cut AND whose label window (`end_date + embargo` trading rows) does not enter the test span (purge); test = rows in the fold's test window that are ≥ `embargo` business days after the train-cut (embargo).

**Context — the leakage-guard tests are a primary deliverable of the whole project.** They must be strong enough to go RED if the purge or embargo is removed. This is the file an interviewer will read first.

**Design (expanding window):** Sort unique dates. Reserve the first `min_train` dates as the seed train. Divide the remaining timeline into `n_folds` contiguous test blocks. Fold *k*: train = all rows with `end_date` ≤ (start of test block *k* minus `embargo` business days) with label windows fully before the test block; test = rows in test block *k* with `end_date` ≥ (start of block *k* plus `embargo`) — i.e. drop the first `embargo` business days of each test block so no 6-month label reaches back over the boundary.

- [ ] **Step 1: Write the failing tests (leakage guards)**

```python
# tests/eval/test_cv.py
import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.eval.cv import Fold, walk_forward_folds


def _dates(n=1500):
    # repeat each business day across 5 tickers to mimic a panel
    base = pd.bdate_range("2015-01-01", periods=n)
    return np.repeat(base.to_numpy(), 5)


def test_no_train_test_date_overlap():
    ed = _dates()
    for fold in walk_forward_folds(ed, n_folds=4):
        train_dates = set(pd.to_datetime(ed[fold.train]))
        test_dates = set(pd.to_datetime(ed[fold.test]))
        assert train_dates.isdisjoint(test_dates)


def test_embargo_gap_at_least_max_horizon():
    ed = _dates()
    embargo = settings.max_horizon
    for fold in walk_forward_folds(ed, n_folds=4, embargo=embargo):
        last_train = pd.to_datetime(ed[fold.train]).max()
        first_test = pd.to_datetime(ed[fold.test]).min()
        gap_bdays = np.busday_count(last_train.date(), first_test.date())
        assert gap_bdays >= embargo, f"gap {gap_bdays} < embargo {embargo}"


def test_purge_removes_overlapping_labels():
    # A train sample whose 126-day label window reaches into the test span must be purged.
    ed = _dates()
    embargo = settings.max_horizon
    for fold in walk_forward_folds(ed, n_folds=4, embargo=embargo):
        first_test = pd.to_datetime(ed[fold.test]).min()
        train_ed = pd.to_datetime(ed[fold.train])
        # every train label window must END strictly before the test span begins
        label_end = train_ed + pd.tseries.offsets.BDay(embargo)
        assert (label_end < first_test).all()


def test_folds_are_nonempty_and_expanding():
    ed = _dates()
    folds = walk_forward_folds(ed, n_folds=4)
    assert len(folds) == 4
    train_sizes = [int(f.train.sum()) for f in folds]
    for f in folds:
        assert f.train.sum() > 0 and f.test.sum() > 0
    assert train_sizes == sorted(train_sizes)  # expanding window


def test_masks_are_disjoint_boolean():
    ed = _dates()
    for f in walk_forward_folds(ed, n_folds=3):
        assert f.train.dtype == bool and f.test.dtype == bool
        assert not (f.train & f.test).any()
        assert len(f.train) == len(ed)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/eval/test_cv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.eval.cv'`

- [ ] **Step 3: Implement `cv.py`**

```python
# src/mhf/eval/cv.py
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mhf.config import settings


@dataclass
class Fold:
    train: np.ndarray  # boolean mask over rows
    test: np.ndarray


def walk_forward_folds(end_dates, n_folds: int = 4, embargo: int | None = None,
                       min_train: int = 252) -> list[Fold]:
    if embargo is None:
        embargo = settings.max_horizon
    ed = pd.to_datetime(pd.Series(end_dates)).reset_index(drop=True)
    uniq = np.sort(ed.unique())
    if len(uniq) <= min_train + n_folds:
        raise ValueError(f"not enough distinct dates ({len(uniq)}) for {n_folds} folds")

    bday = pd.tseries.offsets.BDay(embargo)
    test_pool = uniq[min_train:]
    blocks = np.array_split(test_pool, n_folds)

    folds: list[Fold] = []
    for block in blocks:
        block_start = pd.Timestamp(block[0])
        block_end = pd.Timestamp(block[-1])
        # embargo the head of the test block
        test_lo = block_start + bday
        test_mask = ((ed >= test_lo) & (ed <= block_end)).to_numpy()
        # train = everything whose label window ends before the (embargoed) test span
        train_cut = test_lo  # first scored test date
        label_end = ed + bday
        train_mask = (label_end < train_cut).to_numpy()
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        folds.append(Fold(train=train_mask, test=test_mask))
    return folds
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/eval/test_cv.py -v`
Expected: PASS (5 tests). If `test_folds_are_nonempty_and_expanding` sees fewer than 4 folds because an early block fully embargoes away, raise `min_train` handling: ensure each block spans more than `embargo` business days by construction (with 1500 daily dates and min_train 252, each of 4 blocks ≈ 312 days > 126 — safe). Keep the fixture at 1500 dates.

- [ ] **Step 5: Commit**

```bash
git add src/mhf/eval/cv.py tests/eval/test_cv.py
git commit -m "feat: purged + embargoed walk-forward CV with leakage-guard tests"
```

---

## Task 5: Evaluation metrics

**Files:**
- Create: `src/mhf/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces (all operate per horizon; `q_pred` is `(n, n_quantiles)` aligned to `QUANTILES`, `y` is `(n,)` realized returns):
  - `pinball_loss(y, q_pred, quantiles=QUANTILES) -> float` — mean pinball across the quantile set.
  - `coverage(y, q_low, q_high) -> float` — fraction of `y` within `[q_low, q_high]` (target ≈ 0.8 for p10–p90).
  - `directional_hit_rate(y, p50) -> float` — fraction where `sign(p50) == sign(y)` (zeros count as up).
  - `information_coefficient(y, p50) -> float` — Spearman rank correlation of `p50` vs `y`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_metrics.py
import numpy as np

from mhf.constants import QUANTILES
from mhf.eval.metrics import (
    coverage,
    directional_hit_rate,
    information_coefficient,
    pinball_loss,
)


def test_pinball_zero_for_perfect_median_at_p50():
    y = np.array([0.0, 1.0, -1.0])
    # perfect prediction at every quantile -> zero loss
    q = np.tile(y[:, None], (1, len(QUANTILES)))
    assert pinball_loss(y, q) == 0.0


def test_pinball_positive_for_wrong_prediction():
    y = np.array([1.0, 1.0])
    q = np.zeros((2, len(QUANTILES)))
    assert pinball_loss(y, q) > 0


def test_coverage_counts_band_membership():
    y = np.array([0.0, 5.0, -5.0, 0.5])
    lo = np.full(4, -1.0)
    hi = np.full(4, 1.0)
    assert coverage(y, lo, hi) == 0.5  # 0.0 and 0.5 are inside


def test_directional_hit_rate():
    y = np.array([1.0, -1.0, 2.0, -3.0])
    p50 = np.array([0.5, -0.2, -0.1, -1.0])  # 3 of 4 correct sign
    assert directional_hit_rate(y, p50) == 0.75


def test_information_coefficient_perfect_rank():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p50 = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(information_coefficient(y, p50) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.eval.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/mhf/eval/metrics.py
import numpy as np
from scipy.stats import spearmanr

from mhf.constants import QUANTILES


def pinball_loss(y, q_pred, quantiles=QUANTILES) -> float:
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    total = 0.0
    for j, q in enumerate(quantiles):
        diff = y - q_pred[:, j]
        total += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return float(total / len(quantiles))


def coverage(y, q_low, q_high) -> float:
    y = np.asarray(y, dtype=float)
    inside = (y >= np.asarray(q_low, dtype=float)) & (y <= np.asarray(q_high, dtype=float))
    return float(np.mean(inside))


def directional_hit_rate(y, p50) -> float:
    y = np.asarray(y, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    return float(np.mean((p50 >= 0) == (y >= 0)))


def information_coefficient(y, p50) -> float:
    rho, _ = spearmanr(np.asarray(y, dtype=float), np.asarray(p50, dtype=float))
    return float(rho)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mhf/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: pinball, coverage, directional hit-rate, information coefficient"
```

---

## Task 6: Baseline models (historical-quantile + GARCH volatility)

**Files:**
- Create: `src/mhf/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Consumes: `Y_COLS` (from assemble), `QUANTILES`, `settings.horizons`.
- Produces:
  - `class RandomWalk` with `fit(panel_train) -> self` / `predict_quantiles(panel_rows) -> (len(rows), n_horizons, n_quantiles)` — the canonical efficient-market null: expected forward return = 0, band a zero-mean Gaussian scaled by the train-window per-horizon return std. This is the "dumb baseline" the whole project's honesty thesis rests on beating (or honestly not beating).
  - `class HistoricalQuantile` with `fit(panel_train) -> self` (stores empirical `QUANTILES` of each `Y_COLS` column over the train panel) and `predict_quantiles(panel_rows) -> np.ndarray` shape `(len(rows), n_horizons, n_quantiles)` — broadcasts the fitted per-horizon quantiles to every row. The unconditional distributional baseline.
  - `garch_volatility(returns: pd.Series, horizon: int) -> float` — fits `arch_model(returns*100, vol="GARCH", p=1, q=1)`, forecasts `horizon` steps, returns annualized-scale forward vol (the sqrt of summed variance over the horizon, back in return units). Used for the volatility comparison; wrap the fit in try/except and fall back to the trailing realized std on failure (GARCH occasionally fails to converge on short/degenerate series).

**Scope note (deliberate):** The spec lists four baselines. `RandomWalk` (0-drift null) and `HistoricalQuantile` (unconditional distribution) are implemented; **seasonal-naive is omitted deliberately** — daily equity returns have no exploitable seasonality at 5/21/126-day horizons, so a seasonal-naive baseline would be indistinguishable from random-walk. GARCH covers the volatility null. This is a documented interpretation, not an oversight.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_baselines.py
import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.models.baselines import HistoricalQuantile, RandomWalk, garch_volatility


def _panel(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0.01 * i, 0.05, size=n) for i, c in enumerate(Y_COLS)})
    df["end_date"] = pd.bdate_range("2018-01-01", periods=n)
    return df


def test_random_walk_zero_median_and_monotonic():
    train = _panel()
    m = RandomWalk().fit(train)
    out = m.predict_quantiles(train.iloc[:5])
    assert out.shape == (5, len(Y_COLS), len(QUANTILES))
    assert (np.diff(out, axis=2) >= 0).all()
    np.testing.assert_allclose(out[:, :, 1], 0.0, atol=1e-9)  # p50 == 0 (random walk)


def test_historical_quantile_shape_and_monotonic():
    train = _panel()
    m = HistoricalQuantile().fit(train)
    out = m.predict_quantiles(train.iloc[:10])
    assert out.shape == (10, len(Y_COLS), len(QUANTILES))
    # p10 <= p50 <= p90 for every row/horizon
    assert (np.diff(out, axis=2) >= 0).all()


def test_historical_quantile_matches_empirical():
    train = _panel(seed=1)
    m = HistoricalQuantile().fit(train)
    out = m.predict_quantiles(train.iloc[:1])
    expected = np.quantile(train["y_1w"].to_numpy(), QUANTILES)
    np.testing.assert_allclose(out[0, 0, :], expected, rtol=1e-6)


def test_garch_volatility_positive():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0, 0.02, size=800))
    vol = garch_volatility(rets, horizon=21)
    assert vol > 0 and np.isfinite(vol)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/models/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.models.baselines'`

- [ ] **Step 3: Implement `baselines.py`**

```python
# src/mhf/models/baselines.py
import logging

import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS

logger = logging.getLogger(__name__)


class RandomWalk:
    """Efficient-market null: E[forward return] = 0, band = zero-mean Gaussian on train vol."""

    def __init__(self, quantiles=QUANTILES):
        self.quantiles = quantiles
        self.sigma_: np.ndarray | None = None  # (n_horizons,)

    def fit(self, panel_train: pd.DataFrame) -> "RandomWalk":
        self.sigma_ = np.array([panel_train[c].std() for c in Y_COLS])
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        from scipy.stats import norm

        assert self.sigma_ is not None, "fit first"
        z = norm.ppf(self.quantiles)  # zero-mean quantile z-scores
        band = self.sigma_[:, None] * z[None, :]  # (n_horizons, n_quantiles)
        n = len(panel_rows)
        return np.broadcast_to(band[None], (n, *band.shape)).copy()


class HistoricalQuantile:
    def __init__(self, quantiles=QUANTILES):
        self.quantiles = quantiles
        self.table_: np.ndarray | None = None  # (n_horizons, n_quantiles)

    def fit(self, panel_train: pd.DataFrame) -> "HistoricalQuantile":
        self.table_ = np.stack(
            [np.quantile(panel_train[c].to_numpy(), self.quantiles) for c in Y_COLS]
        )
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        assert self.table_ is not None, "fit first"
        n = len(panel_rows)
        return np.broadcast_to(self.table_[None], (n, *self.table_.shape)).copy()


def garch_volatility(returns: pd.Series, horizon: int) -> float:
    from arch import arch_model

    r = returns.dropna().to_numpy() * 100.0
    try:
        res = arch_model(r, vol="GARCH", p=1, q=1, mean="Zero").fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
        var_path = fc.variance.to_numpy().ravel()[:horizon]
        return float(np.sqrt(var_path.sum()) / 100.0)
    except Exception as e:  # noqa: BLE001 - GARCH convergence is genuinely flaky
        logger.warning("GARCH failed (%s); falling back to realized std", e)
        return float(returns.dropna().std() * np.sqrt(horizon))
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/models/test_baselines.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mhf/models/baselines.py tests/models/test_baselines.py
git commit -m "feat: random-walk + historical-quantile baselines, GARCH(1,1) vol with fallback"
```

---

## Task 7: LightGBM quantile models

**Files:**
- Create: `src/mhf/models/gbm.py`
- Test: `tests/models/test_gbm.py`

**Interfaces:**
- Consumes: `FEATURES`, `Y_COLS`, `QUANTILES`, `settings.horizons`.
- Produces: `class GBMQuantile` with:
  - `__init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, random_state=0)`
  - `fit(panel_train) -> self` — trains one `LGBMRegressor(objective="quantile", alpha=q)` per (horizon, quantile) on `panel_train[FEATURES]` → `panel_train[y_col]`. Stores in a `{(h_idx, q_idx): model}` dict.
  - `predict_quantiles(panel_rows) -> np.ndarray` shape `(len(rows), n_horizons, n_quantiles)`, then **sorts along the quantile axis** so p10 ≤ p50 ≤ p90 (LightGBM quantile crossings are common).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_gbm.py
import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.models.gbm import GBMQuantile


def _panel(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)
    # y depends on feature 0 so the model has real signal to learn
    signal = 0.03 * X[:, 0]
    for i, c in enumerate(Y_COLS):
        df[c] = signal + rng.normal(0, 0.01, size=n)
    df["ticker"] = "AAA"
    df["end_date"] = pd.bdate_range("2017-01-01", periods=n)
    return df


def test_gbm_shape_and_quantile_monotonic():
    train = _panel()
    m = GBMQuantile(n_estimators=50).fit(train)
    out = m.predict_quantiles(train.iloc[:20])
    assert out.shape == (20, len(Y_COLS), len(QUANTILES))
    assert (np.diff(out, axis=2) >= -1e-9).all()  # sorted p10<=p50<=p90


def test_gbm_learns_direction():
    train = _panel(seed=3)
    m = GBMQuantile(n_estimators=100).fit(train)
    hi = train[train[FEATURES[0]] > 1.0].iloc[:30]
    lo = train[train[FEATURES[0]] < -1.0].iloc[:30]
    p50_hi = m.predict_quantiles(hi)[:, 0, 1].mean()
    p50_lo = m.predict_quantiles(lo)[:, 0, 1].mean()
    assert p50_hi > p50_lo  # higher feature-0 -> higher predicted median
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/models/test_gbm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.models.gbm'`

- [ ] **Step 3: Implement `gbm.py`**

```python
# src/mhf/models/gbm.py
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from mhf.constants import QUANTILES
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES


class GBMQuantile:
    def __init__(self, n_estimators=300, learning_rate=0.05, num_leaves=31,
                 min_child_samples=50, random_state=0, quantiles=QUANTILES):
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
            min_child_samples=min_child_samples, random_state=random_state, verbose=-1,
        )
        self.quantiles = quantiles
        self.models_: dict[tuple[int, int], LGBMRegressor] = {}

    def fit(self, panel_train: pd.DataFrame) -> "GBMQuantile":
        X = panel_train[FEATURES].to_numpy()
        for h, ycol in enumerate(Y_COLS):
            y = panel_train[ycol].to_numpy()
            for qi, q in enumerate(self.quantiles):
                model = LGBMRegressor(objective="quantile", alpha=q, **self.params)
                model.fit(X, y)
                self.models_[(h, qi)] = model
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        X = panel_rows[FEATURES].to_numpy()
        n = len(panel_rows)
        out = np.empty((n, len(Y_COLS), len(self.quantiles)))
        for (h, qi), model in self.models_.items():
            out[:, h, qi] = model.predict(X)
        out.sort(axis=2)  # enforce p10 <= p50 <= p90
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/models/test_gbm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mhf/models/gbm.py tests/models/test_gbm.py
git commit -m "feat: LightGBM per-horizon per-quantile regressors with crossing repair"
```

---

## Task 8: Chronos-Bolt fine-tune wrapper

**Files:**
- Create: `src/mhf/models/chronos_ft.py`
- Test: `tests/models/test_chronos_ft.py`

**Interfaces:**
- Consumes: `settings.horizons`, `QUANTILES`, `cache_path` / `download_ohlcv` (raw Close series), `Y_COLS`.
- Produces:
  - `to_tsdf(series_by_ticker: dict[str, pd.Series]) -> TimeSeriesDataFrame` — pure transform: dict of causal log-price series → AutoGluon long-format frame (columns `item_id`, `timestamp`, `target`).
  - `forecast_to_return_quantiles(pred_df, anchor_log_price, horizons=settings.horizons) -> np.ndarray` — pure transform: AutoGluon per-step quantile forecast (columns `"0.1","0.5","0.9"`, index a MultiIndex `(item_id, timestamp)` giving 126 future log-price steps for one item) + the anchor's last observed log price → forward-return quantiles `(n_horizons, n_quantiles)` via `exp(q_logprice[step_h] - anchor) - 1`, reading steps `[5, 21, 126]`. Sort along quantile axis.
  - `class ChronosForecaster` with `fit(train_series, fine_tune_steps=1000, prediction_length=126) -> self` (wraps `TimeSeriesPredictor(...).fit(...)`) and `predict_quantiles(panel_rows) -> np.ndarray` shape `(len(rows), n_horizons, n_quantiles)` (for each row, truncates that ticker's log-price series to ≤ its `end_date`, predicts, converts). `save(path)` / `load(path)` classmethod.

**Context:** The two `to_tsdf` / `forecast_to_return_quantiles` transforms are pure and MUST be unit-tested with synthetic AutoGluon-shaped data (fast, no GPU, no model download). The actual `fit` (which downloads the pretrained checkpoint and fine-tunes) is validated by the entrypoint smoke run and the repo-debugger pass, NOT in unit tests — mark that test `@pytest.mark.slow` and skip by default. Target series is **log price**; returns are recovered multiplicatively so quantile monotonicity is preserved by the monotonic `exp`.

**Known hazard for the debug pass (do not pre-solve here):** yfinance calendars have holiday gaps, so `freq="B"` will make AutoGluon insert NaN targets on holidays and may error or forward-fill. The clean fix (apply during the repo-debugger pass once the real library behaviour is observed) is to reindex each series to its own regular business-day range and `ffill` the log price before `to_tsdf`, OR let AutoGluon infer an irregular freq. Leaving `freq="B"` here is intentional — the smoke run is where this gets tuned against the actual installed AutoGluon version. Also: `forecast_to_return_quantiles` indexes forecast step `h` as row `h-1`, which assumes the predicted frame is contiguous from the anchor; verify that assumption holds under whatever freq handling the debug pass settles on.

- [ ] **Step 1: Write the failing tests (pure transforms only)**

```python
# tests/models/test_chronos_ft.py
import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.models.chronos_ft import forecast_to_return_quantiles, to_tsdf


def test_to_tsdf_long_format():
    s1 = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    s2 = pd.Series([4.0, 5.0], index=pd.bdate_range("2020-01-01", periods=2))
    tsdf = to_tsdf({"AAA": s1, "BBB": s2})
    df = tsdf.reset_index() if hasattr(tsdf, "reset_index") else tsdf
    assert set(df["item_id"]) == {"AAA", "BBB"}
    assert "target" in df.columns and "timestamp" in df.columns
    assert len(df) == 5


def test_forecast_to_return_quantiles_math():
    # Build a fake 126-step forecast where log-price rises linearly by 0.001/step
    steps = 126
    ts = pd.bdate_range("2021-01-01", periods=steps)
    anchor = 4.0  # log price at anchor
    logp = anchor + 0.001 * np.arange(1, steps + 1)
    idx = pd.MultiIndex.from_product([["AAA"], ts], names=["item_id", "timestamp"])
    pred = pd.DataFrame(
        {"0.1": logp - 0.01, "0.5": logp, "0.9": logp + 0.01}, index=idx
    )
    out = forecast_to_return_quantiles(pred, anchor_log_price=anchor)
    assert out.shape == (3, len(QUANTILES))
    # median return at 21d = exp(0.001*21) - 1
    assert abs(out[1, 1] - (np.exp(0.001 * 21) - 1)) < 1e-6
    assert (np.diff(out, axis=1) >= -1e-9).all()  # monotone quantiles
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/models/test_chronos_ft.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.models.chronos_ft'`

- [ ] **Step 3: Implement `chronos_ft.py`**

```python
# src/mhf/models/chronos_ft.py
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.constants import QUANTILES
from mhf.config import settings

logger = logging.getLogger(__name__)
_QCOLS = [str(q) for q in QUANTILES]


def to_tsdf(series_by_ticker: dict[str, pd.Series]):
    from autogluon.timeseries import TimeSeriesDataFrame

    frames = []
    for ticker, s in series_by_ticker.items():
        s = s.dropna()
        frames.append(pd.DataFrame({
            "item_id": ticker,
            "timestamp": pd.to_datetime(s.index),
            "target": s.to_numpy(dtype=float),
        }))
    long_df = pd.concat(frames, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(long_df)


def forecast_to_return_quantiles(pred_df: pd.DataFrame, anchor_log_price: float,
                                 horizons=None) -> np.ndarray:
    if horizons is None:
        horizons = settings.horizons
    # pred_df is one item's 126-step forecast; rows already time-ordered.
    steps = list(horizons.values())
    q = pred_df[_QCOLS].to_numpy()  # (126, n_quantiles) of forecast log-price
    out = np.empty((len(steps), len(_QCOLS)))
    for i, h in enumerate(steps):
        out[i] = np.exp(q[h - 1] - anchor_log_price) - 1.0
    out.sort(axis=1)
    return out


class ChronosForecaster:
    def __init__(self, prediction_length: int = 126, quantiles=QUANTILES):
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.predictor_ = None
        self._series: dict[str, pd.Series] = {}

    def set_series(self, series_by_ticker: dict[str, pd.Series]) -> "ChronosForecaster":
        # log-price series per ticker, indexed by date (causal, full history)
        self._series = {k: np.log(v.dropna()) for k, v in series_by_ticker.items()}
        return self

    def fit(self, train_series: dict[str, pd.Series], fine_tune_steps: int = 1000):
        from autogluon.timeseries import TimeSeriesPredictor

        self.set_series(train_series)
        train_data = to_tsdf(self._series)
        self.predictor_ = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            quantile_levels=list(self.quantiles),
            target="target",
            freq="B",
        ).fit(
            train_data,
            hyperparameters={
                "Chronos": {
                    "model_path": "bolt_small",
                    "fine_tune": True,
                    "fine_tune_steps": fine_tune_steps,
                    "ag_args": {"name_suffix": "FineTuned"},
                }
            },
            enable_ensemble=False,
            verbosity=1,
        )
        return self

    def predict_quantiles(self, panel_rows: pd.DataFrame) -> np.ndarray:
        assert self.predictor_ is not None, "fit first"
        n = len(panel_rows)
        out = np.empty((n, len(settings.horizons), len(self.quantiles)))
        for i, (_, row) in enumerate(panel_rows.reset_index(drop=True).iterrows()):
            s = self._series[row["ticker"]]
            hist = s[s.index <= pd.Timestamp(row["end_date"])]
            ctx = to_tsdf({row["ticker"]: hist})
            pred = self.predictor_.predict(ctx)
            item_pred = pred.loc[row["ticker"]]
            out[i] = forecast_to_return_quantiles(item_pred, anchor_log_price=float(hist.iloc[-1]))
        return out

    def save(self, path: str | Path) -> None:
        self.predictor_.save(str(path))

    @classmethod
    def load(cls, path: str | Path, series_by_ticker: dict[str, pd.Series]):
        from autogluon.timeseries import TimeSeriesPredictor

        obj = cls()
        obj.predictor_ = TimeSeriesPredictor.load(str(path))
        obj.set_series(series_by_ticker)
        return obj
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/models/test_chronos_ft.py -v`
Expected: PASS (2 pure-transform tests). The `fit`/`predict` library path is intentionally NOT unit-tested here — it is exercised by the Task 10 smoke run.

- [ ] **Step 5: Commit**

```bash
git add src/mhf/models/chronos_ft.py tests/models/test_chronos_ft.py
git commit -m "feat: Chronos-Bolt fine-tune wrapper + pure return-quantile transforms"
```

---

## Task 9: Ensemble (convex quantile blend)

**Files:**
- Create: `src/mhf/models/ensemble.py`
- Test: `tests/models/test_ensemble.py`

**Interfaces:**
- Consumes: `pinball_loss`, `QUANTILES`.
- Produces:
  - `fit_blend_weight(preds_a, preds_b, y, quantiles=QUANTILES) -> float` — searches `w in [0,1]` (grid of 0.0..1.0 step 0.05) minimizing pinball loss of `w*preds_a + (1-w)*preds_b` against `y`, per horizon aggregated (sum). `preds_*` are `(n, n_horizons, n_quantiles)`, `y` is `(n, n_horizons)`. Returns the best `w` (weight on `preds_a`).
  - `blend(preds_a, preds_b, w) -> np.ndarray` — `w*preds_a + (1-w)*preds_b`, then sort along quantile axis.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_ensemble.py
import numpy as np

from mhf.models.ensemble import blend, fit_blend_weight


def test_blend_prefers_the_better_model():
    rng = np.random.default_rng(0)
    n, H, Q = 200, 3, 3
    y = rng.normal(0, 0.05, size=(n, H))
    # model A is near-perfect, model B is noise
    good = np.repeat(y[:, :, None], Q, axis=2) + rng.normal(0, 0.001, size=(n, H, Q))
    bad = rng.normal(0, 0.5, size=(n, H, Q))
    w = fit_blend_weight(good, bad, y)
    assert w > 0.8  # weight should land mostly on the good model


def test_blend_is_convex_and_sorted():
    a = np.array([[[0.0, 0.1, 0.3]]])
    b = np.array([[[0.2, 0.1, 0.0]]])
    out = blend(a, b, 0.5)
    assert out.shape == a.shape
    assert (np.diff(out, axis=2) >= 0).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/models/test_ensemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.models.ensemble'`

- [ ] **Step 3: Implement `ensemble.py`**

```python
# src/mhf/models/ensemble.py
import numpy as np

from mhf.constants import QUANTILES
from mhf.eval.metrics import pinball_loss


def blend(preds_a: np.ndarray, preds_b: np.ndarray, w: float) -> np.ndarray:
    out = w * preds_a + (1.0 - w) * preds_b
    out = np.sort(out, axis=2)
    return out


def fit_blend_weight(preds_a, preds_b, y, quantiles=QUANTILES) -> float:
    best_w, best_loss = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        blended = blend(preds_a, preds_b, w)
        loss = sum(
            pinball_loss(y[:, h], blended[:, h, :], quantiles)
            for h in range(y.shape[1])
        )
        if loss < best_loss:
            best_loss, best_w = loss, float(w)
    return best_w
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/models/test_ensemble.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mhf/models/ensemble.py tests/models/test_ensemble.py
git commit -m "feat: convex quantile-blend ensemble fit on validation pinball loss"
```

---

## Task 10: Training entrypoint

**Files:**
- Create: `src/mhf/train.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_training(panel: pd.DataFrame, series_by_ticker: dict[str, pd.Series], *, n_folds=4, fine_tune_steps=1000, use_chronos=True, out_dir=None, wandb_project=None) -> dict` — the orchestration. Returns a metrics dict. Writes artifacts to `out_dir` (default `settings.data_dir / "artifacts"`): `metrics.json`, `feature_reference.parquet` (train-set feature means/stds for drift monitoring in a later plan), `gbm/` (pickled `GBMQuantile`), `chronos/` (AutoGluon predictor via `ChronosForecaster.save`), and `model_card.md`.
  - `main()` CLI: `python -m mhf.train [--smoke] [--no-chronos] [--fine-tune-steps N] [--wandb-project NAME]`. `--smoke` builds the panel from the first 3 tickers, `n_folds=2`, `fine_tune_steps=1`, so the whole path runs in minutes for debugging.

**Design of `run_training`:**
1. Determine the monthly **anchor grid**: within the panel, keep one row per (ticker, month-end) by taking, per ticker, every row whose `end_date` is the last available on-or-before each month boundary. (Helper `monthly_anchors(panel)`.)
2. `folds = walk_forward_folds(panel["end_date"].to_numpy(), n_folds=n_folds)`.
3. For each fold: fit `HistoricalQuantile` and `GBMQuantile` on `panel[fold.train]`; predict on the anchor rows in `panel[fold.test]`; accumulate per-model quantile predictions + realized `Y_COLS`.
4. Chronos (if `use_chronos`): fine-tune once on log-price series truncated to the **first fold's** train max date; predict on all test anchor rows.
5. Fit ensemble weight on the first fold's test anchors (validation), apply to the rest; report blended metrics.
6. Compute `pinball_loss`, `coverage`, `directional_hit_rate`, `information_coefficient` per horizon for every model (baseline, GBM, Chronos, ensemble); assemble metrics dict; write artifacts + model card; optional W&B log.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_smoke.py
import numpy as np
import pandas as pd

from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.train import monthly_anchors, run_training


def _panel(n_per=900, tickers=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in tickers:
        X = rng.normal(size=(n_per, len(FEATURES)))
        df = pd.DataFrame(X, columns=FEATURES)
        sig = 0.02 * X[:, 0]
        for c in Y_COLS:
            df[c] = sig + rng.normal(0, 0.02, size=n_per)
        df["ticker"] = t
        df["end_date"] = pd.bdate_range("2015-01-01", periods=n_per)
        df["base_close"] = 100.0
        frames.append(df)
    cols = ["ticker", "end_date", *FEATURES, *Y_COLS, "base_close"]
    return pd.concat(frames, ignore_index=True)[cols]


def test_monthly_anchors_thins_to_month_ends():
    panel = _panel(n_per=200, tickers=("AAA",))
    anchors = monthly_anchors(panel)
    assert len(anchors) < len(panel)
    assert len(anchors) >= 6  # ~9 months of business days


def test_run_training_smoke_no_chronos(tmp_path):
    panel = _panel()
    metrics = run_training(
        panel, series_by_ticker={}, n_folds=2, use_chronos=False, out_dir=tmp_path
    )
    assert "gbm" in metrics and "baseline" in metrics
    for model in ("gbm", "baseline"):
        for h in Y_COLS:
            assert np.isfinite(metrics[model][h]["pinball"])
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "feature_reference.parquet").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mhf.train'`

- [ ] **Step 3: Implement `train.py`**

```python
# src/mhf/train.py
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mhf.config import settings
from mhf.data.assemble import Y_COLS
from mhf.data.features import FEATURES
from mhf.eval.cv import walk_forward_folds
from mhf.eval.metrics import (
    coverage,
    directional_hit_rate,
    information_coefficient,
    pinball_loss,
)
from mhf.models.baselines import HistoricalQuantile, RandomWalk
from mhf.models.ensemble import blend, fit_blend_weight
from mhf.models.gbm import GBMQuantile

logger = logging.getLogger(__name__)


def monthly_anchors(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["_ym"] = p["end_date"].dt.to_period("M")
    idx = p.groupby(["ticker", "_ym"])["end_date"].idxmax()
    return panel.loc[idx].sort_values(["end_date", "ticker"]).reset_index(drop=True)


def _score(y: np.ndarray, q: np.ndarray) -> dict:
    # y: (n, n_horizons); q: (n, n_horizons, n_quantiles)
    out = {}
    for h, name in enumerate(Y_COLS):
        out[name] = {
            "pinball": pinball_loss(y[:, h], q[:, h, :]),
            "coverage": coverage(y[:, h], q[:, h, 0], q[:, h, 2]),
            "hit_rate": directional_hit_rate(y[:, h], q[:, h, 1]),
            "ic": information_coefficient(y[:, h], q[:, h, 1]),
        }
    return out


def run_training(panel, series_by_ticker, *, n_folds=4, fine_tune_steps=1000,
                 use_chronos=True, out_dir=None, wandb_project=None) -> dict:
    out_dir = Path(out_dir) if out_dir else settings.data_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors = monthly_anchors(panel)
    folds = walk_forward_folds(panel["end_date"].to_numpy(), n_folds=n_folds)

    rw_q, baseline_q, gbm_q, y_all = [], [], [], []
    last_gbm = None
    for fold in folds:
        train = panel[fold.train]
        test_anchor = anchors[anchors["end_date"].isin(panel.loc[fold.test, "end_date"])]
        if len(test_anchor) == 0:
            continue
        rw = RandomWalk().fit(train)
        base = HistoricalQuantile().fit(train)
        gbm = GBMQuantile().fit(train)
        last_gbm = gbm
        rw_q.append(rw.predict_quantiles(test_anchor))
        baseline_q.append(base.predict_quantiles(test_anchor))
        gbm_q.append(gbm.predict_quantiles(test_anchor))
        y_all.append(test_anchor[Y_COLS].to_numpy())

    rw_q = np.concatenate(rw_q)
    baseline_q = np.concatenate(baseline_q)
    gbm_q = np.concatenate(gbm_q)
    y_all = np.concatenate(y_all)

    metrics = {
        "random_walk": _score(y_all, rw_q),
        "baseline": _score(y_all, baseline_q),
        "gbm": _score(y_all, gbm_q),
    }

    if use_chronos and series_by_ticker:
        from mhf.models.chronos_ft import ChronosForecaster

        train_max = panel.loc[folds[0].train, "end_date"].max()
        train_series = {
            t: s[s.index <= train_max] for t, s in series_by_ticker.items()
        }
        chronos = ChronosForecaster().fit(train_series, fine_tune_steps=fine_tune_steps)
        chronos.set_series(series_by_ticker)
        # re-predict on the same test anchors, fold by fold, in the same order so
        # rows align 1:1 with y_all / gbm_q above.
        chronos_q = []
        for fold in folds:
            test_anchor = anchors[anchors["end_date"].isin(panel.loc[fold.test, "end_date"])]
            if len(test_anchor) == 0:
                continue
            chronos_q.append(chronos.predict_quantiles(test_anchor))
        chronos_q = np.concatenate(chronos_q)
        metrics["chronos"] = _score(y_all, chronos_q)
        w = fit_blend_weight(chronos_q, gbm_q, y_all)
        ens = blend(chronos_q, gbm_q, w)
        metrics["ensemble"] = _score(y_all, ens)
        metrics["blend_weight_chronos"] = w
        chronos.save(out_dir / "chronos")

    # artifacts
    if last_gbm is not None:
        (out_dir / "gbm").mkdir(exist_ok=True)
        with open(out_dir / "gbm" / "model.pkl", "wb") as f:
            pickle.dump(last_gbm, f)
    ref = panel[FEATURES].agg(["mean", "std"]).T
    ref.to_parquet(out_dir / "feature_reference.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_model_card(out_dir / "model_card.md", metrics)

    if wandb_project:
        _log_wandb(wandb_project, metrics)
    return metrics


def _write_model_card(path: Path, metrics: dict) -> None:
    lines = ["# Model Card — Multi-Horizon Probabilistic Equity Forecaster", ""]
    lines.append("Out-of-sample metrics (purged/embargoed walk-forward CV):\n")
    for model, per_h in metrics.items():
        if not isinstance(per_h, dict):
            lines.append(f"- **{model}**: {per_h}")
            continue
        lines.append(f"## {model}")
        for h, m in per_h.items():
            lines.append(f"- {h}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    path.write_text("\n".join(lines))


def _log_wandb(project: str, metrics: dict) -> None:
    import wandb

    run = wandb.init(project=project, job_type="train")
    flat = {}
    for model, per_h in metrics.items():
        if isinstance(per_h, dict):
            for h, m in per_h.items():
                for k, v in m.items():
                    flat[f"{model}/{h}/{k}"] = v
        else:
            flat[model] = per_h
    run.log(flat)
    run.finish()


def _build_inputs(smoke: bool):
    from mhf.data.assemble import build_panel
    from mhf.data.ingest import download_ohlcv, fetch_market, fetch_sp500

    tickers, _ = fetch_sp500()
    if smoke:
        tickers = tickers[:3]
    market = fetch_market()
    panel = build_panel(tickers, market)
    series = {}
    for t in panel["ticker"].unique():
        df = download_ohlcv(t)
        if df is not None:
            series[t] = df["Close"]
    return panel, series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-chronos", action="store_true")
    ap.add_argument("--fine-tune-steps", type=int, default=1000)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    panel, series = _build_inputs(args.smoke)
    metrics = run_training(
        panel, series,
        n_folds=2 if args.smoke else 4,
        fine_tune_steps=1 if args.smoke else args.fine_tune_steps,
        use_chronos=not args.no_chronos,
        wandb_project=args.wandb_project,
    )
    logger.info("done; metrics written")
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_train_smoke.py -v`
Expected: PASS (2 tests). The smoke test exercises the full CPU path (panel → CV → baseline+GBM → scoring → artifacts) without Chronos.

- [ ] **Step 5: Commit**

```bash
git add src/mhf/train.py tests/test_train_smoke.py
git commit -m "feat: training entrypoint — walk-forward CV, ensemble, artifacts, model card"
```

---

## After all tasks (controller, before handing off the run)

1. **Whole-branch review** (most-capable model) over Tasks 1–10.
2. **Repo debugger skill** end-to-end: static pass + `python -m mhf.train --smoke` on 2–3 tickers (real yfinance + real Chronos fine-tune with `fine_tune_steps=1`). Fix every bug it surfaces. Do NOT skip Chronos in this pass — the smoke run is the only automated check that the AutoGluon library path actually works on this machine.
3. Only once the smoke run is green, hand the user the real command:
   `pip install -e ".[train]"` then `CUDA_VISIBLE_DEVICES=0 python -m mhf.train --fine-tune-steps 2000 --wandb-project mhf-forecasting`
   (free CPU on the machine first, per the user).

---

## Deferred to later plans (not in scope here)

- Cost-aware backtest (`src/eval/backtest.py`) — post-hoc analysis on saved predictions; belongs with the eval-reporting/notebook plan.
- FastAPI serving, HF Space, Next.js frontend, W&B/HF-Hub registry wiring, CI, drift monitoring — Plans 03+.
- Per-fold Chronos re-fine-tuning — documented limitation; current design fine-tunes once on the earliest train span and forecasts forward (honest, conservative).
