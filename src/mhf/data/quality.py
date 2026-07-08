import pandas as pd


class DataQualityError(ValueError):
    """Raised when a ticker's data is unusable and must be skipped."""


def clean_ohlcv(df: pd.DataFrame, ticker: str, min_rows: int = 200) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataQualityError(f"{ticker}: empty frame")
    if df["Close"].isna().mean() > 0.05:
        raise DataQualityError(f"{ticker}: >5% of Close is NaN")
    df = df[df["Close"] > 0].dropna(subset=["Close"]).copy()
    if len(df) < min_rows:
        raise DataQualityError(f"{ticker}: only {len(df)} rows (need >= {min_rows})")

    returns = df["Close"].pct_change()
    if (returns.abs() > 0.5).any():
        clipped = returns.clip(-0.5, 0.5)
        first = df["Close"].iloc[0]
        rebuilt = first * (1 + clipped).cumprod()
        rebuilt.iloc[0] = first
        df["Close"] = rebuilt.values
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return df
