"""
Data loading module for S&P 500 backtest app.

Fast path (after preload.py has been run):
  All data is read from local parquet/pickle files — no network calls.

Fallback path (if preload hasn't been run):
  Downloads data on demand (slow).
"""

import os
import pickle
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Files written by preload.py
PRICES_FILE    = CACHE_DIR / "prices_all.parquet"
MCAP_HIST_FILE = CACHE_DIR / "mcap_hist.parquet"
MCAP_CUR_FILE  = CACHE_DIR / "market_caps_current.pkl"
BENCHMARK_FILE = CACHE_DIR / "benchmark_all.pkl"
HISTORY_FILE   = CACHE_DIR / "sp500_constituents.pkl"

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"
)


def is_preloaded() -> bool:
    """True when all preload.py output files are present."""
    return all(
        p.exists()
        for p in [PRICES_FILE, MCAP_HIST_FILE, MCAP_CUR_FILE, BENCHMARK_FILE, HISTORY_FILE]
    )


# ---------------------------------------------------------------------------
# Fast path — read pre-computed files
# ---------------------------------------------------------------------------

def load_preloaded(start_date: str, end_date: str):
    """
    Load all data from disk in ~1 second.
    Returns (history_df, prices_df, mcap_df, benchmark).
    """
    with open(HISTORY_FILE, "rb") as f:
        history_df = pickle.load(f)

    prices_df = pd.read_parquet(PRICES_FILE)
    mcap_df   = pd.read_parquet(MCAP_HIST_FILE)

    with open(BENCHMARK_FILE, "rb") as f:
        benchmark_full = pickle.load(f)

    # Slice to requested window
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    prices_df = prices_df.loc[start_ts:end_ts]
    mcap_df   = mcap_df.loc[start_ts:end_ts]
    benchmark = benchmark_full.loc[start_ts:end_ts]

    # Forward-fill small gaps (holidays, missing data)
    prices_df = prices_df.ffill(limit=10)
    mcap_df   = mcap_df.ffill(limit=10)

    return history_df, prices_df, mcap_df, benchmark


# ---------------------------------------------------------------------------
# Constituent helpers (used by backtest engine regardless of path)
# ---------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    return t.replace(".", "-")


def load_sp500_history(force_reload: bool = False) -> pd.DataFrame:
    if not force_reload and HISTORY_FILE.exists():
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)

    resp = requests.get(SP500_HISTORY_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "tickers"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker_list"] = df["tickers"].apply(
        lambda x: [_normalize_ticker(t.strip()) for t in str(x).split(",") if t.strip()]
    )
    df = df.drop(columns=["tickers"])
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(df, f)
    return df


def get_constituents_at_date(history_df: pd.DataFrame, target_date) -> list:
    target_date = pd.Timestamp(target_date)
    mask = history_df["date"] <= target_date
    if not mask.any():
        return list(history_df.iloc[0]["ticker_list"])
    return list(history_df[mask].iloc[-1]["ticker_list"])


def get_all_unique_tickers(history_df: pd.DataFrame) -> list:
    tickers = set()
    for lst in history_df["ticker_list"]:
        tickers.update(lst)
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Fallback path — download on demand (slow, only if preload hasn't been run)
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list,
    start_date: str,
    end_date: str,
    force_reload: bool = False,
    progress_cb=None,
) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"prices_{start_date}_{end_date}.pkl"

    cached_df = pd.DataFrame()
    if not force_reload and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cached_df = pickle.load(f)
        except Exception:
            cached_df = pd.DataFrame()

    missing = [t for t in tickers if t not in cached_df.columns]
    if not missing:
        return cached_df.reindex(columns=tickers)

    batch_size = 100
    new_data: dict = {}
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]

    for idx, batch in enumerate(batches):
        if progress_cb:
            progress_cb(idx, len(batches), batch)
        try:
            raw = yf.download(
                batch, start=start_date, end=end_date,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                continue
            if len(batch) == 1:
                ticker = batch[0]
                new_data[ticker] = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
            else:
                close = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close, pd.DataFrame):
                    for t in batch:
                        if t in close.columns:
                            new_data[t] = close[t]
                else:
                    new_data[batch[0]] = close
        except Exception as e:
            print(f"[download_prices] batch {idx} error: {e}")

    if new_data:
        new_df = pd.DataFrame(new_data)
        combined = pd.concat([cached_df, new_df], axis=1) if not cached_df.empty else new_df
        combined = combined.loc[:, ~combined.columns.duplicated()]
        with open(cache_file, "wb") as f:
            pickle.dump(combined, f)
        cached_df = combined

    return cached_df.reindex(columns=tickers)


def load_current_market_caps(tickers: list, force_reload: bool = False) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Prefer the preload file; fall back to old cache
    cache_file = MCAP_CUR_FILE if MCAP_CUR_FILE.exists() else CACHE_DIR / "market_caps.pkl"

    cached: dict = {}
    if cache_file.exists() and not force_reload:
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            missing = [t for t in tickers if t not in cached]
            if not missing:
                return cached
        except Exception:
            cached = {}

    missing = [t for t in tickers if t not in cached]
    if not missing:
        return cached

    def _fetch(t: str):
        try:
            mc = yf.Ticker(t).fast_info.market_cap
            return t, float(mc) if (mc and mc > 0) else None
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(_fetch, t): t for t in missing}
        for future in as_completed(futures):
            try:
                t, mc = future.result(timeout=5)
                if mc:
                    cached[t] = mc
            except Exception:
                pass

    with open(cache_file, "wb") as f:
        pickle.dump(cached, f)
    return cached


def build_historical_mcap(prices_df: pd.DataFrame, current_mcaps: dict) -> pd.DataFrame:
    if prices_df.empty:
        return pd.DataFrame()
    current_prices = prices_df.ffill().iloc[-1]
    result: dict = {}
    for t in prices_df.columns:
        if t not in current_mcaps:
            continue
        cp = current_prices.get(t)
        if cp is None or np.isnan(cp) or cp <= 0:
            continue
        result[t] = prices_df[t] * (current_mcaps[t] / cp)
    return pd.DataFrame(result, index=prices_df.index)


def load_benchmark(start_date: str, end_date: str) -> pd.Series:
    # Fast path: slice the preloaded full benchmark
    if BENCHMARK_FILE.exists():
        with open(BENCHMARK_FILE, "rb") as f:
            full = pickle.load(f)
        return full.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]

    # Fallback
    cache_file = CACHE_DIR / f"benchmark_{start_date}_{end_date}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    raw = yf.download("^GSPC", start=start_date, end=end_date,
                      auto_adjust=True, progress=False)
    series = raw["Close"].squeeze()
    series.name = "SP500"
    with open(cache_file, "wb") as f:
        pickle.dump(series, f)
    return series
