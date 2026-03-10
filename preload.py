"""
preload.py -- Run this ONCE to download and cache all data.

Usage:
    python preload.py

Downloads:
  * S&P 500 constituent history (GitHub)
  * Adjusted-close prices for all ~1194 historical S&P 500 tickers (yfinance)
  * Current market caps for all tickers (concurrent, with per-ticker timeout)
  * Historical shares outstanding from SEC EDGAR XBRL frames (2009-present)
  * Split history per ticker from yfinance
  * Historical market-cap matrix (EDGAR-corrected 2009+, price-ratio for pre-2009)
  * S&P 500 benchmark (^GSPC)

Historical market-cap accuracy:
  2009-present: Uses actual shares-outstanding from SEC EDGAR filings combined
                with split-adjustment factors. Correctly handles buybacks,
                dilutions, and reverse splits (e.g., AIG post-bailout).
  Pre-2009:     Approximation: hist_mcap = adj_price * (current_mcap/current_price).
                This can be inaccurate for stocks with large share-count changes
                (e.g., AIG in 2002-2008). Rankings in this period are directionally
                correct for stable companies but may misrank outliers.

Everything is saved under cache/ as parquet or pickle files.
After preload completes, the Streamlit app loads all data in ~1 second.
"""

import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

PRICES_FILE       = CACHE_DIR / "prices_all.parquet"
MCAP_HIST_FILE    = CACHE_DIR / "mcap_hist.parquet"
MCAP_CUR_FILE     = CACHE_DIR / "market_caps_current.pkl"
BENCHMARK_FILE    = CACHE_DIR / "benchmark_all.pkl"
HISTORY_FILE      = CACHE_DIR / "sp500_constituents.pkl"
SHARES_EDGAR_FILE = CACHE_DIR / "shares_edgar.parquet"
SPLITS_FILE       = CACHE_DIR / "splits.pkl"

START_DATE = "1996-01-01"
END_DATE   = str(date.today())

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"
)

SEC_HEADERS = {"User-Agent": "SP500Backtest research@example.com"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(label: str, done: int, total: int, width: int = 40) -> str:
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "#" * filled + "." * (width - filled)
    return f"\r  {label} [{bar}] {done}/{total} ({pct:.0%})"


def _normalize(t: str) -> str:
    return t.replace(".", "-")


# ---------------------------------------------------------------------------
# Step 1 -- Constituent history
# ---------------------------------------------------------------------------

def fetch_history() -> pd.DataFrame:
    if HISTORY_FILE.exists():
        print("  [OK] Constituent history already cached -- loading from disk.")
        with open(HISTORY_FILE, "rb") as f:
            return pickle.load(f)

    print("  Downloading S&P 500 constituent history from GitHub...")
    resp = requests.get(SP500_HISTORY_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "tickers"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ticker_list"] = df["tickers"].apply(
        lambda x: [_normalize(t.strip()) for t in str(x).split(",") if t.strip()]
    )
    df = df.drop(columns=["tickers"])
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"  [OK] Saved -- {len(df):,} daily snapshots, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 -- Prices
# ---------------------------------------------------------------------------

def fetch_prices(tickers: list) -> pd.DataFrame:
    if PRICES_FILE.exists():
        print("  [OK] Prices already cached -- loading from disk...", end="", flush=True)
        t0 = time.time()
        df = pd.read_parquet(PRICES_FILE)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            print(f" {df.shape[1]} tickers x {df.shape[0]} days ({time.time()-t0:.1f}s)")
            return df
        print(f"\n  Cache exists but {len(missing)} tickers missing -- downloading those.")
        existing = df
    else:
        existing = pd.DataFrame()
        missing = tickers

    batch_size = 200
    batches = [missing[i : i + batch_size] for i in range(0, len(missing), batch_size)]
    total = len(batches)
    new_data: dict = {}

    for idx, batch in enumerate(batches):
        print(_bar("Prices", idx + 1, total), end="", flush=True)
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue
            close = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close, pd.DataFrame):
                for t in batch:
                    if t in close.columns:
                        col = close[t].dropna()
                        if not col.empty:
                            new_data[t] = close[t]
            else:
                new_data[batch[0]] = close
        except Exception as e:
            print(f"\n  [!] Batch {idx+1} error: {e}", end="")

    print()

    if new_data:
        new_df = pd.DataFrame(new_data)
        if not existing.empty:
            combined = pd.concat([existing, new_df], axis=1)
            combined = combined.loc[:, ~combined.columns.duplicated()]
        else:
            combined = new_df
    else:
        combined = existing

    combined = combined.sort_index()
    combined.to_parquet(PRICES_FILE, compression="snappy")
    print(f"  [OK] Saved prices: {combined.shape[1]:,} tickers x {combined.shape[0]:,} days")
    return combined


# ---------------------------------------------------------------------------
# Step 3 -- Market caps (current, for legacy formula fallback)
# ---------------------------------------------------------------------------

def fetch_market_caps(history_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
    if MCAP_CUR_FILE.exists():
        with open(MCAP_CUR_FILE, "rb") as f:
            cached: dict = pickle.load(f)
    else:
        cached = {}

    current_members = set(history_df.iloc[-1]["ticker_list"])
    one_year_ago = prices_df.index[-1] - pd.Timedelta(days=365)
    recently_active = set(
        t for t in prices_df.columns
        if prices_df[t].loc[one_year_ago:].notna().sum() > 100
    )
    target = sorted(current_members | recently_active)

    missing = [t for t in target if t not in cached]
    if not missing:
        print(f"  [OK] Market caps already cached for all active tickers ({len(cached):,}).")
        return cached

    print(f"  Fetching market caps for {len(missing):,} active tickers "
          f"(5 workers, retries on rate-limit)...")

    done = 0
    total = len(missing)

    def _fetch_with_retry(t: str, retries: int = 3) -> tuple:
        for attempt in range(retries):
            try:
                mc = yf.Ticker(t).fast_info.market_cap
                if mc and mc > 0:
                    return t, float(mc)
                return t, None
            except Exception as e:
                err = str(e)
                if "401" in err or "Unauthorized" in err or "Crumb" in err:
                    time.sleep(2 ** attempt)
                    continue
                return t, None
        return t, None

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_with_retry, t): t for t in missing}
        for future in as_completed(futures):
            try:
                t, mc = future.result(timeout=20)
                if mc:
                    cached[t] = mc
            except Exception:
                pass
            done += 1
            if done % 25 == 0 or done == total:
                print(_bar("Market caps", done, total), end="", flush=True)

    print()
    with open(MCAP_CUR_FILE, "wb") as f:
        pickle.dump(cached, f)
    got = len([t for t in target if t in cached])
    print(f"  [OK] Market caps: {got}/{len(target)} active tickers resolved.")
    return cached


def _filter_edgar_outliers(df: pd.DataFrame, max_ratio: float = 20.0) -> pd.DataFrame:
    """
    Remove per-ticker values that are more than max_ratio × the column median.
    Handles EDGAR filers that mistakenly report shares in thousands/millions,
    producing values up to 1000× larger than actual shares outstanding.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        med = s.median()
        if med <= 0:
            continue
        bad = (df[col] > med * max_ratio) | ((df[col] > 0) & (df[col] < med / max_ratio))
        df.loc[bad, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Step 4 -- Historical shares from SEC EDGAR XBRL (2009-present)
#
# Uses the EDGAR frames API which returns shares-outstanding for ALL companies
# per calendar quarter. ~64 requests for the full 2009-present period.
# Accuracy: actual reported shares, not a price-ratio approximation.
# ---------------------------------------------------------------------------

def fetch_shares_edgar(tickers: list, prices_df: pd.DataFrame) -> pd.DataFrame:
    if SHARES_EDGAR_FILE.exists():
        print("  [OK] EDGAR shares already cached -- loading from disk...", end="", flush=True)
        t0 = time.time()
        df = pd.read_parquet(SHARES_EDGAR_FILE)
        print(f" {df.shape[1]} tickers x {df.shape[0]} days ({time.time()-t0:.1f}s)")
        return df

    print("  Downloading historical shares from SEC EDGAR XBRL frames...")

    # CIK -> ticker mapping
    resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                        headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    cik_to_ticker = {v["cik_str"]: v["ticker"] for v in resp.json().values()}
    tickers_set = set(tickers)

    # Build list of quarters to fetch
    today = date.today()
    quarters = []
    for year in range(2009, today.year + 1):
        for q in range(1, 5):
            if year == today.year and q > (today.month - 1) // 3 + 1:
                break
            quarters.append((year, q))

    records = []  # list of (date, ticker, shares)
    total = len(quarters)

    for i, (year, q) in enumerate(quarters):
        if i % 8 == 0 or i == total - 1:
            print(_bar("EDGAR frames", i + 1, total), end="", flush=True)

        url = (f"https://data.sec.gov/api/xbrl/frames/dei/"
               f"EntityCommonStockSharesOutstanding/shares/CY{year}Q{q}I.json")
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            for rec in resp.json().get("data", []):
                ticker = cik_to_ticker.get(rec["cik"])
                if ticker and ticker in tickers_set:
                    records.append((pd.Timestamp(rec["end"]), ticker, int(rec["val"])))
        except Exception:
            pass

        time.sleep(0.13)  # stay under SEC 10 req/sec limit

    print()

    if not records:
        print("  [!] No EDGAR share data retrieved. Using legacy formula only.")
        return pd.DataFrame(index=prices_df.index)

    # Build sparse DataFrame from records
    df_r = pd.DataFrame(records, columns=["date", "ticker", "shares"])
    df_r["date"] = pd.to_datetime(df_r["date"]).dt.tz_localize(None)
    df_r = df_r.sort_values("date").drop_duplicates(["date", "ticker"], keep="last")
    shares_sparse = df_r.pivot(index="date", columns="ticker", values="shares").astype(float)

    # Outlier filtering: EDGAR filers sometimes report shares in thousands/millions
    # instead of actual units, creating values 1000x too large. Remove values that
    # deviate more than 20x from the per-ticker median before forward-filling.
    shares_sparse = _filter_edgar_outliers(shares_sparse)

    # Forward-fill to daily price index
    full_index = prices_df.index
    combined_idx = full_index.union(shares_sparse.index).sort_values()
    shares_daily = shares_sparse.reindex(combined_idx).ffill()
    shares_daily = shares_daily.reindex(full_index)

    # Zero out anything before EDGAR coverage starts (avoid using stale forward-fill)
    edgar_start = pd.Timestamp("2009-06-01")  # first full quarter filings
    shares_daily.loc[:edgar_start] = np.nan

    shares_daily.to_parquet(SHARES_EDGAR_FILE, compression="snappy")
    n_tickers = int(shares_daily.notna().any().sum())
    print(f"  [OK] EDGAR shares: {n_tickers} tickers with data from ~2009 to {today}")
    return shares_daily


# ---------------------------------------------------------------------------
# Step 5 -- Split history (needed to convert reported shares to adj-equivalent)
# ---------------------------------------------------------------------------

def fetch_splits(tickers: list) -> dict:
    if SPLITS_FILE.exists():
        print("  [OK] Splits already cached.")
        with open(SPLITS_FILE, "rb") as f:
            return pickle.load(f)

    print(f"  Downloading split history for {len(tickers):,} tickers (30 workers)...")

    splits_dict: dict = {}
    done = 0
    total = len(tickers)

    def _fetch(t: str) -> tuple:
        try:
            s = yf.Ticker(t).splits
            return t, s if (s is not None and not s.empty) else None
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(_fetch, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                t, s = future.result(timeout=10)
                if s is not None:
                    splits_dict[t] = s
            except Exception:
                pass
            done += 1
            if done % 50 == 0 or done == total:
                print(_bar("Splits", done, total), end="", flush=True)

    print()
    with open(SPLITS_FILE, "wb") as f:
        pickle.dump(splits_dict, f)
    print(f"  [OK] Splits: {len(splits_dict)} tickers have recorded splits.")
    return splits_dict


def _cum_split_factor_series(price_index: pd.DatetimeIndex, splits: pd.Series) -> np.ndarray:
    """
    For each date in price_index, return the product of all split ratios that
    occurred STRICTLY AFTER that date.

    This converts reported (actual) shares at date t to the split-adjusted
    equivalent relative to today:
        shares_adj(t) = actual_shares(t) * cum_factor(t)

    Then: hist_mcap(t) = adj_price(t) * shares_adj(t)
                       = adj_price(t) * actual_shares(t) * cum_factor(t)
                       = actual_price(t) * actual_shares(t)   [always correct]

    Split ratios from yfinance:
      forward split 4:1  -> ratio = 4.0  (price /= 4, shares *= 4)
      reverse split 1:20 -> ratio = 0.05 (price *= 20, shares /= 20)
    """
    # Strip timezone from split index if present
    split_idx = splits.index
    if hasattr(split_idx, "tz") and split_idx.tz is not None:
        split_idx = split_idx.tz_localize(None)
    splits_clean = pd.Series(splits.values, index=split_idx).sort_index()

    if splits_clean.empty:
        return np.ones(len(price_index))

    # Reverse cumulative product: vals[i] = product of splits[i:]
    vals = splits_clean.values[::-1].cumprod()[::-1]

    # For each price date, find the index of the first split strictly after it
    split_dates = splits_clean.index.values.astype("datetime64[ns]")
    price_dates = price_index.values.astype("datetime64[ns]")
    idx = np.searchsorted(split_dates, price_dates, side="right")

    idx_safe = np.clip(idx, 0, len(vals) - 1)
    cum_factors = np.where(idx < len(vals), vals[idx_safe], 1.0)
    return cum_factors


# ---------------------------------------------------------------------------
# Step 6 -- Historical market-cap matrix (EDGAR-corrected)
# ---------------------------------------------------------------------------

def build_mcap_hist(
    prices_df: pd.DataFrame,
    current_mcaps: dict,
    shares_edgar: pd.DataFrame | None = None,
    splits_dict: dict | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Build the daily historical market-cap matrix.

    For tickers with EDGAR data (2009+):
        hist_mcap(t) = adj_price(t) * actual_shares(t) * cum_split_factor(t)
                     = actual_price(t) * actual_shares(t)   [exact]

    For pre-2009 or tickers without EDGAR data:
        hist_mcap(t) = adj_price(t) * (current_mcap / current_price)
                     [approximation; inaccurate for large share-count changes]

    Also filters tickers with current_price < $1 (near-worthless / dead companies
    whose very low current price causes the legacy formula to blow up).
    """
    if MCAP_HIST_FILE.exists() and not force_rebuild:
        print("  [OK] Historical mcap matrix already cached -- loading from disk...",
              end="", flush=True)
        t0 = time.time()
        df = pd.read_parquet(MCAP_HIST_FILE)
        print(f" {df.shape[1]} tickers x {df.shape[0]} days ({time.time()-t0:.1f}s)")
        return df

    print("  Building historical market-cap matrix (EDGAR-corrected where available)...")
    current_prices = prices_df.ffill().iloc[-1]
    result: dict = {}
    n_edgar = 0
    n_legacy = 0

    for t in prices_df.columns:
        if t not in current_mcaps:
            continue
        cp = float(current_prices.get(t, np.nan))
        if np.isnan(cp) or cp <= 0:
            continue
        # Skip near-dead companies: current_price < $1 causes legacy formula to blow up
        # (e.g. MCIC at $0.0001 -> implied $15,000T historical mcap)
        if cp < 1.0:
            continue

        prices = prices_df[t]

        # ---- EDGAR-corrected path (2009+) ----
        if shares_edgar is not None and t in shares_edgar.columns:
            s_col = shares_edgar[t]
            has_edgar = s_col.notna() & (s_col > 0)

            # Cumulative split factor: converts actual shares -> adj-equivalent
            cf = _cum_split_factor_series(
                prices.index,
                splits_dict.get(t, pd.Series(dtype=float)) if splits_dict else pd.Series(dtype=float),
            )
            cf_series = pd.Series(cf, index=prices.index)

            edgar_mcap  = prices * s_col * cf_series          # EDGAR formula
            legacy_mcap = prices * (current_mcaps[t] / cp)    # legacy fallback

            combined = legacy_mcap.copy()
            combined[has_edgar] = edgar_mcap[has_edgar]
            result[t] = combined
            n_edgar += 1

        # ---- Legacy path (no EDGAR data for this ticker) ----
        else:
            result[t] = prices * (current_mcaps[t] / cp)
            n_legacy += 1

    mcap_df = pd.DataFrame(result, index=prices_df.index)
    mcap_df.to_parquet(MCAP_HIST_FILE, compression="snappy")
    print(f"  [OK] Saved: {mcap_df.shape[1]:,} tickers x {mcap_df.shape[0]:,} days")
    print(f"       EDGAR-corrected (2009+): {n_edgar} | Legacy approximation: {n_legacy}")
    return mcap_df


# ---------------------------------------------------------------------------
# Step 7 -- Benchmark
# ---------------------------------------------------------------------------

def fetch_benchmark() -> pd.Series:
    if BENCHMARK_FILE.exists():
        print("  [OK] Benchmark already cached.")
        with open(BENCHMARK_FILE, "rb") as f:
            return pickle.load(f)

    print("  Downloading S&P 500 benchmark (^GSPC)...")
    raw = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    series = raw["Close"].squeeze()
    series.name = "SP500"
    with open(BENCHMARK_FILE, "wb") as f:
        pickle.dump(series, f)
    print(f"  [OK] Saved benchmark: {len(series):,} days")
    return series


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  S&P N Backtest -- Data Preloader")
    print(f"  Coverage: {START_DATE} -> {END_DATE}")
    print("=" * 60)
    t_total = time.time()

    print("\n[1/7] Constituent history")
    history_df = fetch_history()
    all_tickers = sorted({
        t
        for lst in history_df["ticker_list"]
        for t in lst
    })
    print(f"  {len(all_tickers):,} unique tickers across full history.")

    print("\n[2/7] Price data")
    prices_df = fetch_prices(all_tickers)

    print("\n[3/7] Current market caps")
    current_mcaps = fetch_market_caps(history_df, prices_df)

    print("\n[4/7] Historical shares from SEC EDGAR (2009-present)")
    shares_edgar = fetch_shares_edgar(all_tickers, prices_df)

    print("\n[5/7] Split history from yfinance")
    splits_dict = fetch_splits(list(prices_df.columns))

    print("\n[6/7] Historical market-cap matrix")
    # Force rebuild because we now have EDGAR + splits data
    mcap_df = build_mcap_hist(
        prices_df, current_mcaps,
        shares_edgar=shares_edgar,
        splits_dict=splits_dict,
        force_rebuild=True,
    )

    print("\n[7/7] Benchmark (^GSPC)")
    fetch_benchmark()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Preload complete in {elapsed/60:.1f} min.")
    print(f"  Cache location: {CACHE_DIR.resolve()}")
    print(f"  Note: Market cap rankings are accurate from 2009 onwards.")
    print(f"        Pre-2009 rankings are approximate (no historical share data).")
    print(f"  App is ready -- run:  python -m streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
