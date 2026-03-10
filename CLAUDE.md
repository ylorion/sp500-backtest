# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download and cache all market data (run once before first use; takes ~5-10 min)
python preload.py

# Run the Streamlit app
python -m streamlit run app.py

# Re-run preload after clearing cache (e.g. stale data or after code changes)
rm -rf cache/
python preload.py
```

## Architecture

The app is split into five modules with a strict dependency order:

```
preload.py  (one-time data download — run before first use)
     |
data_loader.py  (disk I/O only at runtime)
     |
backtest_engine.py  (pure computation)
     |
metrics_calculator.py  (pure computation)
     |
app.py  (Streamlit UI — calls all of the above)
```

### Data flow

**preload.py** downloads everything once and writes to `cache/` (7 steps):

| File | Contents |
|------|----------|
| `prices_all.parquet` | Adjusted-close prices, ~838 S&P 500 historical tickers, 1996–today |
| `market_caps_current.pkl` | Current market caps from yfinance (active tickers only) |
| `shares_edgar.parquet` | Quarterly shares-outstanding from SEC EDGAR XBRL (2009–today) |
| `splits.pkl` | Per-ticker split ratios from yfinance |
| `mcap_hist.parquet` | Historical market-cap matrix (EDGAR-corrected 2009+, see below) |
| `benchmark_all.pkl` | `^GSPC` daily close |
| `sp500_constituents.pkl` | Daily S&P 500 membership snapshots (GitHub: fja05680/sp500) |

**data_loader.py** exposes two paths:
- `is_preloaded()` / `load_preloaded(start, end)` — reads parquet files, slices to date range, ~0.3 s
- Fallback functions (`download_prices`, `load_current_market_caps`, etc.) for on-demand download if preload hasn't been run

**backtest_engine.py** — `run_backtest()` / `run_multi_backtest()`:
- Checks EVERY trading day whether the set of top-N stocks by market cap has changed.
- Rebalances immediately (and only) when the composition changes — detected via `frozenset` comparison so rank-swaps within the held window don't trigger a spurious rebalance.
- Pre-computes top-N for all days at once via vectorised NumPy (`argpartition`, `ix_`, `take_along_axis`).
- Returns `(nav_series: pd.Series, holding_changes: pd.DataFrame)`.
- Optional `trade_cost_pct` and `capital_gains_tax_pct` applied only on actual rebalance days.

**metrics_calculator.py** — all functions are stateless and take `pd.Series` of daily returns. Key: `calculate_metrics()` returns a comparison `pd.DataFrame` (strategy vs benchmark). Sortino denominator is daily-scale only (NOT annualised) — the `sqrt(252)` factor is only in the numerator.

**app.py** — Streamlit UI:
- Sidebar: date range, strategy definitions (rank_min/rank_max/label, multiple allowed), risk-free rate, transaction cost %, capital gains tax %.
- On "Run Backtest": loads data via `dl.load_preloaded()`, calls `be.run_multi_backtest()`, stores results in `st.session_state`.
- Five tabs: Performance Chart, Metrics, Holdings Changes, Annual Returns, Data & Export.
- Diamond markers on chart at every holding-change date; hover shows tickers held + added/removed.

### Key design decisions

**Historical market cap accuracy:**
- **2009–present:** `hist_mcap(t) = adj_price(t) × actual_shares(t) × cum_split_factor(t→today)` using SEC EDGAR XBRL quarterly share data. Equals `actual_price(t) × actual_shares(t)` exactly. Handles buybacks, dilutions, and reverse splits correctly.
- **Pre-2009:** Approximation: `hist_mcap(t) = adj_price(t) × (current_mcap / current_price)`. Assumes constant shares outstanding — inaccurate for stocks with major share-count changes (e.g. AIG shows as #1 in 2002 due to its 2009 1:20 reverse split + bailout dilution; actual #1 was GE). A UI warning is shown when the backtest period starts before 2009.
- Tickers with current price < $1 are excluded (dead/delisted companies whose near-zero price causes the legacy formula to blow up).
- EDGAR outlier filtering: filings that report shares in thousands instead of actual units (>20× per-ticker median) are removed before forward-filling.

**Other:**
- **Ticker normalisation:** dots → hyphens (`BRK.B` → `BRK-B`) throughout, matching yfinance.
- **Session-state caching:** loaded data keyed on `"{start_date}_{end_date}"` so changing strategies without changing the date range does not re-read disk.
- **Constituent data source:** GitHub repo `fja05680/sp500`, file `S&P 500 Historical Components & Changes(01-17-2026).csv`. One row per trading day, comma-separated tickers. Covers 1996-01-02 to 2026-01-14, ~1194 unique tickers.
