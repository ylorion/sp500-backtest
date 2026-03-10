"""
Backtesting engine for the S&P N strategy.

Strategy:
  Every trading day, check whether the set of top-N S&P 500 stocks by
  market cap has changed. If it has, rebalance immediately to the new
  composition (market-cap weighted within the selection). No costs unless
  trade_cost_pct / capital_gains_tax_pct are set.

No transaction costs are included by default.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core pre-computation: top-N tickers + weights for every trading day
# ---------------------------------------------------------------------------

def _build_daily_top_n(
    history_df: pd.DataFrame,
    mcap_df: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    rank_min: int,
    rank_max: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Vectorised computation of the top-N S&P 500 stocks by market cap
    for every trading day in trading_index.

    Returns
    -------
    selected_idx : (n_days, n_select) int array  — column indices into all_tickers
    weights_mat  : (n_days, n_select) float array — market-cap weights
    all_tickers  : list[str]  — column order of mcap_df (after reindex)
    """
    n_days = len(trading_index)
    n_select = rank_max - rank_min + 1

    # Align mcap to trading days; forward-fill weekend/holiday gaps
    mcap_aligned = mcap_df.reindex(trading_index, method="ffill")
    all_tickers = mcap_aligned.columns.tolist()
    n_tickers = len(all_tickers)
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}

    # ------------------------------------------------------------------ #
    # Build constituent mask: mask[day_i, ticker_i] = True iff ticker    #
    # is a valid S&P 500 member on that day.                             #
    # ------------------------------------------------------------------ #
    # history_df has one row per change-event (daily snapshots in our
    # dataset). We use searchsorted to map each trading day to the
    # correct history row, then bulk-assign with np.ix_.
    history_dates = np.array(history_df["date"].values, dtype="datetime64[ns]")
    trading_arr = np.array(trading_index.values, dtype="datetime64[ns]")
    hist_row_for_day = np.searchsorted(history_dates, trading_arr, side="right") - 1
    hist_row_for_day = np.clip(hist_row_for_day, 0, len(history_df) - 1)

    mask = np.zeros((n_days, n_tickers), dtype=bool)
    for hist_i in np.unique(hist_row_for_day):
        day_indices = np.where(hist_row_for_day == hist_i)[0]
        col_indices = np.fromiter(
            (ticker_to_idx[t]
             for t in history_df.iloc[hist_i]["ticker_list"]
             if t in ticker_to_idx),
            dtype=np.intp,
        )
        if col_indices.size and day_indices.size:
            mask[np.ix_(day_indices, col_indices)] = True

    # ------------------------------------------------------------------ #
    # Masked market cap (non-constituent or NaN -> 0)                    #
    # ------------------------------------------------------------------ #
    mcap_values = mcap_aligned.values.copy().astype(float)
    np.nan_to_num(mcap_values, copy=False, nan=0.0)
    mcap_values[~mask] = 0.0

    # ------------------------------------------------------------------ #
    # Top-N selection via partial sort (fast even for large n_tickers)   #
    # ------------------------------------------------------------------ #
    kth = min(rank_max, n_tickers - 1)
    # argpartition gives the top-kth elements (unsorted among themselves)
    top_part = np.argpartition(-mcap_values, kth=kth, axis=1)[:, :rank_max]
    # Sort only within those rank_max candidates
    top_mcap_part = np.take_along_axis(mcap_values, top_part, axis=1)
    inner_order = np.argsort(-top_mcap_part, axis=1)
    sorted_top = np.take_along_axis(top_part, inner_order, axis=1)
    # Slice to the requested rank window (0-indexed)
    selected_idx = sorted_top[:, rank_min - 1 : rank_max]   # (n_days, n_select)

    # Weights = market-cap share within the selection
    top_mcaps = np.take_along_axis(mcap_values, selected_idx, axis=1)
    row_totals = top_mcaps.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1.0, row_totals)
    weights_mat = top_mcaps / row_totals                     # (n_days, n_select)

    return selected_idx, weights_mat, all_tickers


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    history_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    mcap_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    rank_min: int,
    rank_max: int,
    progress_cb: Callable | None = None,
    trade_cost_pct: float = 0.0,
    capital_gains_tax_pct: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run the daily-checked top-N backtest.

    Checks every trading day whether the set of top-N S&P 500 stocks has
    changed; rebalances immediately (and only) when it has.

    Returns
    -------
    portfolio_values : pd.Series  — daily NAV starting at 1.0
    holding_changes  : pd.DataFrame — one row per actual change event
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    trading_index = prices_df.loc[start_ts:end_ts].index
    if len(trading_index) == 0:
        raise ValueError("No price data in the requested date range.")

    n_days = len(trading_index)
    n_select = rank_max - rank_min + 1

    if progress_cb:
        progress_cb(0.05)

    # ------------------------------------------------------------------ #
    # 1. Pre-compute top-N for every trading day (vectorised)            #
    # ------------------------------------------------------------------ #
    selected_idx, weights_mat, all_tickers = _build_daily_top_n(
        history_df, mcap_df, trading_index, rank_min, rank_max
    )
    n_tickers = len(all_tickers)
    ticker_arr = np.array(all_tickers, dtype=object)

    if progress_cb:
        progress_cb(0.45)

    # ------------------------------------------------------------------ #
    # 2. Detect days where the COMPOSITION (set of tickers) changed      #
    # ------------------------------------------------------------------ #
    # Compare sorted ticker tuples so that rank-swaps within the window
    # don't falsely trigger a rebalance (e.g. AAPL #1 <-> MSFT #1 in S&P
    # 1-2 keeps the same stocks).
    top_tickers_mat = ticker_arr[selected_idx]          # (n_days, n_select) object array
    top_sets = [frozenset(top_tickers_mat[i]) for i in range(n_days)]

    change_mask = np.zeros(n_days, dtype=bool)
    change_mask[0] = True
    for i in range(1, n_days):
        change_mask[i] = top_sets[i] != top_sets[i - 1]

    if progress_cb:
        progress_cb(0.55)

    # ------------------------------------------------------------------ #
    # 3. Build full weight matrix and daily portfolio returns            #
    # ------------------------------------------------------------------ #
    # full_weights[i, j] = portfolio weight of ticker j on day i
    full_weights = np.zeros((n_days, n_tickers))
    row_idx = np.repeat(np.arange(n_days), n_select)
    col_idx = selected_idx.ravel()
    full_weights[row_idx, col_idx] = weights_mat.ravel()

    # Daily returns matrix (fill NaN -> 0, i.e. "no return" for missing data)
    prices_aligned = prices_df.reindex(trading_index)[all_tickers]
    daily_rets = prices_aligned.pct_change().fillna(0.0).values   # (n_days, n_tickers)

    # On non-change days the portfolio drifts naturally (no rebalance),
    # so we hold a fixed number of shares. Approximate: use the weights
    # from the last rebalance date as constant for that period — standard
    # simplification for a market-cap backtest.
    port_daily = (full_weights * daily_rets).sum(axis=1)           # (n_days,)

    if progress_cb:
        progress_cb(0.65)

    # ------------------------------------------------------------------ #
    # 4. Compound NAV; apply costs/taxes only on actual change days      #
    # ------------------------------------------------------------------ #
    nav = 1.0
    nav_series: dict[pd.Timestamp, float] = {}
    holding_records: list[dict] = []

    change_day_indices = np.where(change_mask)[0]
    nav_at_prev_change = 1.0   # for capital-gains tax approximation

    for seg_i, seg_start in enumerate(change_day_indices):
        seg_end = (
            change_day_indices[seg_i + 1]
            if seg_i + 1 < len(change_day_indices)
            else n_days
        )
        seg_day = trading_index[seg_start]

        curr_set = top_sets[seg_start]
        prev_set = top_sets[seg_start - 1] if seg_start > 0 else frozenset()
        added   = curr_set - prev_set
        removed = prev_set - curr_set

        # ---- Apply costs on every rebalance except the first ----------
        if seg_i > 0:
            prev_w = full_weights[seg_start - 1]
            curr_w = full_weights[seg_start]
            turnover = float(np.abs(curr_w - prev_w).sum()) / 2.0

            if trade_cost_pct > 0.0:
                nav *= 1.0 - turnover * trade_cost_pct

            if capital_gains_tax_pct > 0.0:
                # Approximate gain on sold positions using portfolio return
                # since the last rebalancing as a proxy for stock-level gains.
                sold_fraction = float(
                    np.sum(np.maximum(0.0, prev_w - curr_w))
                )
                period_gain = max(0.0, nav / nav_at_prev_change - 1.0)
                nav *= 1.0 - sold_fraction * period_gain * capital_gains_tax_pct

        nav_at_prev_change = nav

        # ---- Record change --------------------------------------------
        holdings_str = ", ".join(
            f"{t} ({w:.1%})"
            for t, w in sorted(
                zip(top_tickers_mat[seg_start], weights_mat[seg_start]),
                key=lambda x: -x[1],
            )
        )
        holding_records.append(
            {
                "Date": seg_day.date(),
                "Holdings": holdings_str,
                "Added":   ", ".join(sorted(added))   or "—",
                "Removed": ", ".join(sorted(removed)) or "—",
            }
        )

        # ---- Compound daily returns for this segment -----------------
        for day_i in range(seg_start, seg_end):
            day = trading_index[day_i]
            if day_i > 0:
                nav *= 1.0 + port_daily[day_i]
            nav_series[day] = nav

    if progress_cb:
        progress_cb(1.0)

    portfolio_values = pd.Series(nav_series).sort_index()
    portfolio_values = portfolio_values[
        ~portfolio_values.index.duplicated(keep="last")
    ]
    holding_changes = (
        pd.DataFrame(holding_records) if holding_records else pd.DataFrame()
    )
    return portfolio_values, holding_changes


# ---------------------------------------------------------------------------
# Multi-strategy runner
# ---------------------------------------------------------------------------

def run_multi_backtest(
    strategies: list[dict],
    history_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    mcap_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    trade_cost_pct: float = 0.0,
    capital_gains_tax_pct: float = 0.0,
    progress_cb: Callable | None = None,
) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """Run multiple strategy configurations and return a results dict."""
    results = {}
    n = len(strategies)
    for i, strat in enumerate(strategies):
        def cb(pct, base=i, total=n):
            if progress_cb:
                progress_cb((base + pct) / total)

        nav, changes = run_backtest(
            history_df=history_df,
            prices_df=prices_df,
            mcap_df=mcap_df,
            start_date=start_date,
            end_date=end_date,
            rank_min=strat["rank_min"],
            rank_max=strat["rank_max"],
            trade_cost_pct=trade_cost_pct,
            capital_gains_tax_pct=capital_gains_tax_pct,
            progress_cb=cb,
        )
        results[strat["label"]] = (nav, changes)

    return results
