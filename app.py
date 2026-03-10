"""
S&P N Backtest App
==================
Compare holding the top-N largest S&P 500 stocks (by market cap) versus
the full S&P 500 index.

Strategies supported:
  • S&P 1  – hold only the single largest stock
  • S&P N  – hold the N largest stocks, weighted by market cap
  • S&P N1–N2 – hold stocks ranked N1 through N2 (e.g. 2nd–5th largest)

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest_engine as be
import data_loader as dl
import metrics_calculator as mc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="S&P N Backtest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "data_loaded": False,
        "history_df": None,
        "prices_df": None,
        "mcap_df": None,
        "benchmark": None,
        "results": {},
        "last_params": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
STRATEGY_COLORS = [
    "#00b4d8", "#f77f00", "#06d6a0", "#ef476f",
    "#ffd166", "#b5838d", "#a8dadc", "#e9c46a",
]
BENCH_COLOR = "#adb5bd"

# Pre-computed low-opacity fill versions of strategy colors
STRATEGY_FILLS = [
    "rgba(0,180,216,0.15)", "rgba(247,127,0,0.15)", "rgba(6,214,160,0.15)",
    "rgba(239,71,111,0.15)", "rgba(255,209,102,0.15)", "rgba(181,131,141,0.15)",
    "rgba(168,218,220,0.15)", "rgba(233,196,106,0.15)",
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 S&P N Backtest")
    st.caption("Compare top-N S&P 500 stocks vs the full index")
    st.divider()

    # ---- Date range --------------------------------------------------------
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start", value=date(2000, 1, 1), min_value=date(1996, 1, 1),
            max_value=date.today() - timedelta(days=365),
        )
    with col2:
        end_date = st.date_input(
            "End", value=date.today() - timedelta(days=1),
            min_value=date(1997, 1, 1), max_value=date.today(),
        )

    # ---- Strategy definition -----------------------------------------------
    st.subheader("Strategies")
    st.caption("Add one or more strategies to compare simultaneously.")

    if "strategies" not in st.session_state:
        st.session_state["strategies"] = [
            {"rank_min": 1, "rank_max": 1, "label": "S&P 1"},
        ]

    def _default_label(rmin, rmax):
        if rmin == rmax:
            return f"S&P {rmin}"
        return f"S&P {rmin}–{rmax}"

    strategies_cfg = []
    to_delete = []

    for idx, strat in enumerate(st.session_state["strategies"]):
        with st.expander(f"Strategy {idx+1}: {strat['label']}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                rmin = st.number_input(
                    "From rank", min_value=1, max_value=500,
                    value=strat["rank_min"], key=f"rmin_{idx}",
                    help="1 = largest stock by market cap",
                )
            with c2:
                rmax = st.number_input(
                    "To rank", min_value=1, max_value=500,
                    value=max(strat["rank_max"], rmin), key=f"rmax_{idx}",
                )
            rmax = max(rmax, rmin)
            label = st.text_input(
                "Label", value=strat.get("label", _default_label(rmin, rmax)),
                key=f"label_{idx}",
            )
            if st.button("Remove", key=f"del_{idx}", type="secondary"):
                to_delete.append(idx)
            strategies_cfg.append({"rank_min": rmin, "rank_max": rmax, "label": label or _default_label(rmin, rmax)})

    for idx in sorted(to_delete, reverse=True):
        st.session_state["strategies"].pop(idx)
    if to_delete:
        st.rerun()

    if st.button("+ Add Strategy", use_container_width=True):
        st.session_state["strategies"].append({"rank_min": 1, "rank_max": 5, "label": "S&P 1–5"})
        st.rerun()

    # Sync strategies_cfg back
    st.session_state["strategies"] = strategies_cfg

    # ---- Backtest settings -------------------------------------------------
    st.subheader("Settings")
    rf_rate = st.slider(
        "Risk-free rate (annual %)", min_value=0.0, max_value=8.0,
        value=0.0, step=0.25,
    ) / 100

    st.subheader("Costs & Taxes")
    trade_cost_pct = st.slider(
        "Transaction cost (% per trade, one-way)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        help="Applied on the traded notional at each rebalancing.",
    ) / 100
    capital_gains_tax_pct = st.slider(
        "Capital gains tax (% on realised profit)",
        min_value=0.0, max_value=50.0, value=0.0, step=1.0,
        help="Applied on realised gains when positions are sold at rebalancing.",
    ) / 100

    st.divider()
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
    refresh_btn = st.button(
        "Refresh Data (clear cache)", use_container_width=True,
        help="Re-download all data from source",
    )

    if refresh_btn:
        import shutil
        shutil.rmtree("cache", ignore_errors=True)
        st.session_state["data_loaded"] = False
        st.success("Cache cleared.")
        st.rerun()

    st.divider()
    st.caption(
        "Data: S&P 500 constituent history via fja05680/sp500 (GitHub), "
        "prices via yfinance. "
        "Market cap 2009+: actual shares from SEC EDGAR filings. "
        "Market cap pre-2009: approximation via price × (current_mcap / current_price) — "
        "rankings in this period may be inaccurate for companies with major share-count "
        "changes (e.g. AIG post-bailout, large buyback programs)."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(start: str, end: str):
    """Load from preloaded parquet files (fast) or fall back to live download."""
    if dl.is_preloaded():
        return dl.load_preloaded(start, end)

    # Fallback: slow on-demand download (shown only when preload hasn't been run)
    history_df = dl.load_sp500_history()
    all_tickers = dl.get_all_unique_tickers(history_df)
    prices_df = dl.download_prices(all_tickers, start_date=start, end_date=end)
    prices_df = prices_df.ffill(limit=10)
    current_mcaps = dl.load_current_market_caps(all_tickers)
    mcap_df = dl.build_historical_mcap(prices_df, current_mcaps)
    benchmark = dl.load_benchmark(start, end)
    return history_df, prices_df, mcap_df, benchmark


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("S&P N Backtest")
st.markdown(
    "How does holding only the **top N largest** S&P 500 stocks compare to owning the entire index?"
)

# Warn if preload hasn't been run
if not dl.is_preloaded():
    st.warning(
        "**Data not preloaded.** For instant startup, run once in your terminal:\n\n"
        "```\npython preload.py\n```\n\n"
        "This downloads all historical prices and market caps (~10–20 min). "
        "After that, every backtest loads in under a second.",
        icon="⚠️",
    )

if run_btn:
    strategies = st.session_state["strategies"]

    # Warn about pre-2009 accuracy
    if start_date < date(2009, 1, 1):
        st.warning(
            "**Pre-2009 data note:** Market cap rankings before 2009 use an approximation "
            "(`adj_price × current_mcap / current_price`) that assumes constant shares outstanding. "
            "This breaks for companies with major share-count changes — for example, AIG appears "
            "as the largest S&P 500 stock in 2002–2008 due to its 2009 reverse split + bailout "
            "dilution, when the actual leaders were GE and MSFT. "
            "For reliable rankings, set the start date to **2009 or later**.",
            icon="⚠️",
        )

    # ---- Load data (session-state cache keyed on date range) ---------------
    data_key = f"{start_date}_{end_date}"
    if st.session_state.get("data_key") != data_key or not st.session_state["data_loaded"]:
        with st.spinner("Loading data from cache…"):
            try:
                history_df, prices_df, mcap_df, benchmark = _load_data(
                    str(start_date), str(end_date)
                )
                st.session_state.update(
                    {
                        "data_loaded": True,
                        "data_key": data_key,
                        "history_df": history_df,
                        "prices_df": prices_df,
                        "mcap_df": mcap_df,
                        "benchmark": benchmark,
                    }
                )
            except Exception as e:
                st.error(f"Data loading failed: {e}")
                st.stop()

    # Pull data from session state (may have just been loaded above)
    history_df = st.session_state["history_df"]
    prices_df = st.session_state["prices_df"]
    mcap_df = st.session_state["mcap_df"]
    benchmark = st.session_state["benchmark"]

    # ---- Run backtest(s) ---------------------------------------------------
    progress_bar = st.progress(0.0, text="Running backtest…")

    def overall_progress(pct):
        progress_bar.progress(min(pct, 1.0), text=f"Backtesting… {pct:.0%}")

    try:
        results = be.run_multi_backtest(
            strategies=strategies,
            history_df=history_df,
            prices_df=prices_df,
            mcap_df=mcap_df,
            start_date=str(start_date),
            end_date=str(end_date),
            trade_cost_pct=trade_cost_pct,
            capital_gains_tax_pct=capital_gains_tax_pct,
            progress_cb=overall_progress,
        )
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()
    finally:
        progress_bar.empty()

    st.session_state["results"] = results
    st.session_state["rf_rate"] = rf_rate
    st.session_state["benchmark"] = benchmark

# ---- Display results -------------------------------------------------------
results = st.session_state.get("results", {})
benchmark: pd.Series = st.session_state.get("benchmark")

if not results:
    st.info("Configure a strategy in the sidebar and click **Run Backtest**.")
    st.stop()

# Normalise benchmark to start at 1
bench_aligned = benchmark.loc[
    benchmark.index >= min(v[0].index[0] for v in results.values())
]
bench_nav = bench_aligned / bench_aligned.iloc[0]
bench_returns = bench_aligned.pct_change().dropna()

rf_rate = st.session_state.get("rf_rate", 0.0)

tab_chart, tab_metrics, tab_holdings, tab_annual, tab_data = st.tabs(
    ["📊 Performance Chart", "📋 Metrics", "🔄 Holdings Changes", "📅 Annual Returns", "📁 Data & Export"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Performance chart
# ---------------------------------------------------------------------------

def _change_markers(nav_series: pd.Series, changes_df: pd.DataFrame, color: str, label: str):
    """Build a Scatter trace with diamond markers at every holding-change date."""
    if changes_df.empty:
        return None

    xs, ys, texts = [], [], []
    for _, row in changes_df.iterrows():
        dt = pd.Timestamp(row["Date"])
        # Snap to nearest available trading day in nav_series
        candidates = nav_series.index[nav_series.index >= dt]
        if len(candidates) == 0:
            continue
        dt = candidates[0]
        nav_val = nav_series[dt]

        added   = str(row.get("Added",   "—"))
        removed = str(row.get("Removed", "—"))
        holdings = str(row.get("Holdings", ""))

        lines = [f"<b>{label}</b>", f"Hält: {holdings}"]
        if added not in ("—", "nan", ""):
            lines.append(f"Neu: {added}")
        if removed not in ("—", "nan", ""):
            lines.append(f"Raus: {removed}")

        xs.append(dt)
        ys.append(nav_val)
        texts.append("<br>".join(lines))

    if not xs:
        return None

    return go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=11,
            color=color,
            line=dict(width=2, color="white"),
        ),
        name=f"{label} – Wechsel",
        text=texts,
        hovertemplate="%{text}<br><i>%{x|%d.%m.%Y}</i><extra></extra>",
        showlegend=True,
    )


with tab_chart:
    col_log, col_mkr = st.columns(2)
    with col_log:
        use_log = st.checkbox("Logarithmische Skala", value=False)
    with col_mkr:
        show_markers = st.checkbox("Wechsel-Marker anzeigen", value=True)

    fig = go.Figure()

    # Benchmark
    fig.add_trace(
        go.Scatter(
            x=bench_nav.index,
            y=bench_nav.values,
            name="S&P 500 (full index)",
            line=dict(color=BENCH_COLOR, width=2, dash="dot"),
            hovertemplate="S&P 500: %{y:.3f}<extra></extra>",
        )
    )

    for i, (label, (nav_series, changes_df)) in enumerate(results.items()):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]

        # Main NAV line
        fig.add_trace(
            go.Scatter(
                x=nav_series.index,
                y=nav_series.values,
                name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            )
        )

        # Holding-change markers (only when enabled)
        if show_markers:
            marker_trace = _change_markers(nav_series, changes_df, color, label)
            if marker_trace:
                fig.add_trace(marker_trace)

    fig.update_layout(
        title="Portfolio Value (normalisiert auf 1.0 am Start)",
        xaxis_title="Datum",
        yaxis_title="Portfolio-Wert",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=540,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if use_log:
        fig.update_yaxes(type="log", tickformat=".2g")
    else:
        fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=bench_returns.index,
            y=mc.drawdown_series(bench_returns).values * 100,
            name="S&P 500",
            fill="tozeroy",
            line=dict(color=BENCH_COLOR, width=1),
            fillcolor="rgba(173,181,189,0.15)",
            hovertemplate="S&P 500 DD: %{y:.1f}%<extra></extra>",
        )
    )
    for i, (label, (nav_series, changes_df)) in enumerate(results.items()):
        rets = nav_series.pct_change().dropna()
        dd = mc.drawdown_series(rets)
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        fig_dd.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                name=label,
                fill="tozeroy",
                line=dict(color=color, width=1),
                fillcolor=STRATEGY_FILLS[i % len(STRATEGY_FILLS)],
                hovertemplate=f"{label} DD: %{{y:.1f}}%<extra></extra>",
            )
        )
        # Holding-change markers on drawdown chart (same toggle)
        if show_markers and not changes_df.empty:
            dd_markers = _change_markers(
                dd * 100,
                changes_df, color, label,
            )
            if dd_markers:
                dd_markers.update(showlegend=False)
                fig_dd.add_trace(dd_markers)

    fig_dd.update_layout(
        title="Drawdown (%)",
        xaxis_title="Datum",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    for i, (label, (nav_series, _)) in enumerate(results.items()):
        rets = nav_series.pct_change().dropna()
        metrics_df = mc.calculate_metrics(rets, bench_returns, label=label, rf_annual=rf_rate)

        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        st.markdown(f"### {label}")
        st.dataframe(
            metrics_df.style
            .set_properties(**{"text-align": "right"})
            .highlight_max(axis=1, color="#1a4731", subset=[label, "S&P 500"])
            .set_table_styles(
                [{"selector": "th", "props": [("background-color", "#1e2130")]}]
            ),
            use_container_width=True,
        )
        st.divider()

# ---------------------------------------------------------------------------
# TAB 3 — Holdings Changes
# ---------------------------------------------------------------------------
with tab_holdings:
    for label, (nav_series, changes_df) in results.items():
        st.markdown(f"### {label} — Holding Changes")
        if changes_df.empty:
            st.info("No holding changes recorded (strategy may have stable holdings).")
        else:
            st.dataframe(changes_df, use_container_width=True, height=400)

            csv_changes = changes_df.to_csv(index=False).encode()
            st.download_button(
                f"Download {label} changes (CSV)",
                data=csv_changes,
                file_name=f"holdings_changes_{label.replace(' ', '_')}.csv",
                mime="text/csv",
            )
        st.divider()

# ---------------------------------------------------------------------------
# TAB 4 — Annual Returns
# ---------------------------------------------------------------------------
with tab_annual:
    # Build combined annual-returns table
    all_annual: dict[str, pd.Series] = {"S&P 500": mc.annual_returns(bench_returns)}
    for label, (nav_series, _) in results.items():
        rets = nav_series.pct_change().dropna()
        all_annual[label] = mc.annual_returns(rets)

    annual_df = pd.DataFrame(all_annual)
    annual_df.index.name = "Year"
    annual_df = annual_df.sort_index()

    # Bar chart
    fig_ann = go.Figure()
    years = annual_df.index.tolist()
    bar_width = 0.8 / len(annual_df.columns)

    for i, col in enumerate(annual_df.columns):
        color = BENCH_COLOR if col == "S&P 500" else STRATEGY_COLORS[(i - 1) % len(STRATEGY_COLORS)]
        offset = (i - len(annual_df.columns) / 2) * bar_width
        fig_ann.add_trace(
            go.Bar(
                x=[y + offset for y in range(len(years))],
                y=(annual_df[col] * 100).values,
                name=col,
                marker_color=color,
                width=bar_width * 0.9,
                text=(annual_df[col] * 100).round(1).astype(str) + "%",
                textposition="outside",
                textfont=dict(size=9),
            )
        )

    fig_ann.update_layout(
        title="Annual Returns (%)",
        xaxis=dict(
            tickvals=list(range(len(years))),
            ticktext=[str(y) for y in years],
            title="Year",
        ),
        yaxis_title="Return (%)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig_ann.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    st.plotly_chart(fig_ann, use_container_width=True)

    # Table
    display_annual = (annual_df * 100).round(2).astype(str) + "%"
    st.dataframe(display_annual.sort_index(ascending=False), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5 — Data & Export
# ---------------------------------------------------------------------------
with tab_data:
    st.subheader("Portfolio Values")

    # Build combined NAV table
    nav_dict: dict = {"S&P 500": bench_nav}
    for label, (nav_series, _) in results.items():
        nav_dict[label] = nav_series

    combined_nav = pd.DataFrame(nav_dict).ffill()
    combined_nav.index.name = "Date"
    combined_nav.index = combined_nav.index.date

    # Returns table
    returns_dict: dict = {"S&P 500": bench_returns}
    for label, (nav_series, _) in results.items():
        returns_dict[label] = nav_series.pct_change()
    combined_rets = pd.DataFrame(returns_dict).dropna(how="all")
    combined_rets.index.name = "Date"
    combined_rets.index = combined_rets.index.date

    st.dataframe(combined_nav.tail(252).style.format("{:.4f}"), use_container_width=True)

    # ---- CSV exports -------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        csv_nav = combined_nav.to_csv().encode()
        st.download_button(
            "Download NAV series (CSV)",
            data=csv_nav,
            file_name="portfolio_values.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        csv_rets = combined_rets.to_csv().encode()
        st.download_button(
            "Download daily returns (CSV)",
            data=csv_rets,
            file_name="daily_returns.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Combined metrics export
    st.subheader("Export All Metrics")
    all_metrics_rows = []
    for label, (nav_series, _) in results.items():
        rets = nav_series.pct_change().dropna()
        mdf = mc.calculate_metrics(rets, bench_returns, label=label, rf_annual=rf_rate)
        mdf.columns = [f"{col} ({label})" if col != "S&P 500" else col for col in mdf.columns]
        all_metrics_rows.append(mdf)

    if all_metrics_rows:
        combined_metrics = pd.concat(all_metrics_rows, axis=1)
        combined_metrics = combined_metrics.loc[:, ~combined_metrics.columns.duplicated()]
        csv_metrics = combined_metrics.to_csv().encode()
        st.download_button(
            "Download metrics (CSV)",
            data=csv_metrics,
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # All holdings changes combined export
    all_changes = []
    for label, (_, changes_df) in results.items():
        if not changes_df.empty:
            df_copy = changes_df.copy()
            df_copy.insert(0, "Strategy", label)
            all_changes.append(df_copy)

    if all_changes:
        st.subheader("All Holdings Changes")
        combined_changes = pd.concat(all_changes, ignore_index=True)
        csv_all_changes = combined_changes.to_csv(index=False).encode()
        st.download_button(
            "Download all holdings changes (CSV)",
            data=csv_all_changes,
            file_name="all_holdings_changes.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(combined_changes, use_container_width=True)
