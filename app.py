"""
S&P N Backtest App
==================
Compare holding the top-N largest S&P 500 stocks (by market cap) versus
the full S&P 500 index.

Strategies supported:
  • S&P 1  – hold only the single largest stock
  • S&P N  – hold the N largest stocks, weighted by market cap

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

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
# Colour palette (dynamic, turbo colorscale)
# ---------------------------------------------------------------------------
BENCH_COLOR = "#adb5bd"


def _strategy_color(i: int, n: int) -> str:
    """Return color for strategy i (0-indexed) out of n total strategies."""
    frac = i / max(n - 1, 1)
    return pc.sample_colorscale("turbo", [frac])[0]


def _strategy_fill(i: int, n: int) -> str:
    color = _strategy_color(i, n)
    return color.replace("rgb(", "rgba(").replace(")", ",0.15)")


# ---------------------------------------------------------------------------
# Monte Carlo function
# ---------------------------------------------------------------------------
def _run_monte_carlo(results, bench_nav, period_days, n_sims):
    """
    Sample n_sims non-overlapping random windows of period_days trading days
    from each strategy's pre-computed NAV series.
    Returns dict: {label: list of annualized_period_returns}
    """
    rng = np.random.default_rng(42)

    # Build nav_dict including benchmark
    nav_dict = {"S&P 500": bench_nav}
    for label, (nav, _) in results.items():
        nav_dict[label] = nav

    # Common dates across all series
    common_idx = nav_dict[next(iter(nav_dict))].index
    for nav in nav_dict.values():
        common_idx = common_idx.intersection(nav.index)
    common_idx = common_idx.sort_values()
    n_dates = len(common_idx)

    if n_dates <= period_days:
        return {}

    # Sample start positions (with replacement to allow more sims than windows)
    max_start = n_dates - period_days
    starts = rng.choice(max_start, size=min(n_sims, max_start * 2), replace=True)
    # Deduplicate and take first n_sims
    starts = list(dict.fromkeys(int(s) for s in starts))[:n_sims]

    mc_results = {label: [] for label in nav_dict}
    for start_i in starts:
        end_i = start_i + period_days
        t_start = common_idx[start_i]
        t_end = common_idx[end_i]
        for label, nav in nav_dict.items():
            v0 = nav.loc[t_start]
            v1 = nav.loc[t_end]
            period_ret = v1 / v0 - 1
            ann_ret = (1 + period_ret) ** (252 / period_days) - 1
            mc_results[label].append(ann_ret)

    return mc_results


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
    max_n = st.slider("Strategien: S&P 1 bis S&P N", min_value=1, max_value=20, value=10)

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
# Marker helper (unchanged)
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

    # Build strategies from max_n slider
    strategies = [{"rank_min": 1, "rank_max": n, "label": f"S&P {n}"} for n in range(1, max_n + 1)]

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
    st.session_state["max_n"] = max_n
    # Clear stale Monte Carlo results when backtest is re-run
    st.session_state.pop("mc_results", None)
    st.session_state.pop("mc_period_label", None)
    st.session_state.pop("mc_n_sims", None)

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
max_n = st.session_state.get("max_n", 10)

tab_chart, tab_overview, tab_mc, tab_holdings, tab_annual, tab_metrics, tab_data = st.tabs([
    "📊 Performance", "🎯 Übersicht", "🎲 Monte Carlo",
    "🔄 Holdings", "📅 Jährliche Renditen", "📋 Metriken", "📁 Export"
])

# ---------------------------------------------------------------------------
# TAB 1 — Performance chart
# ---------------------------------------------------------------------------
with tab_chart:
    col_log, col_mkr = st.columns(2)
    with col_log:
        use_log = st.checkbox("Logarithmische Skala", value=False)
    with col_mkr:
        show_markers = st.checkbox("Wechsel-Marker anzeigen", value=True)

    n_results = len(results)
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
        color = _strategy_color(i, n_results)

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
        color = _strategy_color(i, n_results)
        fill = _strategy_fill(i, n_results)
        fig_dd.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                name=label,
                fill="tozeroy",
                line=dict(color=color, width=1),
                fillcolor=fill,
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
# TAB 2 — Übersicht (new)
# ---------------------------------------------------------------------------
with tab_overview:
    n_results = len(results)

    # Compute per-strategy stats
    strat_stats = {}
    for i, (label, (nav_series, _)) in enumerate(results.items()):
        rets = nav_series.pct_change().dropna()
        ann_ret = (1 + rets.mean()) ** 252 - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = mc.sharpe(rets, rf_rate)
        max_dd = mc.drawdown_series(rets).min()
        strat_stats[label] = {
            "color": _strategy_color(i, n_results),
            "index": i,
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_dd": max_dd,
        }

    # Benchmark stats
    bench_ann_ret = (1 + bench_returns.mean()) ** 252 - 1
    bench_ann_vol = bench_returns.std() * np.sqrt(252)
    bench_sharpe = mc.sharpe(bench_returns, rf_rate)
    bench_max_dd = mc.drawdown_series(bench_returns).min()

    col_scatter, col_sharpe = st.columns(2)

    with col_scatter:
        fig_scatter = go.Figure()

        # Strategy scatter points
        for label, stats in strat_stats.items():
            fig_scatter.add_trace(
                go.Scatter(
                    x=[stats["ann_vol"]],
                    y=[stats["ann_ret"]],
                    mode="markers+text",
                    name=label,
                    text=[label],
                    textposition="top center",
                    marker=dict(
                        color=stats["color"],
                        size=14,
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        f"Rendite: {stats['ann_ret']:.1%}<br>"
                        f"Volatilität: {stats['ann_vol']:.1%}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        # Benchmark point
        fig_scatter.add_trace(
            go.Scatter(
                x=[bench_ann_vol],
                y=[bench_ann_ret],
                mode="markers+text",
                name="S&P 500",
                text=["S&P 500"],
                textposition="top center",
                marker=dict(
                    color=BENCH_COLOR,
                    size=14,
                    symbol="diamond",
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=(
                    f"<b>S&P 500</b><br>"
                    f"Rendite: {bench_ann_ret:.1%}<br>"
                    f"Volatilität: {bench_ann_vol:.1%}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # Dashed reference line from origin through benchmark
        slope = bench_ann_ret / bench_ann_vol if bench_ann_vol != 0 else 0
        x_max = max(
            [s["ann_vol"] for s in strat_stats.values()] + [bench_ann_vol]
        ) * 1.3
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, x_max],
                y=[0, slope * x_max],
                mode="lines",
                line=dict(color=BENCH_COLOR, dash="dash", width=1),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_scatter.update_layout(
            title="Risiko vs. Rendite (annualisiert)",
            xaxis_title="Volatilität (ann.)",
            yaxis_title="Rendite (ann.)",
            height=420,
            template="plotly_dark",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_sharpe:
        # Sharpe ratio horizontal bar chart (all strategies + benchmark, sorted desc)
        sharpe_data = {label: stats["sharpe"] for label, stats in strat_stats.items()}
        sharpe_data["S&P 500"] = bench_sharpe

        sharpe_sorted = sorted(sharpe_data.items(), key=lambda x: x[1], reverse=True)
        sharpe_labels = [item[0] for item in sharpe_sorted]
        sharpe_values = [item[1] for item in sharpe_sorted]

        bar_colors = []
        for lbl in sharpe_labels:
            if lbl == "S&P 500":
                bar_colors.append(BENCH_COLOR)
            else:
                bar_colors.append(strat_stats[lbl]["color"])

        fig_sharpe = go.Figure(
            go.Bar(
                x=sharpe_values,
                y=sharpe_labels,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.2f}" for v in sharpe_values],
                textposition="outside",
            )
        )
        fig_sharpe.update_layout(
            title="Sharpe Ratio",
            xaxis_title="Sharpe Ratio",
            height=420,
            template="plotly_dark",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # Summary metrics table
    st.subheader("Kennzahlen-Übersicht")
    summary_cols = list(strat_stats.keys()) + ["S&P 500"]
    summary_data = {
        "Rendite (ann.)": [
            f"{strat_stats[lbl]['ann_ret']:.1%}" if lbl != "S&P 500" else f"{bench_ann_ret:.1%}"
            for lbl in summary_cols
        ],
        "Volatilität (ann.)": [
            f"{strat_stats[lbl]['ann_vol']:.1%}" if lbl != "S&P 500" else f"{bench_ann_vol:.1%}"
            for lbl in summary_cols
        ],
        "Sharpe": [
            f"{strat_stats[lbl]['sharpe']:.2f}" if lbl != "S&P 500" else f"{bench_sharpe:.2f}"
            for lbl in summary_cols
        ],
        "Max Drawdown": [
            f"{strat_stats[lbl]['max_dd']:.1%}" if lbl != "S&P 500" else f"{bench_max_dd:.1%}"
            for lbl in summary_cols
        ],
    }
    summary_df = pd.DataFrame(summary_data, index=summary_cols).T
    summary_df.index.name = "Kennzahl"
    st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Monte Carlo (new)
# ---------------------------------------------------------------------------
with tab_mc:
    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        period_label = st.selectbox(
            "Simulationszeitraum",
            ["6 Monate", "1 Jahr", "2 Jahre", "3 Jahre", "5 Jahre", "10 Jahre"],
            index=1,
        )
    with col_mc2:
        n_sims = st.slider("Anzahl Simulationen", 50, 500, 200, step=50)
    with col_mc3:
        mc_run = st.button("Monte Carlo starten", type="primary", use_container_width=True)

    period_days_map = {
        "6 Monate": 126, "1 Jahr": 252, "2 Jahre": 504,
        "3 Jahre": 756, "5 Jahre": 1260, "10 Jahre": 2520,
    }
    period_days = period_days_map[period_label]

    if mc_run or "mc_results" in st.session_state:
        if mc_run:
            with st.spinner("Monte Carlo läuft..."):
                mc_data = _run_monte_carlo(results, bench_nav, period_days, n_sims)
                st.session_state["mc_results"] = mc_data
                st.session_state["mc_period_label"] = period_label
                st.session_state["mc_n_sims"] = len(next(iter(mc_data.values()), []))
        else:
            mc_data = st.session_state.get("mc_results", {})

        if not mc_data:
            st.warning(
                "Nicht genug historische Daten für den gewählten Simulationszeitraum. "
                "Bitte einen kürzeren Zeitraum wählen oder den Backtest-Zeitraum verlängern."
            )
        else:
            actual_n_sims = st.session_state.get("mc_n_sims", n_sims)
            stored_period = st.session_state.get("mc_period_label", period_label)

            # Box plot of annualized returns
            fig_mc = go.Figure()
            ordered_labels = (
                [f"S&P {n}" for n in range(1, max_n + 1) if f"S&P {n}" in mc_data]
                + ["S&P 500"]
            )
            strategy_labels_ordered = [lbl for lbl in ordered_labels if lbl != "S&P 500"]

            for idx, label in enumerate(ordered_labels):
                vals = mc_data.get(label, [])
                if not vals:
                    continue
                if label == "S&P 500":
                    color = BENCH_COLOR
                else:
                    strat_idx = strategy_labels_ordered.index(label)
                    color = _strategy_color(strat_idx, len(strategy_labels_ordered))
                fig_mc.add_trace(
                    go.Box(
                        y=[v * 100 for v in vals],
                        name=label,
                        marker_color=color,
                        boxpoints="outliers",
                        line_width=2,
                    )
                )

            fig_mc.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
            fig_mc.update_layout(
                title=f"Renditeverteilung – {actual_n_sims} zufällige {stored_period}-Zeiträume",
                yaxis_title="Annualisierte Rendite (%)",
                height=450,
                template="plotly_dark",
                showlegend=False,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Win rate bar chart
            if "S&P 500" in mc_data:
                bench_vals = mc_data["S&P 500"]
                win_rates = {}
                win_rate_labels = [
                    f"S&P {n}" for n in range(1, max_n + 1) if f"S&P {n}" in mc_data
                ]
                for label in win_rate_labels:
                    strat_vals = mc_data[label]
                    win_rate = np.mean([s > b for s, b in zip(strat_vals, bench_vals)])
                    win_rates[label] = win_rate * 100

                if win_rates:
                    fig_win = go.Figure(
                        go.Bar(
                            x=list(win_rates.keys()),
                            y=list(win_rates.values()),
                            marker_color=[
                                _strategy_color(i, len(win_rates))
                                for i in range(len(win_rates))
                            ],
                            text=[f"{v:.0f}%" for v in win_rates.values()],
                            textposition="outside",
                        )
                    )
                    fig_win.add_hline(
                        y=50, line_dash="dash", line_color="white", opacity=0.5,
                        annotation_text="50%", annotation_position="right",
                    )
                    fig_win.update_layout(
                        title="Outperformance-Rate vs. S&P 500 (%)",
                        yaxis=dict(title="% der Simulationen", range=[0, 105]),
                        height=350,
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_win, use_container_width=True)

            # Summary stats table
            st.subheader("Monte Carlo Kennzahlen")
            mc_summary_rows = []
            for label in ordered_labels:
                vals = mc_data.get(label, [])
                if not vals:
                    continue
                vals_arr = np.array(vals)
                bench_arr = np.array(mc_data.get("S&P 500", []))
                win_rate_val = (
                    float(np.mean(vals_arr > bench_arr)) * 100
                    if len(bench_arr) == len(vals_arr) and label != "S&P 500"
                    else float("nan")
                )
                mc_summary_rows.append(
                    {
                        "Strategie": label,
                        "Median Rendite": f"{np.median(vals_arr) * 100:.1f}%",
                        "5. Perzentile": f"{np.percentile(vals_arr, 5) * 100:.1f}%",
                        "95. Perzentile": f"{np.percentile(vals_arr, 95) * 100:.1f}%",
                        "Outperformance vs. S&P 500": (
                            f"{win_rate_val:.0f}%" if not np.isnan(win_rate_val) else "—"
                        ),
                    }
                )
            if mc_summary_rows:
                mc_summary_df = pd.DataFrame(mc_summary_rows).set_index("Strategie")
                st.dataframe(mc_summary_df, use_container_width=True)
    else:
        st.info("Klicke **Monte Carlo starten**, um die Simulation zu starten.")

# ---------------------------------------------------------------------------
# TAB 4 — Holdings Changes
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
# TAB 5 — Annual Returns
# ---------------------------------------------------------------------------
with tab_annual:
    n_results = len(results)

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
        if col == "S&P 500":
            color = BENCH_COLOR
        else:
            # i starts at 0 for S&P 500, so strategy index is i-1
            strat_idx = i - 1
            color = _strategy_color(strat_idx, n_results)
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
# TAB 6 — Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    n_results = len(results)
    for i, (label, (nav_series, _)) in enumerate(results.items()):
        rets = nav_series.pct_change().dropna()
        metrics_df = mc.calculate_metrics(rets, bench_returns, label=label, rf_annual=rf_rate)

        color = _strategy_color(i, n_results)
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
# TAB 7 — Data & Export
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
