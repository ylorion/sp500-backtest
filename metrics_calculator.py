"""
Performance metrics for the S&P N backtest app.
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252  # trading days


def _to_returns(prices_or_returns: pd.Series) -> pd.Series:
    """Accept either a price series or a return series."""
    if (prices_or_returns.dropna() > 1).all():
        return prices_or_returns.pct_change().dropna()
    return prices_or_returns.dropna()


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    roll_max = cum.expanding().max()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    cum = (1 + returns).cumprod()
    roll_max = cum.expanding().max()
    return (cum - roll_max) / roll_max


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_daily = rf_annual / PERIODS_PER_YEAR
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / std)


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    rf_daily = rf_annual / PERIODS_PER_YEAR
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    # Daily downside semi-deviation (no annualization here — numerator carries sqrt(N))
    daily_downside_std = np.sqrt((downside ** 2).mean())
    if daily_downside_std == 0:
        return np.nan
    return float(np.sqrt(PERIODS_PER_YEAR) * excess.mean() / daily_downside_std)


def beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([returns, bench_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    bench_var = cov[1, 1]
    if bench_var == 0:
        return np.nan
    return float(cov[0, 1] / bench_var)


def alpha(
    returns: pd.Series,
    bench_returns: pd.Series,
    beta_val: float,
    rf_annual: float = 0.0,
) -> float:
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    years = n / PERIODS_PER_YEAR
    port_ann = (1 + returns.mean()) ** PERIODS_PER_YEAR - 1
    bench_ann = (1 + bench_returns.mean()) ** PERIODS_PER_YEAR - 1
    return float(port_ann - (rf_annual + beta_val * (bench_ann - rf_annual)))


def calmar(returns: pd.Series) -> float:
    ann_ret = (1 + returns.mean()) ** PERIODS_PER_YEAR - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(ann_ret / mdd)


def information_ratio(returns: pd.Series, bench_returns: pd.Series) -> float:
    aligned = pd.concat([returns, bench_returns], axis=1).dropna()
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    std = active.std()
    if std == 0:
        return np.nan
    return float(np.sqrt(PERIODS_PER_YEAR) * active.mean() / std)


def annual_returns(returns: pd.Series) -> pd.Series:
    """Compound annual returns grouped by calendar year."""
    return returns.groupby(returns.index.year).apply(
        lambda r: (1 + r).prod() - 1
    )


def calculate_metrics(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    label: str = "Strategy",
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Return a tidy DataFrame comparing strategy vs benchmark metrics."""
    # Align
    df = pd.concat(
        [port_returns.rename("port"), bench_returns.rename("bench")], axis=1
    ).dropna()
    p, b = df["port"], df["bench"]

    n_days = len(p)
    years = n_days / PERIODS_PER_YEAR

    total_p = (1 + p).prod() - 1
    total_b = (1 + b).prod() - 1
    ann_p = (1 + total_p) ** (1 / years) - 1 if years > 0 else np.nan
    ann_b = (1 + total_b) ** (1 / years) - 1 if years > 0 else np.nan
    vol_p = p.std() * np.sqrt(PERIODS_PER_YEAR)
    vol_b = b.std() * np.sqrt(PERIODS_PER_YEAR)
    b_val = beta(p, b)

    rows = {
        "Total Return": (f"{total_p:.2%}", f"{total_b:.2%}"),
        "Annualized Return": (f"{ann_p:.2%}", f"{ann_b:.2%}"),
        "Annualized Volatility": (f"{vol_p:.2%}", f"{vol_b:.2%}"),
        "Sharpe Ratio": (
            f"{sharpe(p, rf_annual):.2f}",
            f"{sharpe(b, rf_annual):.2f}",
        ),
        "Sortino Ratio": (
            f"{sortino(p, rf_annual):.2f}",
            f"{sortino(b, rf_annual):.2f}",
        ),
        "Max Drawdown": (
            f"{max_drawdown(p):.2%}",
            f"{max_drawdown(b):.2%}",
        ),
        "Beta (vs S&P 500)": (f"{b_val:.3f}", "1.000"),
        "Alpha (annualized)": (
            f"{alpha(p, b, b_val, rf_annual):.2%}",
            "0.00%",
        ),
        "Calmar Ratio": (f"{calmar(p):.2f}", f"{calmar(b):.2f}"),
        "Information Ratio": (f"{information_ratio(p, b):.2f}", "—"),
        "Trading Days": (f"{n_days:,}", f"{n_days:,}"),
    }

    result = pd.DataFrame(rows, index=[label, "S&P 500"]).T
    result.index.name = "Metric"
    return result
