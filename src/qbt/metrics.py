"""Performance and risk metrics for daily return series.

All functions accept simple (not log) daily returns and assume
``TRADING_DAYS`` bars per year. They are pure and vectorized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbt.data import TRADING_DAYS


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + _clean(returns)).cumprod()


def total_return(returns: pd.Series) -> float:
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else 0.0


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    if len(r) == 0:
        return 0.0
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio of excess returns (``risk_free`` is annual)."""
    r = _clean(returns) - risk_free / periods_per_year
    sd = r.std(ddof=1)
    if len(r) < 2 or not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Sortino ratio using downside deviation below ``risk_free``."""
    r = _clean(returns) - risk_free / periods_per_year
    if len(r) < 2:
        return 0.0
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    if downside < 1e-12:
        return 0.0
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage drawdown from the running peak (values are <= 0)."""
    equity = equity_curve(returns)
    peak = equity.cummax().clip(lower=1.0)  # starting capital counts as a peak
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch of bars spent below a previous equity peak."""
    underwater = (drawdown_series(returns) < 0).to_numpy()
    if not underwater.any():
        return 0
    # Length of each run of consecutive True values.
    padded = np.concatenate(([False], underwater, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]
    return int((ends - starts).max())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    return float(cagr(returns, periods_per_year) / abs(mdd)) if mdd < 0 else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Share of *active* bars (non-zero return) that were profitable."""
    r = _clean(returns)
    active = r[r != 0]
    return float((active > 0).mean()) if len(active) else 0.0


def profit_factor(returns: pd.Series) -> float:
    r = _clean(returns)
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else float("inf")


@dataclass(frozen=True)
class Drawdown:
    start: pd.Timestamp
    trough: pd.Timestamp
    end: pd.Timestamp | None
    depth: float
    length: int
    """Bars from the prior peak until recovery (or the last bar)."""


def top_drawdowns(returns: pd.Series, n: int = 5) -> list[Drawdown]:
    """The ``n`` deepest drawdown episodes, deepest first."""
    dd = drawdown_series(returns)
    if dd.empty:
        return []
    underwater = (dd < 0).to_numpy()
    padded = np.concatenate(([False], underwater, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    episodes: list[Drawdown] = []
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        window = dd.iloc[start:stop]
        recovered = stop < len(dd)
        episodes.append(
            Drawdown(
                start=dd.index[start - 1] if start > 0 else dd.index[start],
                trough=window.idxmin(),
                end=dd.index[stop] if recovered else None,
                depth=float(window.min()),
                length=int(stop - start),
            )
        )
    episodes.sort(key=lambda d: d.depth)
    return episodes[:n]


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded returns pivoted to a ``year x month`` table (1..12)."""
    r = _clean(returns)
    if r.empty:
        return pd.DataFrame(columns=range(1, 13), dtype=float)
    monthly = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    table = monthly.unstack()
    table.index.name, table.columns.name = "year", "month"
    return table.reindex(columns=range(1, 13))


def annual_returns(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    return (1.0 + r).groupby(r.index.year).prod() - 1.0


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    r = _clean(returns)
    mean = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return (mean / sd.where(sd > 1e-12)) * np.sqrt(TRADING_DAYS)


def compute_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Headline statistics for a daily return series.

    ``turnover`` (one-way, fraction of equity per bar) is annualized by
    summing per year; ``exposure`` is gross exposure per bar.
    """
    r = _clean(returns)
    out: dict[str, float] = {
        "total_return": total_return(r),
        "cagr": cagr(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r, risk_free),
        "sortino": sortino_ratio(r, risk_free),
        "calmar": calmar_ratio(r),
        "max_drawdown": max_drawdown(r),
        "max_dd_duration": float(max_drawdown_duration(r)),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
        "skew": float(r.skew()) if len(r) > 2 else 0.0,
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
    }
    if turnover is not None:
        out["turnover"] = float(_clean(turnover).mean() * TRADING_DAYS)
    if exposure is not None:
        out["avg_exposure"] = float(_clean(exposure).mean())
    return out


# Display metadata: label and format for each metric key.
METRIC_FORMATS: dict[str, tuple[str, str]] = {
    "total_return": ("Total return", "pct"),
    "cagr": ("CAGR", "pct"),
    "ann_vol": ("Annualized vol", "pct"),
    "sharpe": ("Sharpe", "ratio"),
    "sortino": ("Sortino", "ratio"),
    "calmar": ("Calmar", "ratio"),
    "max_drawdown": ("Max drawdown", "pct"),
    "max_dd_duration": ("Max DD duration (days)", "int"),
    "hit_rate": ("Hit rate", "pct"),
    "profit_factor": ("Profit factor", "ratio"),
    "skew": ("Skew", "ratio"),
    "best_day": ("Best day", "pct"),
    "worst_day": ("Worst day", "pct"),
    "turnover": ("Turnover (x/yr)", "ratio"),
    "avg_exposure": ("Avg gross exposure", "ratio"),
}


def format_metric(key: str, value: float) -> str:
    kind = METRIC_FORMATS.get(key, (key, "ratio"))[1]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if np.isinf(value):
        return "inf"
    if kind == "pct":
        text = f"{value * 100:.2f}%"
    elif kind == "int":
        text = f"{int(round(value))}"
    else:
        text = f"{value:.2f}"
    return text[1:] if text.startswith("-") and not any(ch in "123456789" for ch in text) else text
