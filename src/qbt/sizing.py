"""Position sizing helpers.

Strategies emit *target weights* (fraction of equity per asset). These helpers
turn raw signals into weights and apply portfolio-level risk overlays. All
functions are vectorized over the full history and only use information
available at each bar (rolling windows, no centered statistics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qbt.data import TRADING_DAYS


def equal_weight(signals: pd.DataFrame) -> pd.DataFrame:
    """Split capital equally across assets: ``weight = signal / n_assets``."""
    return signals / signals.shape[1]


def realized_vol(returns: pd.DataFrame, lookback: int = 60, min_periods: int | None = None):
    """Annualized rolling volatility of simple returns."""
    min_periods = min_periods or max(2, lookback // 2)
    return returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)


def inverse_vol_weights(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Naive risk parity: weights proportional to 1/vol, summing to 1."""
    vol = realized_vol(returns, lookback).replace(0.0, np.nan)
    inv = 1.0 / vol
    return inv.div(inv.sum(axis=1), axis=0)


def rolling_covariance(returns: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Rolling annualized covariance matrices, shape ``(T, N, N)``.

    Rows before ``lookback`` observations are filled with NaN.
    """
    values = returns.to_numpy(dtype=float)
    n_obs, n_assets = values.shape
    out = np.full((n_obs, n_assets, n_assets), np.nan)
    if n_obs < lookback:
        return out
    # Strided windows: (T - L + 1, L, N) without copying.
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
    windows = np.moveaxis(windows, -1, 1)  # (W, L, N)
    demeaned = windows - windows.mean(axis=1, keepdims=True)
    cov = np.einsum("wli,wlj->wij", demeaned, demeaned) / (lookback - 1)
    out[lookback - 1 :] = cov * TRADING_DAYS
    return out


def portfolio_vol(weights: pd.DataFrame, cov: np.ndarray) -> pd.Series:
    """Ex-ante annualized portfolio volatility ``sqrt(w' S w)`` per bar."""
    w = weights.to_numpy(dtype=float)
    var = np.einsum("ti,tij,tj->t", w, cov, w)
    return pd.Series(np.sqrt(np.clip(var, 0.0, None)), index=weights.index)


def cap_gross_leverage(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Scale rows down so that ``sum(|w|) <= max_gross``."""
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")
    gross = weights.abs().sum(axis=1)
    scale = (max_gross / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return weights.mul(scale, axis=0)


def vol_target(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target: float = 0.10,
    lookback: int = 60,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale a weight book so its ex-ante volatility matches ``target``.

    Uses the rolling covariance of asset returns, so diversification between
    positions is accounted for. Leverage is capped at ``max_leverage`` gross.
    Bars without enough history produce zero weights.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    cov = rolling_covariance(returns.fillna(0.0), lookback)
    vol = portfolio_vol(weights, cov)
    scale = (target / vol).replace([np.inf, -np.inf], np.nan)
    scaled = weights.mul(scale, axis=0).fillna(0.0)
    return cap_gross_leverage(scaled, max_leverage)
