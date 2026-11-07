"""Trend-following strategies: moving-average crossover and time-series momentum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbt.sizing import cap_gross_leverage, equal_weight, realized_vol
from qbt.strategies.base import Strategy, register


@register("sma")
@dataclass
class SMACrossover(Strategy):
    """Long when the fast moving average is above the slow one.

    With ``allow_short`` the strategy flips short instead of going flat.
    Capital is split equally across assets.
    """

    title = "SMA crossover"
    description = "Moving-average crossover (fast SMA vs slow SMA)"

    fast: int = 20
    slow: int = 100
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.fast < 1 or self.slow < 2:
            raise ValueError("window lengths must be positive")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")

    @property
    def warmup(self) -> int:
        return self.slow

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(self.fast, min_periods=self.fast).mean()
        slow = prices.rolling(self.slow, min_periods=self.slow).mean()
        bearish = -1.0 if self.allow_short else 0.0
        signal = pd.DataFrame(
            np.where(fast > slow, 1.0, bearish), index=prices.index, columns=prices.columns
        )
        signal = signal.where(slow.notna())
        return equal_weight(signal)


@register("momentum")
@dataclass
class TimeSeriesMomentum(Strategy):
    """Time-series momentum in the spirit of Moskowitz, Ooi & Pedersen (2012).

    Each asset is held long (or short) according to the sign of its return
    over ``lookback`` bars, skipping the most recent ``skip`` bars to avoid
    short-term reversal. Positions are scaled so each asset contributes
    roughly ``asset_vol`` of annualized volatility, then the book is split
    equally across assets and capped at ``max_leverage`` gross.
    """

    title = "Time-series momentum"
    description = "Time-series momentum with per-asset volatility scaling"
    default_rebalance = "W"

    lookback: int = 252
    skip: int = 21
    vol_lookback: int = 60
    asset_vol: float = 0.15
    allow_short: bool = True
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.lookback <= self.skip:
            raise ValueError("lookback must be greater than skip")
        if self.skip < 0 or self.vol_lookback < 2:
            raise ValueError("skip must be >= 0 and vol_lookback >= 2")
        if self.asset_vol <= 0 or self.max_leverage <= 0:
            raise ValueError("asset_vol and max_leverage must be positive")

    @property
    def warmup(self) -> int:
        return max(self.lookback, self.vol_lookback)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        past = prices.shift(self.skip) / prices.shift(self.lookback) - 1.0
        signal = np.sign(past)
        if not self.allow_short:
            signal = signal.clip(lower=0.0)
        vol = realized_vol(prices.pct_change(fill_method=None), self.vol_lookback)
        scale = (self.asset_vol / vol).replace([np.inf, -np.inf], np.nan)
        weights = equal_weight(signal * scale)
        return cap_gross_leverage(weights.fillna(0.0), self.max_leverage).where(weights.notna())
