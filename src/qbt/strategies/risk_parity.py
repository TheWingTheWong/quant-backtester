"""Volatility-targeted risk parity across a basket of assets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qbt.sizing import inverse_vol_weights, vol_target
from qbt.strategies.base import Strategy, register


@register("riskparity")
@dataclass
class VolTargetRiskParity(Strategy):
    """Inverse-volatility weights scaled to a portfolio volatility target.

    Step 1 allocates proportionally to ``1 / sigma_i`` so every asset carries
    a similar amount of stand-alone risk. Step 2 estimates the portfolio's
    ex-ante volatility from the rolling covariance matrix (so correlations
    count) and levers the whole book up or down to ``target_vol``, capped at
    ``max_leverage`` gross. Best paired with weekly or monthly rebalancing.
    """

    title = "Risk parity, vol-targeted"
    description = "Inverse-vol risk parity with a portfolio volatility target"

    vol_lookback: int = 60
    target_vol: float = 0.10
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.vol_lookback < 5:
            raise ValueError("vol_lookback must be >= 5")
        if self.target_vol <= 0 or self.max_leverage <= 0:
            raise ValueError("target_vol and max_leverage must be positive")

    @property
    def warmup(self) -> int:
        return self.vol_lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change(fill_method=None)
        base = inverse_vol_weights(returns, self.vol_lookback)
        scaled = vol_target(
            base.fillna(0.0),
            returns,
            target=self.target_vol,
            lookback=self.vol_lookback,
            max_leverage=self.max_leverage,
        )
        return scaled.where(base.notna())
