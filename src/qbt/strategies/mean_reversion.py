"""Mean reversion on a rolling z-score (Bollinger bands)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbt.sizing import equal_weight
from qbt.strategies.base import Strategy, register


def zscore(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Distance of price from its rolling mean in rolling standard deviations."""
    mean = prices.rolling(lookback, min_periods=lookback).mean()
    std = prices.rolling(lookback, min_periods=lookback).std()
    return (prices - mean) / std.where(std > 0)


@register("meanrev")
@dataclass
class BollingerMeanReversion(Strategy):
    """Fade moves outside the Bollinger bands, exit near the mean.

    * enter long when ``z < -entry_z`` (price below the lower band);
    * enter short when ``z > entry_z`` (if ``allow_short``);
    * exit to flat once ``|z| < exit_z``.

    The position between events is carried forward, which gives the classic
    hysteresis of a band strategy while staying fully vectorized: entries
    and exits become sparse events and a forward fill holds the state.
    """

    description = "Bollinger-band / z-score mean reversion with hysteresis"

    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")

    @property
    def warmup(self) -> int:
        return self.lookback

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        z = zscore(prices, self.lookback)
        events = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        events = events.mask(z.abs() < self.exit_z, 0.0)
        events = events.mask(z < -self.entry_z, 1.0)
        if self.allow_short:
            events = events.mask(z > self.entry_z, -1.0)
        # Before the first valid z-score there is no position (NaN); after it,
        # a missing event simply means "keep the current state".
        valid = z.notna().cummax()
        state = events.ffill().fillna(0.0).where(valid)
        return equal_weight(state)
