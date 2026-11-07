"""Static equal-weight allocation, useful as a baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qbt.strategies.base import Strategy, register


@register("equal")
@dataclass
class EqualWeight(Strategy):
    """Hold every asset at ``1 / n`` of equity, rebalanced by the simulator."""

    title = "Equal weight"
    description = "Equal-weight allocation (baseline)"

    leverage: float = 1.0

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        weight = self.leverage / prices.shape[1]
        weights = pd.DataFrame(weight, index=prices.index, columns=prices.columns)
        return weights.where(prices.notna())
