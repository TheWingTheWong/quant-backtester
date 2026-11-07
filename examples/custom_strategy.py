"""Write a strategy in a dozen lines and backtest it.

A strategy is a dataclass of parameters plus ``generate_weights``, a pure
function from prices to target weights. Row ``t`` may only use data up to
bar ``t``; the engine handles execution timing, drift and costs.

Run with:  uv run python examples/custom_strategy.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qbt import generate_prices, run_backtest
from qbt.strategies import Strategy


@dataclass
class DualMomentum(Strategy):
    """Hold the single asset with the best trailing return, if it is positive."""

    lookback: int = 126

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        trailing = prices.pct_change(self.lookback, fill_method=None)
        best = trailing.eq(trailing.max(axis=1), axis=0) & (trailing > 0)
        weights = best.astype(float)
        return weights.where(trailing.notna().all(axis=1), np.nan, axis=0)


def main() -> None:
    prices = generate_prices(n_assets=4, seed=42)
    result = run_backtest(DualMomentum(lookback=126), prices, rebalance="M")
    stats = result.metrics()
    print(
        f"{result.name}: CAGR {stats['cagr']:.2%}, Sharpe {stats['sharpe']:.2f}, "
        f"max drawdown {stats['max_drawdown']:.2%}, turnover {stats['turnover']:.1f}x/yr"
    )


if __name__ == "__main__":
    main()
