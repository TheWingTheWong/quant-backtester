"""Compare every built-in strategy on the same synthetic market.

Run with:  uv run python examples/compare_strategies.py
"""

from __future__ import annotations

import pandas as pd

from qbt import CostModel, generate_prices, run_backtest
from qbt.strategies import REGISTRY


def main() -> None:
    prices = generate_prices(n_assets=4, n_days=252 * 10, seed=42)
    costs = CostModel(commission_bps=1, slippage_bps=2)

    rows = {}
    for _, cls in sorted(REGISTRY.items()):
        result = run_backtest(cls(), prices, costs=costs, rebalance=cls.default_rebalance)
        stats = result.metrics()
        rows[cls.title] = {
            "CAGR": f"{stats['cagr']:.2%}",
            "Vol": f"{stats['ann_vol']:.2%}",
            "Sharpe": f"{stats['sharpe']:.2f}",
            "Max DD": f"{stats['max_drawdown']:.2%}",
            "Turnover": f"{stats['turnover']:.1f}x",
        }

    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
