"""Parameter grid search and walk-forward analysis."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from qbt.costs import CostModel
from qbt.engine import Rebalance, run_backtest
from qbt.metrics import compute_metrics
from qbt.strategies.base import Strategy

ParamGrid = Mapping[str, Iterable[Any]]

# Metrics where a smaller value is better.
_MINIMIZE = {"ann_vol", "max_dd_duration", "turnover"}


def expand_grid(grid: ParamGrid) -> list[dict[str, Any]]:
    """Cartesian product of a ``{name: values}`` grid."""
    keys = list(grid)
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _build(strategy_cls: type[Strategy], params: dict[str, Any]) -> Strategy | None:
    """Instantiate, returning ``None`` for invalid combos (e.g. fast >= slow)."""
    try:
        return strategy_cls(**params)
    except ValueError:
        return None


def grid_search(
    strategy_cls: type[Strategy],
    prices: pd.DataFrame,
    grid: ParamGrid,
    metric: str = "sharpe",
    costs: CostModel | None = None,
    rebalance: Rebalance = "D",
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    **backtest_kwargs: Any,
) -> pd.DataFrame:
    """Backtest every valid parameter combination and rank by ``metric``.

    Weights are generated on the full ``prices`` history so indicators are
    warmed up, but only returns between ``start`` and ``end`` are scored.
    Combinations rejected by the strategy's validation are skipped.

    Returns:
        One row per combination: the parameters followed by all metrics,
        sorted best-first.
    """
    rows: list[dict[str, Any]] = []
    for params in expand_grid(grid):
        strategy = _build(strategy_cls, params)
        if strategy is None:
            continue
        result = run_backtest(
            strategy, prices, costs=costs, rebalance=rebalance, benchmark=False, **backtest_kwargs
        )
        window = slice(start, end)
        stats = compute_metrics(
            result.returns.loc[window],
            turnover=result.turnover.loc[window],
            exposure=result.gross_exposure.loc[window],
        )
        rows.append({**params, **stats})
    if not rows:
        raise ValueError("no valid parameter combinations in grid")
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise KeyError(f"unknown metric {metric!r}")
    ascending = metric in _MINIMIZE
    return table.sort_values(metric, ascending=ascending, kind="stable").reset_index(drop=True)


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict[str, Any]
    train_score: float
    test_score: float


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    returns: pd.Series
    """Stitched out-of-sample returns across all test windows."""
    turnover: pd.Series
    metric: str

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.returns, turnover=self.turnover)

    def folds_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "train": f"{f.train_start:%Y-%m-%d}..{f.train_end:%Y-%m-%d}",
                    "test": f"{f.test_start:%Y-%m-%d}..{f.test_end:%Y-%m-%d}",
                    **f.params,
                    f"train_{self.metric}": f.train_score,
                    f"test_{self.metric}": f.test_score,
                }
            )
        return pd.DataFrame(rows)


def walk_forward(
    strategy_cls: type[Strategy],
    prices: pd.DataFrame,
    grid: ParamGrid,
    train_bars: int = 756,
    test_bars: int = 252,
    metric: str = "sharpe",
    anchored: bool = False,
    costs: CostModel | None = None,
    rebalance: Rebalance = "D",
    progress: Callable[[int, int], None] | None = None,
    **backtest_kwargs: Any,
) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward optimization.

    For each fold the grid is searched on the training window, the winner is
    run on the following test window, and the test returns are stitched into
    one out-of-sample track record. Only data up to the end of each test
    window is visible to the backtest, so no fold can peek ahead.

    Args:
        train_bars: length of each in-sample window.
        test_bars: length of each out-of-sample window (also the step size).
        anchored: if true, training windows all start at the first bar.
    """
    if train_bars < 2 or test_bars < 1:
        raise ValueError("train_bars must be >= 2 and test_bars >= 1")
    n = len(prices)
    if n < train_bars + test_bars:
        raise ValueError(
            f"need at least {train_bars + test_bars} bars for one fold, got {n}"
        )

    index = prices.index
    folds: list[Fold] = []
    oos_returns: list[pd.Series] = []
    oos_turnover: list[pd.Series] = []
    starts = list(range(train_bars, n, test_bars))
    for i, test_start in enumerate(starts):
        test_stop = min(test_start + test_bars, n)
        train_first = 0 if anchored else test_start - train_bars
        train_idx = index[train_first:test_start]
        test_idx = index[test_start:test_stop]
        visible = prices.iloc[:test_stop]

        ranked = grid_search(
            strategy_cls,
            prices.iloc[:test_start],
            grid,
            metric=metric,
            costs=costs,
            rebalance=rebalance,
            start=train_idx[0],
            end=train_idx[-1],
            **backtest_kwargs,
        )
        # Read column-wise: a mixed-dtype row would upcast ints to floats.
        params = {k: _native(ranked[k].iloc[0]) for k in grid}
        strategy = strategy_cls(**params)
        result = run_backtest(
            strategy, visible, costs=costs, rebalance=rebalance, benchmark=False, **backtest_kwargs
        )
        test_ret = result.returns.loc[test_idx]
        test_stats = compute_metrics(test_ret)
        folds.append(
            Fold(
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                params=params,
                train_score=float(ranked[metric].iloc[0]),
                test_score=float(test_stats.get(metric, float("nan"))),
            )
        )
        oos_returns.append(test_ret)
        oos_turnover.append(result.turnover.loc[test_idx])
        if progress is not None:
            progress(i + 1, len(starts))

    return WalkForwardResult(
        folds=folds,
        returns=pd.concat(oos_returns).rename("walk_forward"),
        turnover=pd.concat(oos_turnover),
        metric=metric,
    )


def _native(value: Any) -> Any:
    """Convert numpy scalars from a DataFrame row back to Python types."""
    return value.item() if hasattr(value, "item") else value
