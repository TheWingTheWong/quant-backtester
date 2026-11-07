"""Vectorized portfolio simulator.

The simulator takes a frame of *target weights* and turns it into a daily
return stream, tracking weight drift between rebalances, turnover, trading
costs, borrow fees and cash financing - without a Python loop over bars.

Timeline convention
-------------------
``target_weights.loc[t]`` is the book a strategy wants after observing the
close of bar ``t``. With ``delay=0`` the trade is filled at that same close
(market-on-close), so the position earns the return of bar ``t + 1`` onward.
``delay=k`` postpones execution by ``k`` bars, which is a cheap way to stress
test how sensitive a strategy is to execution latency.

How drift is handled without a loop
-----------------------------------
Bars are grouped into *holding periods* that start after each rebalance. If a
period starts with weights ``w`` (cash weight ``1 - sum(w)``), the value of
every sleeve at bar ``t`` relative to the period start is the cumulative
product of its gross returns within the period, ``G[t]``. The portfolio value
is ``g[t] = sum_i w_i G_i[t] + w_cash G_cash[t]``, the daily return is
``g[t] / g[t-1] - 1`` and the drifted weights are ``w * G[t] / g[t]``. A
grouped ``cumprod`` computes ``G`` for every period at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from qbt.costs import CostModel

if TYPE_CHECKING:  # pragma: no cover
    from qbt.strategies.base import Strategy

Rebalance = str | int

_PERIOD_ALIASES = {"W": "W", "M": "M", "Q": "Q", "Y": "Y", "A": "Y"}


@dataclass
class BacktestResult:
    """Everything produced by a simulation run."""

    returns: pd.Series
    """Net daily returns after all costs."""
    gross_returns: pd.Series
    """Daily returns before trading costs and borrow fees."""
    weights: pd.DataFrame
    """End-of-bar (post-trade) portfolio weights."""
    target_weights: pd.DataFrame
    """Target weights as emitted by the strategy (after overlays)."""
    turnover: pd.Series
    """One-way turnover per bar, as a fraction of equity."""
    costs: pd.Series
    """Trading costs plus borrow fees per bar, as a fraction of equity."""
    prices: pd.DataFrame
    initial_capital: float = 1_000_000.0
    name: str = "strategy"
    params: dict[str, Any] = field(default_factory=dict)
    benchmark: pd.Series | None = None
    """Optional benchmark return series aligned to ``returns``."""

    @property
    def equity(self) -> pd.Series:
        """Equity curve in currency units."""
        return self.initial_capital * (1.0 + self.returns).cumprod()

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    @property
    def net_exposure(self) -> pd.Series:
        return self.weights.sum(axis=1)

    @property
    def trades(self) -> pd.DataFrame:
        """Weight changes on bars where the book was traded."""
        traded = self.turnover > 0
        pre_trade = self.weights.shift(1).fillna(0.0)
        # Weight deltas are approximated against yesterday's post-trade book;
        # exact pre-trade (drifted) deltas are reflected in ``turnover``.
        return (self.weights - pre_trade).loc[traded]

    def metrics(self, risk_free: float = 0.0) -> dict[str, float]:
        from qbt.metrics import compute_metrics

        return compute_metrics(
            self.returns,
            turnover=self.turnover,
            exposure=self.gross_exposure,
            risk_free=risk_free,
        )


def rebalance_mask(index: pd.DatetimeIndex, rebalance: Rebalance = "D") -> np.ndarray:
    """Boolean array marking bars on which the portfolio is rebalanced.

    ``rebalance`` may be ``"D"`` (every bar), a calendar frequency
    (``"W"``, ``"M"``, ``"Q"``, ``"Y"`` - the last bar of each period), or an
    integer ``n`` (every ``n`` bars, starting with the first).
    """
    n = len(index)
    if isinstance(rebalance, (int, np.integer)):
        if rebalance < 1:
            raise ValueError("integer rebalance interval must be >= 1")
        return (np.arange(n) % int(rebalance)) == 0
    freq = str(rebalance).upper()
    if freq == "D":
        return np.ones(n, dtype=bool)
    if freq not in _PERIOD_ALIASES:
        raise ValueError(f"unknown rebalance frequency {rebalance!r}")
    periods = index.to_period(_PERIOD_ALIASES[freq])
    positions = pd.Series(np.arange(n), index=index)
    last = positions.groupby(periods).transform("max").to_numpy()
    return positions.to_numpy() == last


def simulate(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    costs: CostModel | None = None,
    rebalance: Rebalance = "D",
    delay: int = 0,
    initial_capital: float = 1_000_000.0,
    name: str = "strategy",
    params: dict[str, Any] | None = None,
) -> BacktestResult:
    """Simulate a portfolio that trades toward ``target_weights``.

    Args:
        prices: wide close-price frame.
        target_weights: desired weights per bar; NaN means "no position".
            Columns missing from ``prices`` are rejected; missing bars or
            assets are treated as zero weight.
        costs: cost model; defaults to :class:`CostModel` defaults.
        rebalance: how often to trade; see :func:`rebalance_mask`.
        delay: bars between signal and execution (``0`` = same close).
        initial_capital: starting equity, only used for the equity curve.
        name, params: metadata stored on the result.
    """
    costs = costs if costs is not None else CostModel()
    if delay < 0:
        raise ValueError("delay must be >= 0")
    unknown = set(target_weights.columns) - set(prices.columns)
    if unknown:
        raise ValueError(f"weights reference unknown assets: {sorted(unknown)}")
    if len(prices) < 2:
        raise ValueError("need at least two bars to simulate")

    index = prices.index
    assets = prices.columns
    targets = target_weights.reindex(index=index, columns=assets).shift(delay)
    # Assets cannot be held before they have a price.
    tradable = prices.notna()
    targets = targets.where(tradable, 0.0).fillna(0.0)

    asset_ret = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    n_bars, n_assets = asset_ret.shape

    rebal = rebalance_mask(index, rebalance)
    target_arr = targets.to_numpy(dtype=float)

    # Holding period of each bar's *return*: the trade at close s affects s+1..
    period = np.concatenate(([0], np.cumsum(rebal)[:-1]))
    period_weights = np.vstack([np.zeros((1, n_assets)), target_arr[rebal]])
    start_w = period_weights[period]  # (T, N) weights at the start of the period
    start_cash = 1.0 - start_w.sum(axis=1)

    growth_input = np.column_stack([1.0 + asset_ret, np.full(n_bars, 1.0 + costs.daily_cash_rate)])
    growth = pd.DataFrame(growth_input).groupby(period).cumprod().to_numpy()
    asset_growth, cash_growth = growth[:, :n_assets], growth[:, n_assets]

    value = (start_w * asset_growth).sum(axis=1) + start_cash * cash_growth
    period_start = np.concatenate(([True], period[1:] != period[:-1]))
    prev_value = np.where(period_start, 1.0, np.roll(value, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        gross = value / prev_value - 1.0
        drifted = np.where(value[:, None] != 0, start_w * asset_growth / value[:, None], 0.0)

    post_trade = np.where(rebal[:, None], target_arr, drifted)
    turnover = np.where(rebal, np.abs(target_arr - drifted).sum(axis=1), 0.0)

    held = np.vstack([np.zeros((1, n_assets)), post_trade[:-1]])  # book during bar t
    short_exposure = np.clip(-held, 0.0, None).sum(axis=1)

    trade_cost = turnover * costs.trade_cost_rate
    borrow = costs.borrow_cost(short_exposure)
    net = (1.0 + gross) * (1.0 - trade_cost) - 1.0 - borrow

    return BacktestResult(
        returns=pd.Series(net, index=index, name=name),
        gross_returns=pd.Series(gross, index=index, name=name),
        weights=pd.DataFrame(post_trade, index=index, columns=assets),
        target_weights=targets,
        turnover=pd.Series(turnover, index=index, name="turnover"),
        costs=pd.Series(trade_cost + borrow, index=index, name="costs"),
        prices=prices,
        initial_capital=initial_capital,
        name=name,
        params=dict(params or {}),
    )


def run_backtest(
    strategy: Strategy,
    prices: pd.DataFrame,
    costs: CostModel | None = None,
    rebalance: Rebalance = "D",
    delay: int = 0,
    vol_target: float | None = None,
    vol_lookback: int = 60,
    max_leverage: float | None = None,
    initial_capital: float = 1_000_000.0,
    benchmark: bool = True,
) -> BacktestResult:
    """Generate weights with ``strategy`` and simulate them.

    Optional overlays are applied to the strategy's weights before
    simulation: ``vol_target`` scales the book to an annualized volatility,
    ``max_leverage`` caps gross exposure. If ``benchmark`` is true the result
    carries an equal-weight buy-and-hold benchmark under the same costs.
    """
    from qbt import sizing

    weights = strategy.generate_weights(prices)
    if vol_target is not None:
        returns = prices.pct_change(fill_method=None)
        weights = sizing.vol_target(
            weights,
            returns,
            target=vol_target,
            lookback=vol_lookback,
            max_leverage=max_leverage if max_leverage is not None else 2.0,
        )
    elif max_leverage is not None:
        weights = sizing.cap_gross_leverage(weights.fillna(0.0), max_leverage)

    result = simulate(
        prices,
        weights,
        costs=costs,
        rebalance=rebalance,
        delay=delay,
        initial_capital=initial_capital,
        name=strategy.name,
        params=strategy.params,
    )
    if benchmark:
        result.benchmark = buy_and_hold(prices, costs=costs).returns.rename("benchmark")
    return result


def buy_and_hold(prices: pd.DataFrame, costs: CostModel | None = None) -> BacktestResult:
    """Equal-weight buy-and-hold: buy once on the first bar and let it drift."""
    first = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    available = prices.iloc[0].notna()
    first.iloc[0] = np.where(available, 1.0 / max(available.sum(), 1), 0.0)
    # Rebalancing only on bar 0 means the book drifts freely afterwards.
    mask_interval = len(prices) + 1
    return simulate(prices, first, costs=costs, rebalance=mask_interval, name="buy_and_hold")
