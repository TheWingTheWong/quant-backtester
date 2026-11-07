import numpy as np
import pandas as pd
import pytest

from qbt.costs import CostModel
from qbt.data import TRADING_DAYS
from qbt.engine import buy_and_hold, rebalance_mask, simulate


def reference_simulation(prices, targets, rebal, cost_rate=0.0):
    """Straightforward bar-by-bar loop used as ground truth for the engine."""
    rets = prices.pct_change().fillna(0.0).to_numpy()
    w_target = targets.reindex(prices.index).fillna(0.0).to_numpy()
    n, k = rets.shape
    weights = np.zeros(k)  # post-trade weights at previous close
    out = np.zeros(n)
    for t in range(n):
        # earn the bar's return on the book held since the last close
        gross_assets = weights * (1 + rets[t])
        cash = 1.0 - weights.sum()
        value = gross_assets.sum() + cash
        r = value - 1.0
        drifted = gross_assets / value
        if rebal[t]:
            turnover = np.abs(w_target[t] - drifted).sum()
            r = (1 + r) * (1 - turnover * cost_rate) - 1
            weights = w_target[t].copy()
        else:
            weights = drifted
        out[t] = r
    return out


def test_fully_invested_single_asset_tracks_price(single_asset):
    weights = pd.DataFrame(1.0, index=single_asset.index, columns=["A"])
    result = simulate(single_asset, weights, costs=CostModel.zero(), initial_capital=1.0)
    expected = single_asset["A"] / single_asset["A"].iloc[0]
    np.testing.assert_allclose(result.equity.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_no_lookahead_trade_at_close_earns_next_bar(single_asset):
    weights = pd.DataFrame(np.nan, index=single_asset.index, columns=["A"])
    weights.iloc[100:] = 1.0  # decide at close of bar 100
    result = simulate(single_asset, weights, costs=CostModel.zero())
    asset_ret = single_asset["A"].pct_change()
    assert (result.returns.iloc[:101] == 0).all()
    assert result.returns.iloc[101] == pytest.approx(asset_ret.iloc[101])


def test_delay_postpones_execution(single_asset):
    weights = pd.DataFrame(1.0, index=single_asset.index, columns=["A"])
    fast = simulate(single_asset, weights, costs=CostModel.zero(), delay=0)
    slow = simulate(single_asset, weights, costs=CostModel.zero(), delay=3)
    assert slow.returns.iloc[:4].abs().sum() == 0
    np.testing.assert_allclose(slow.returns.iloc[4:], fast.returns.iloc[4:])


def test_zero_weights_earn_cash_rate(single_asset):
    weights = pd.DataFrame(0.0, index=single_asset.index, columns=["A"])
    costs = CostModel(commission_bps=0, slippage_bps=0, cash_rate=0.04)
    result = simulate(single_asset, weights, costs=costs)
    daily = (1.04) ** (1 / TRADING_DAYS) - 1
    np.testing.assert_allclose(result.returns.to_numpy(), daily)


def test_short_position_mirrors_returns_with_daily_rebalance(single_asset):
    weights = pd.DataFrame(-1.0, index=single_asset.index, columns=["A"])
    result = simulate(single_asset, weights, costs=CostModel.zero())
    asset_ret = single_asset["A"].pct_change().fillna(0.0)
    np.testing.assert_allclose(result.gross_returns.iloc[1:], -asset_ret.iloc[1:], atol=1e-12)


def test_borrow_fee_charged_on_shorts(single_asset):
    weights = pd.DataFrame(-0.5, index=single_asset.index, columns=["A"])
    free = simulate(single_asset, weights, costs=CostModel.zero())
    costly = simulate(
        single_asset, weights, costs=CostModel(commission_bps=0, slippage_bps=0, borrow_bps=100)
    )
    fee = 0.5 * 0.01 / TRADING_DAYS
    diff = (free.returns - costly.returns).iloc[1:]
    np.testing.assert_allclose(diff.to_numpy(), fee, rtol=1e-9)


def test_trading_costs_scale_with_turnover(single_asset):
    weights = pd.DataFrame(1.0, index=single_asset.index, columns=["A"])
    weights.iloc[::2] = 0.0  # flip in and out every bar
    costs = CostModel(commission_bps=1, slippage_bps=4)
    result = simulate(single_asset, weights, costs=costs)
    assert result.turnover.iloc[1:].min() > 0.99
    np.testing.assert_allclose(result.costs.iloc[1:], result.turnover.iloc[1:] * 5e-4)
    assert result.returns.sum() < result.gross_returns.sum()


@pytest.mark.parametrize("freq", ["D", "W", "M", 5])
def test_engine_matches_reference_loop(prices, freq):
    rng = np.random.default_rng(11)
    raw = pd.DataFrame(
        rng.uniform(-0.4, 0.6, prices.shape), index=prices.index, columns=prices.columns
    )
    raw.iloc[:20] = np.nan
    costs = CostModel(commission_bps=2, slippage_bps=3)
    result = simulate(prices, raw, costs=costs, rebalance=freq)
    rebal = rebalance_mask(prices.index, freq)
    expected = reference_simulation(prices, raw, rebal, costs.trade_cost_rate)
    np.testing.assert_allclose(result.returns.to_numpy(), expected, atol=1e-12)


def test_buy_and_hold_matches_average_price_relative(prices):
    result = buy_and_hold(prices, costs=CostModel.zero())
    final = result.equity.iloc[-1] / result.initial_capital
    assert final == pytest.approx((prices.iloc[-1] / prices.iloc[0]).mean())
    assert result.turnover.iloc[1:].sum() == 0


def test_rebalance_mask_month_end_and_interval():
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    monthly = rebalance_mask(index, "M")
    assert list(index[monthly].strftime("%m-%d")) == ["01-31", "02-29", "03-29"]
    every = rebalance_mask(index, 10)
    assert every[0] and every[10] and every.sum() == int(np.ceil(len(index) / 10))
    with pytest.raises(ValueError):
        rebalance_mask(index, "X")
    with pytest.raises(ValueError):
        rebalance_mask(index, 0)


def test_assets_without_prices_get_no_weight():
    index = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": np.linspace(100, 110, 10), "B": np.nan}, index=index)
    prices.loc[index[5] :, "B"] = np.linspace(50, 55, 5)
    weights = pd.DataFrame(0.5, index=index, columns=["A", "B"])
    result = simulate(prices, weights, costs=CostModel.zero())
    assert (result.weights["B"].iloc[:5] == 0).all()
    assert result.weights["B"].iloc[5] == pytest.approx(0.5)
    assert np.isfinite(result.returns).all()


def test_simulate_rejects_unknown_assets(single_asset):
    weights = pd.DataFrame(1.0, index=single_asset.index, columns=["ZZZ"])
    with pytest.raises(ValueError, match="unknown assets"):
        simulate(single_asset, weights)
