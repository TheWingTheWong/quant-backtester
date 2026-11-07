import numpy as np
import pandas as pd
import pytest

from qbt.costs import CostModel
from qbt.engine import run_backtest
from qbt.metrics import annualized_vol
from qbt.strategies import (
    REGISTRY,
    BollingerMeanReversion,
    SMACrossover,
    TimeSeriesMomentum,
    VolTargetRiskParity,
    get_strategy,
)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_weights_have_price_shape(prices, key):
    weights = REGISTRY[key]().generate_weights(prices)
    assert weights.shape == prices.shape
    assert weights.index.equals(prices.index)
    assert list(weights.columns) == list(prices.columns)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_no_lookahead(prices, key):
    """Weights up to bar t must not change when future bars are removed."""
    strategy = REGISTRY[key]()
    full = strategy.generate_weights(prices)
    cut = 700
    partial = strategy.generate_weights(prices.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_freq=False)


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_strategy_backtests(prices, key):
    cls = REGISTRY[key]
    result = run_backtest(cls(), prices, rebalance=cls.default_rebalance)
    assert np.isfinite(result.returns).all()
    assert result.benchmark is not None
    assert len(result.returns) == len(prices)


def test_registry_lookup():
    assert get_strategy("sma") is SMACrossover
    with pytest.raises(KeyError, match="available"):
        get_strategy("nope")


def test_sma_validation_and_signal():
    with pytest.raises(ValueError):
        SMACrossover(fast=50, slow=20)
    index = pd.bdate_range("2020-01-01", periods=300)
    up = pd.DataFrame({"A": np.linspace(100, 200, 300)}, index=index)
    w = SMACrossover(fast=5, slow=20).generate_weights(up)
    assert w["A"].iloc[:19].isna().all()
    assert (w["A"].iloc[19:] == 1.0).all()
    down = pd.DataFrame({"A": np.linspace(200, 100, 300)}, index=index)
    assert (SMACrossover(5, 20).generate_weights(down)["A"].iloc[19:] == 0.0).all()
    assert (SMACrossover(5, 20, allow_short=True).generate_weights(down)["A"].iloc[19:] == -1).all()


def test_sma_splits_capital_across_assets(prices):
    w = SMACrossover().generate_weights(prices).dropna()
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-12


def test_momentum_follows_trend_sign():
    index = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.005, 600)
    up = 100 * np.exp(np.cumsum(0.001 + noise))
    down = 100 * np.exp(np.cumsum(-0.001 + noise))
    prices = pd.DataFrame({"UP": up, "DOWN": down}, index=index)
    w = TimeSeriesMomentum(lookback=120, skip=10).generate_weights(prices).iloc[300:]
    assert (w["UP"] > 0).mean() > 0.9
    assert (w["DOWN"] < 0).mean() > 0.9
    long_only = TimeSeriesMomentum(lookback=120, skip=10, allow_short=False)
    assert (long_only.generate_weights(prices).fillna(0) >= 0).all().all()


def test_momentum_respects_leverage_cap(prices):
    w = TimeSeriesMomentum(asset_vol=1.0, max_leverage=1.5).generate_weights(prices)
    assert w.abs().sum(axis=1).max() <= 1.5 + 1e-9


def test_mean_reversion_buys_dips_and_sells_rips():
    index = pd.bdate_range("2020-01-01", periods=400)
    t = np.arange(400)
    wave = 100 + 5 * np.sin(2 * np.pi * t / 40)
    prices = pd.DataFrame({"A": wave}, index=index)
    strategy = BollingerMeanReversion(lookback=20, entry_z=1.2, exit_z=0.2)
    w = strategy.generate_weights(prices)["A"]
    z = (prices["A"] - prices["A"].rolling(20).mean()) / prices["A"].rolling(20).std()
    assert (w[z < -1.2] > 0).all()
    assert (w[z > 1.2] < 0).all()
    assert (w[z.abs() < 0.2] == 0).all()
    # hysteresis: some bars hold a position while |z| is between exit and entry
    between = (z.abs() > 0.2) & (z.abs() < 1.2)
    assert (w[between] != 0).any()


def test_mean_reversion_validation():
    with pytest.raises(ValueError):
        BollingerMeanReversion(entry_z=1.0, exit_z=1.5)


def test_risk_parity_hits_vol_target(prices):
    strategy = VolTargetRiskParity(vol_lookback=60, target_vol=0.08, max_leverage=4.0)
    result = run_backtest(strategy, prices, costs=CostModel.zero(), rebalance="D")
    realized = annualized_vol(result.returns.iloc[120:])
    assert realized == pytest.approx(0.08, rel=0.25)


def test_risk_parity_overweights_low_vol_assets(prices):
    # the generator orders assets from low to high volatility
    w = VolTargetRiskParity().generate_weights(prices).dropna().mean()
    assert w.iloc[0] > w.iloc[-1]
    assert (w > 0).all()
