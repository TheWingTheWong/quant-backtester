import numpy as np
import pandas as pd
import pytest

from qbt import sizing


def test_rolling_covariance_matches_pandas(prices):
    rets = prices.pct_change().dropna()
    cov = sizing.rolling_covariance(rets, lookback=30)
    assert cov.shape == (len(rets), rets.shape[1], rets.shape[1])
    assert np.isnan(cov[28]).all()
    expected = rets.iloc[100 - 29 : 101].cov().to_numpy() * 252
    np.testing.assert_allclose(cov[100], expected, rtol=1e-10)


def test_inverse_vol_weights_sum_to_one(prices):
    w = sizing.inverse_vol_weights(prices.pct_change(), 60).dropna()
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_cap_gross_leverage():
    w = pd.DataFrame({"A": [1.0, 0.2], "B": [-1.0, 0.1]})
    capped = sizing.cap_gross_leverage(w, 1.0)
    assert capped.abs().sum(axis=1).tolist() == pytest.approx([1.0, 0.3])
    with pytest.raises(ValueError):
        sizing.cap_gross_leverage(w, 0)


def test_vol_target_scales_ex_ante_vol(prices):
    rets = prices.pct_change()
    w = pd.DataFrame(0.25, index=prices.index, columns=prices.columns)
    scaled = sizing.vol_target(w, rets, target=0.05, lookback=60, max_leverage=10)
    cov = sizing.rolling_covariance(rets.fillna(0.0), 60)
    vol = sizing.portfolio_vol(scaled, cov).iloc[100:]
    np.testing.assert_allclose(vol, 0.05, rtol=1e-9)
    assert (scaled.iloc[:58] == 0).all().all()
