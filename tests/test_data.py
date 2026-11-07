import numpy as np
import pandas as pd
import pytest

from qbt.data import generate_prices, load_csv, simulate_regimes


def test_generator_is_deterministic_and_well_formed():
    a = generate_prices(n_assets=3, n_days=500, seed=1)
    b = generate_prices(n_assets=3, n_days=500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (500, 3)
    assert isinstance(a.index, pd.DatetimeIndex)
    assert a.index.is_monotonic_increasing and a.index.is_unique
    assert (a > 0).all().all()
    assert np.allclose(a.iloc[0], 100.0)


def test_generator_seeds_differ():
    a = generate_prices(n_assets=2, n_days=300, seed=1)
    b = generate_prices(n_assets=2, n_days=300, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_generator_correlation_is_roughly_respected():
    prices = generate_prices(n_assets=2, n_days=252 * 20, seed=3, correlation=0.6)
    corr = np.log(prices).diff().dropna().corr().iloc[0, 1]
    assert 0.45 < corr < 0.75


def test_generator_returns_regimes():
    prices, regimes = generate_prices(n_assets=1, n_days=2000, seed=5, return_regimes=True)
    assert regimes.index.equals(prices.index)
    assert set(regimes.unique()) <= {"bull", "bear", "chop"}
    # bear regimes are more volatile than chop regimes
    rets = np.log(prices.iloc[:, 0]).diff()
    by_regime = rets.groupby(regimes).std()
    if {"bear", "chop"} <= set(by_regime.index):
        assert by_regime["bear"] > by_regime["chop"]


def test_regime_chain_validates_matrix():
    with pytest.raises(ValueError):
        simulate_regimes(10, np.array([[0.5, 0.4], [0.5, 0.5]]))
    states = simulate_regimes(1000, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert (states == 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"n_assets": 0}, {"n_days": 1}, {"correlation": 1.0}, {"n_assets": 2, "tickers": ["X"]}],
)
def test_generator_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_prices(**kwargs)


def test_load_wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("Date,SPY,TLT\n2024-01-03,101,99\n2024-01-02,100,100\n2024-01-03,102,98\n")
    prices = load_csv(path)
    assert list(prices.columns) == ["SPY", "TLT"]
    assert prices.index.is_monotonic_increasing
    assert len(prices) == 2  # duplicate date keeps the last row
    assert prices.loc["2024-01-03", "SPY"] == 102


def test_load_ohlcv_prefers_adjusted_close(tmp_path):
    path = tmp_path / "aapl.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume\n"
        "2024-01-02,10,11,9,10.5,10.0,100\n"
        "2024-01-03,10.5,12,10,11.5,11.0,120\n"
    )
    prices = load_csv(path)
    assert list(prices.columns) == ["AAPL"]
    assert prices["AAPL"].tolist() == [10.0, 11.0]


def test_load_csv_rejects_non_positive_prices(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,X\n2024-01-02,1\n2024-01-03,0\n")
    with pytest.raises(ValueError, match="positive"):
        load_csv(path)


def test_csv_round_trip(tmp_path):
    original = generate_prices(n_assets=2, n_days=50, seed=9)
    path = tmp_path / "prices.csv"
    original.to_csv(path)
    loaded = load_csv(path)
    pd.testing.assert_frame_equal(loaded, original, check_freq=False)
