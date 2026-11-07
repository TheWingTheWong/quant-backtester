import numpy as np
import pandas as pd
import pytest

from qbt import metrics as m


def series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_cagr_of_constant_growth():
    daily = (1.10) ** (1 / 252) - 1
    r = series([daily] * 252 * 3)
    assert m.cagr(r) == pytest.approx(0.10, rel=1e-9)
    assert m.total_return(r) == pytest.approx(1.10**3 - 1, rel=1e-9)
    assert m.annualized_vol(r) == pytest.approx(0.0, abs=1e-12)
    assert m.sharpe_ratio(r) == 0.0  # zero volatility is reported as 0, not inf


def test_sharpe_and_sortino_on_known_sample():
    rng = np.random.default_rng(1)
    r = series(rng.normal(0.0005, 0.01, 5000))
    expected = r.mean() / r.std() * np.sqrt(252)
    assert m.sharpe_ratio(r) == pytest.approx(expected)
    assert m.sortino_ratio(r) > m.sharpe_ratio(r) > 0
    assert m.sharpe_ratio(r, risk_free=0.5) < 0


def test_drawdown_depth_and_duration():
    # +10%, -50% , then recover in steps and set a new high
    r = series([0.10, -0.50, 0.50, 0.30, 0.20, 0.0])
    dd = m.drawdown_series(r)
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    assert dd.iloc[0] == 0.0
    # equity: 1.1, .55, .825, 1.0725, 1.287 -> underwater for bars 1..3
    assert m.max_drawdown_duration(r) == 3
    assert m.calmar_ratio(r) == pytest.approx(m.cagr(r) / 0.5)


def test_drawdown_counts_initial_capital_as_peak():
    r = series([-0.1, 0.05])
    assert m.max_drawdown(r) == pytest.approx(-0.1)


def test_top_drawdowns_sorted_and_recovery():
    r = series([0.1, -0.2, 0.3, -0.05, 0.0, -0.1])
    episodes = m.top_drawdowns(r, n=5)
    assert [round(e.depth, 4) for e in episodes] == sorted(round(e.depth, 4) for e in episodes)
    assert episodes[0].depth == pytest.approx(-0.2)
    assert episodes[0].end is not None
    assert episodes[1].end is None  # last episode never recovers


def test_hit_rate_ignores_flat_days():
    r = series([0.01, -0.01, 0.0, 0.0, 0.02])
    assert m.hit_rate(r) == pytest.approx(2 / 3)
    assert m.profit_factor(r) == pytest.approx(3.0)


def test_monthly_returns_compound_within_month():
    idx = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-02-01", "2024-03-01"])
    r = pd.Series([0.1, 0.1, -0.05, 0.02], index=idx)
    table = m.monthly_returns(r)
    assert table.loc[2023, 1] == pytest.approx(0.21)
    assert table.loc[2023, 2] == pytest.approx(-0.05)
    assert np.isnan(table.loc[2023, 3])
    assert list(table.columns) == list(range(1, 13))
    assert m.annual_returns(r).loc[2023] == pytest.approx(1.21 * 0.95 - 1)


def test_compute_metrics_keys_and_turnover_annualization():
    r = series([0.001, -0.002, 0.003] * 100)
    turnover = series([0.1] * 300)
    out = m.compute_metrics(r, turnover=turnover, exposure=series([1.0] * 300))
    assert out["turnover"] == pytest.approx(25.2)
    assert out["avg_exposure"] == pytest.approx(1.0)
    for key in ("cagr", "sharpe", "sortino", "calmar", "max_drawdown", "max_dd_duration"):
        assert key in out


def test_empty_series_is_safe():
    r = series([])
    out = m.compute_metrics(r)
    assert out["cagr"] == 0.0 and out["max_drawdown"] == 0.0


@pytest.mark.parametrize(
    ("key", "value", "text"),
    [
        ("cagr", 0.1234, "12.34%"),
        ("sharpe", 1.234, "1.23"),
        ("max_dd_duration", 41.6, "42"),
        ("sharpe", -0.001, "0.00"),
        ("profit_factor", float("inf"), "inf"),
    ],
)
def test_format_metric(key, value, text):
    assert m.format_metric(key, value) == text
