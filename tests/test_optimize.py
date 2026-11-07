import pandas as pd
import pytest

from qbt.optimize import expand_grid, grid_search, walk_forward
from qbt.strategies import SMACrossover


def test_expand_grid():
    combos = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert combos[0] == {"a": 1, "b": "x"}


def test_grid_search_skips_invalid_and_sorts(prices):
    table = grid_search(SMACrossover, prices, {"fast": [10, 50], "slow": [20, 100]})
    # (50, 20) is invalid; three remain
    assert len(table) == 3
    assert table["sharpe"].is_monotonic_decreasing
    assert table["fast"].dtype.kind == "i"


def test_grid_search_minimizes_where_appropriate(prices):
    table = grid_search(
        SMACrossover, prices, {"fast": [5, 20], "slow": [50, 200]}, metric="max_dd_duration"
    )
    assert table["max_dd_duration"].is_monotonic_increasing


def test_grid_search_unknown_metric(prices):
    with pytest.raises(KeyError):
        grid_search(SMACrossover, prices, {"fast": [10], "slow": [50]}, metric="alpha")


def test_walk_forward_folds_are_contiguous_and_out_of_sample(prices):
    grid = {"fast": [10, 20], "slow": [50, 100]}
    result = walk_forward(SMACrossover, prices, grid, train_bars=504, test_bars=126)
    assert len(result.folds) == len(range(504, len(prices), 126))
    for fold in result.folds:
        assert fold.train_end < fold.test_start
        assert fold.params["fast"] in grid["fast"] and fold.params["slow"] in grid["slow"]
    # stitched returns cover everything after the first training window exactly once
    expected_index = prices.index[504:]
    pd.testing.assert_index_equal(result.returns.index, expected_index, check_names=False)
    table = result.folds_table()
    assert {"train", "test", "fast", "slow", "train_sharpe", "test_sharpe"} <= set(table)
    assert "sharpe" in result.metrics()


def test_walk_forward_anchored_starts_at_first_bar(prices):
    result = walk_forward(
        SMACrossover,
        prices,
        {"fast": [10], "slow": [50]},
        train_bars=504,
        test_bars=252,
        anchored=True,
    )
    assert all(f.train_start == prices.index[0] for f in result.folds)


def test_walk_forward_needs_enough_data(prices):
    with pytest.raises(ValueError):
        walk_forward(SMACrossover, prices.iloc[:300], {"fast": [10], "slow": [50]})
