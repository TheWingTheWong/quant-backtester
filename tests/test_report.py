import re

import numpy as np
import pandas as pd

from qbt.engine import run_backtest
from qbt.report import render_tearsheet, write_tearsheet
from qbt.report.charts import heatmap, linear_ticks, log_ticks, nice_step
from qbt.strategies import SMACrossover


def test_tearsheet_is_self_contained(prices, tmp_path):
    result = run_backtest(SMACrossover(), prices)
    html = render_tearsheet(result, title="SMA crossover", subtitle="fast 20 · slow 100")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html.lower()
    assert not re.search(r"""(src|href)=["']https?://""", html)
    assert "@import" not in html
    # equity, drawdown and rolling Sharpe plots plus the monthly heatmap
    assert html.count('class="plot-svg"') == 3
    assert 'class="heatmap"' in html
    assert "Growth of 1M" in html
    assert "Monthly returns" in html
    path = write_tearsheet(result, tmp_path / "out" / "report.html", title="SMA")
    assert path.exists() and path.stat().st_size > 10_000


def test_tearsheet_without_benchmark(prices):
    result = run_backtest(SMACrossover(), prices, benchmark=False)
    html = render_tearsheet(result)
    assert "Benchmark" not in html.split("<main>")[1].split("Monthly returns")[0]


def test_tearsheet_escapes_text(prices):
    result = run_backtest(SMACrossover(), prices)
    html = render_tearsheet(result, title="<b>x</b>", settings={"Data": "<script>"})
    assert "<b>x</b>" not in html
    assert "<script>" not in html


def test_heatmap_cells_and_year_column():
    idx = pd.bdate_range("2022-01-03", "2023-12-29")
    r = pd.Series(np.full(len(idx), 0.0005), index=idx)
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod().unstack() - 1
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    svg = heatmap(monthly, yearly)
    assert svg.count("<rect") == 2 * 13
    assert ">Year<" in svg
    assert "Jan 2022:" in svg


def test_tick_helpers():
    assert nice_step(1.0, 5) == 0.2
    ticks = linear_ticks(-0.23, 0.0, fmt=str, target=4)
    assert ticks[-1].value == 0.0 and all(t.value <= 0 for t in ticks)
    logs = log_ticks(900_000, 2_100_000, fmt=str)
    assert [t.value for t in logs][:2] == [1_000_000, 1_250_000]
