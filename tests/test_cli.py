import json

import pytest

from qbt.cli import format_table, main, parse_grid
from qbt.strategies import SMACrossover, TimeSeriesMomentum

SMALL = ["--years", "4", "--assets", "3", "--seed", "3"]


def test_run_prints_metrics_and_writes_tearsheet(tmp_path, capsys):
    out = tmp_path / "sma.html"
    code = main(
        ["run", "--strategy", "sma", "--fast", "20", "--slow", "100", "-o", str(out), *SMALL]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "SMA crossover  fast=20 slow=100" in text
    for label in ("CAGR", "Sharpe", "Sortino", "Calmar", "Max drawdown", "Hit rate", "Turnover"):
        assert label in text
    assert out.exists()


def test_run_json_output(capsys):
    code = main(["run", "-s", "momentum", "--lookback", "126", "--no-report", "--json", *SMALL])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["params"]["lookback"] == 126
    assert {"sharpe", "cagr", "max_drawdown"} <= set(payload["metrics"])


def test_run_with_overlays_and_bool_flags(capsys):
    args = [
        "run",
        "-s",
        "sma",
        "--allow-short",
        "--overlay-vol",
        "0.1",
        "--overlay-leverage",
        "1.5",
        "--rebalance",
        "W",
        "--no-report",
        *SMALL,
    ]
    assert main(args) == 0
    assert "allow_short=True" in capsys.readouterr().out


def test_invalid_parameters_exit_with_error(capsys):
    assert main(["run", "-s", "sma", "--fast", "100", "--slow", "20", "--no-report"]) == 2
    assert "must be shorter" in capsys.readouterr().err


def test_unknown_strategy_flag_is_rejected():
    with pytest.raises(SystemExit):
        main(["run", "-s", "meanrev", "--fast", "10"])


def test_grid_and_walkforward(capsys, tmp_path):
    save = tmp_path / "grid.csv"
    args = ["grid", "-s", "sma", "-g", "fast=10,20", "-g", "slow=50:100:50", "--save", str(save)]
    assert main([*args, *SMALL]) == 0
    out = capsys.readouterr().out
    assert "4 valid combinations" in out and save.exists()
    wf = [
        "walkforward",
        "-s",
        "sma",
        "-g",
        "fast=10,20",
        "-g",
        "slow=100",
        "--train",
        "504",
        "--test",
        "126",
        *SMALL,
    ]
    assert main(wf) == 0
    assert "Stitched out-of-sample performance" in capsys.readouterr().out


def test_data_export_and_csv_backtest(tmp_path, capsys):
    csv = tmp_path / "prices.csv"
    assert main(["data", "-o", str(csv), "--assets", "2", "--years", "3"]) == 0
    assert main(["run", "-s", "meanrev", "--csv", str(csv), "--no-report"]) == 0
    assert "2 assets" in capsys.readouterr().out


def test_strategies_lists_registry(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    for key in ("sma", "momentum", "meanrev", "riskparity", "equal"):
        assert key in out


def test_parse_grid_types_and_ranges():
    grid = parse_grid(["fast=5:15:5", "allow_short=true,false"], SMACrossover)
    assert grid == {"fast": [5, 10, 15], "allow_short": [True, False]}
    grid = parse_grid(["asset-vol=0.1,0.2"], TimeSeriesMomentum)
    assert grid == {"asset_vol": [0.1, 0.2]}
    with pytest.raises(ValueError):
        parse_grid(["nope=1"], SMACrossover)
    with pytest.raises(ValueError):
        parse_grid(["fast"], SMACrossover)


def test_format_table_alignment():
    table = format_table(["Metric", "Value"], [["Sharpe", "1.00"], ["CAGR", "12.5%"]])
    lines = table.splitlines()
    assert lines[0].startswith("Metric")
    assert lines[2].endswith(" 1.00") and lines[3].endswith("12.5%")
