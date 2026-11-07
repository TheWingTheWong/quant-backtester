"""Price data loading.

Every loader returns a *wide* ``DataFrame`` of close prices: a sorted, unique
``DatetimeIndex`` and one float column per asset. That is the only data
contract the rest of the engine relies on.

Three sources are supported:

* :func:`generate_prices` - a synthetic regime-switching GBM generator, so the
  whole project runs offline and deterministically from a seed.
* :func:`load_csv` - wide CSVs (``date,AAA,BBB,...``) or single-asset OHLCV
  CSVs (uses ``Adj Close`` or ``Close``).
* :func:`load_yfinance` - optional, only if the ``yfinance`` extra is installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

try:  # optional dependency
    import yfinance as _yf  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - depends on environment
    _yf = None


@dataclass(frozen=True)
class Regime:
    """Annualized drift/volatility for one market regime."""

    name: str
    mu: float
    sigma: float


DEFAULT_REGIMES: tuple[Regime, ...] = (
    Regime("bull", mu=0.14, sigma=0.14),
    Regime("bear", mu=-0.22, sigma=0.32),
    Regime("chop", mu=0.02, sigma=0.10),
)

# Daily transition probabilities between DEFAULT_REGIMES (rows sum to 1).
# Expected regime durations: bull ~ 250d, bear ~ 80d, chop ~ 100d.
DEFAULT_TRANSITIONS = np.array(
    [
        [0.996, 0.002, 0.002],
        [0.008, 0.9875, 0.0045],
        [0.006, 0.004, 0.990],
    ]
)


def simulate_regimes(
    n_days: int,
    transitions: np.ndarray = DEFAULT_TRANSITIONS,
    rng: np.random.Generator | None = None,
    start_state: int = 0,
) -> np.ndarray:
    """Simulate a discrete Markov chain and return the state index per day."""
    rng = rng if rng is not None else np.random.default_rng()
    transitions = np.asarray(transitions, dtype=float)
    if transitions.ndim != 2 or transitions.shape[0] != transitions.shape[1]:
        raise ValueError("transitions must be a square matrix")
    if not np.allclose(transitions.sum(axis=1), 1.0):
        raise ValueError("each row of the transition matrix must sum to 1")

    cumulative = transitions.cumsum(axis=1)
    draws = rng.random(n_days)
    states = np.empty(n_days, dtype=np.int64)
    state = start_state
    # A Markov chain is inherently sequential; this loop is O(n_days) scalar
    # work and is not on the backtest hot path.
    for t in range(n_days):
        states[t] = state
        state = int(np.searchsorted(cumulative[state], draws[t], side="right"))
        state = min(state, len(cumulative) - 1)
    return states


def generate_prices(
    n_assets: int = 4,
    n_days: int = TRADING_DAYS * 10,
    start: str = "2015-01-02",
    seed: int | None = 42,
    correlation: float = 0.3,
    regimes: Sequence[Regime] = DEFAULT_REGIMES,
    transitions: np.ndarray = DEFAULT_TRANSITIONS,
    start_price: float = 100.0,
    tickers: Sequence[str] | None = None,
    return_regimes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic close prices from a regime-switching GBM.

    A single market-wide regime path is drawn from a Markov chain. Each asset
    follows geometric Brownian motion whose drift and volatility depend on the
    current regime, scaled by an asset-specific multiplier so that assets have
    different risk profiles. Shocks are equicorrelated with ``correlation``.

    Args:
        n_assets: number of assets (columns).
        n_days: number of business days to simulate.
        start: first date of the index.
        seed: RNG seed; ``None`` for non-deterministic output.
        correlation: pairwise correlation of daily shocks, in ``[0, 1)``.
        regimes: regime definitions (annualized mu/sigma).
        transitions: daily Markov transition matrix matching ``regimes``.
        start_price: initial price for every asset.
        tickers: optional column names; defaults to ``SYN0, SYN1, ...``.
        return_regimes: also return the regime name per day.

    Returns:
        Wide price DataFrame, or ``(prices, regimes)`` if ``return_regimes``.
    """
    if n_assets < 1 or n_days < 2:
        raise ValueError("need at least one asset and two days")
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must be in [0, 1)")
    if len(regimes) != len(transitions):
        raise ValueError("regimes and transitions must have the same size")

    rng = np.random.default_rng(seed)
    states = simulate_regimes(n_days, transitions, rng)

    mu = np.array([r.mu for r in regimes])[states]  # (T,)
    sigma = np.array([r.sigma for r in regimes])[states]  # (T,)

    # Asset-specific risk multipliers: a spread of low- to high-vol assets.
    vol_scale = np.linspace(0.6, 1.6, n_assets) if n_assets > 1 else np.ones(1)
    beta = rng.uniform(0.7, 1.3, n_assets)

    # Equicorrelated Gaussian shocks via Cholesky.
    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    shocks = rng.standard_normal((n_days, n_assets)) @ chol.T

    dt = 1.0 / TRADING_DAYS
    sig = sigma[:, None] * vol_scale[None, :]
    drift = (mu[:, None] * beta[None, :] - 0.5 * sig**2) * dt
    log_ret = drift + sig * np.sqrt(dt) * shocks
    log_ret[0] = 0.0  # first row is the starting price

    prices = start_price * np.exp(np.cumsum(log_ret, axis=0))
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    names = list(tickers) if tickers is not None else [f"SYN{i}" for i in range(n_assets)]
    if len(names) != n_assets:
        raise ValueError("len(tickers) must equal n_assets")

    frame = pd.DataFrame(prices, index=index, columns=names)
    if return_regimes:
        labels = pd.Series(np.array([r.name for r in regimes])[states], index=index, name="regime")
        return frame, labels
    return frame


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a price frame to the engine's data contract."""
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError("no usable price data")
    if (frame <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    return frame


_DATE_COLUMNS = ("date", "datetime", "timestamp", "time")
_CLOSE_COLUMNS = ("adj close", "adj_close", "adjclose", "close")
_OHLCV = {"open", "high", "low", "close", "adj close", "adj_close", "adjclose", "volume"}


def load_csv(path: str | Path, date_column: str | None = None) -> pd.DataFrame:
    """Load prices from a CSV file.

    Two layouts are recognized automatically:

    * **wide**: a date column followed by one close-price column per asset;
    * **OHLCV**: a single asset with ``Open/High/Low/Close[/Adj Close]/Volume``
      columns. The adjusted close is preferred; the asset is named after the
      file stem.
    """
    path = Path(path)
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{path} is empty")

    lowered = {c: str(c).strip().lower() for c in raw.columns}
    if date_column is None:
        date_column = next((c for c, low in lowered.items() if low in _DATE_COLUMNS), None)
        if date_column is None:
            date_column = raw.columns[0]
    raw = raw.set_index(date_column)
    lowered.pop(date_column, None)

    if (
        set(lowered.values()) & {"close", "adj close", "adj_close", "adjclose"}
        and set(lowered.values()) <= _OHLCV
    ):
        for wanted in _CLOSE_COLUMNS:
            match = next((c for c, low in lowered.items() if low == wanted), None)
            if match is not None:
                return _clean(raw[[match]].rename(columns={match: path.stem.upper()}))

    return _clean(raw)


def load_yfinance(
    tickers: Sequence[str], start: str = "2010-01-01", end: str | None = None
) -> pd.DataFrame:  # pragma: no cover - requires network
    """Download adjusted closes with ``yfinance`` (optional extra)."""
    if _yf is None:
        raise ImportError(
            "yfinance is not installed; install the extra with "
            "`uv sync --extra yfinance` or `pip install quant-backtester[yfinance]`"
        )
    data = _yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"no data returned for {list(tickers)}")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return _clean(close)
