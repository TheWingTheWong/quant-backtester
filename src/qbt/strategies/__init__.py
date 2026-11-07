"""Built-in strategies. Importing this package registers all of them."""

from qbt.strategies.base import REGISTRY, Strategy, get_strategy, register
from qbt.strategies.buy_hold import EqualWeight
from qbt.strategies.mean_reversion import BollingerMeanReversion, zscore
from qbt.strategies.risk_parity import VolTargetRiskParity
from qbt.strategies.trend import SMACrossover, TimeSeriesMomentum

__all__ = [
    "REGISTRY",
    "BollingerMeanReversion",
    "EqualWeight",
    "SMACrossover",
    "Strategy",
    "TimeSeriesMomentum",
    "VolTargetRiskParity",
    "get_strategy",
    "register",
    "zscore",
]
