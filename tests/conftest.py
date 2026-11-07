import numpy as np
import pandas as pd
import pytest

from qbt.data import generate_prices


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """Four synthetic assets, five years, fixed seed."""
    return generate_prices(n_assets=4, n_days=252 * 5, seed=7)


@pytest.fixture
def single_asset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0004, 0.01, 600)
    rets[0] = 0.0
    index = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame({"A": 100 * np.cumprod(1 + rets)}, index=index)
