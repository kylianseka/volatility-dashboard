import numpy as np
import pandas as pd
import pytest


def simulate_ohlc(
    days: int,
    intraday_sigma: float,
    overnight_sigma: float,
    drift: float = 0.0,
    steps: int = 390,
    seed: int = 7,
) -> pd.DataFrame:
    """Barres OHLC issues d'un mouvement brownien discrétisé (volatilités annualisées)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    step_sd = intraday_sigma * np.sqrt(dt / steps)
    step_mu = drift * dt / steps
    gaps = rng.normal(0.0, overnight_sigma * np.sqrt(dt), days)
    paths = rng.normal(step_mu, step_sd, (days, steps)).cumsum(axis=1)

    log_close_prev = 0.0
    rows = []
    for day in range(days):
        log_open = log_close_prev + gaps[day]
        path = log_open + np.concatenate(([0.0], paths[day]))
        rows.append(np.exp([path[0], path.max(), path.min(), path[-1]]) * 100.0)
        log_close_prev = path[-1]
    index = pd.bdate_range("2015-01-01", periods=days)
    df = pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])
    df["Volume"] = 1_000
    return df


@pytest.fixture(scope="session")
def gbm_with_gaps() -> pd.DataFrame:
    return simulate_ohlc(days=1500, intraday_sigma=0.30, overnight_sigma=0.20)
