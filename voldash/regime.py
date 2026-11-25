"""Indicateurs de régime : ratio de tension, rang de percentile, cône de volatilité."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from voldash.config import (
    CONE_HORIZONS,
    LONG_WINDOW,
    PERCENTILE_CALM,
    PERCENTILE_STRESS,
    SHORT_WINDOW,
    TENSION_HIGH,
    TENSION_LOW,
)
from voldash.estimators import close_to_close, yang_zhang


def tension_ratio(df: pd.DataFrame) -> pd.Series:
    """R_t = sigma_YZ(21j) / sigma_YZ(252j), fenêtres fixes quel que soit le réglage UI."""
    short = yang_zhang(df, SHORT_WINDOW).volatility
    long = yang_zhang(df, LONG_WINDOW).volatility
    return (short / long).where(long > 0)


def rolling_percentile_rank(series: pd.Series, lookback: int = LONG_WINDOW) -> pd.Series:
    """VR_t : part des `lookback` valeurs précédentes <= valeur courante (en %).

    Vectorisé via une vue glissante ; NaN tant que l'historique est incomplet.
    """
    values = series.to_numpy(dtype=float)
    result = np.full(values.shape, np.nan)
    if len(values) <= lookback:
        return pd.Series(result, index=series.index)
    windows = sliding_window_view(values, lookback + 1)
    past, current = windows[:, :-1], windows[:, -1:]
    complete = ~np.isnan(windows).any(axis=1)
    ranks = (past <= current).mean(axis=1) * 100.0
    result[lookback:] = np.where(complete, ranks, np.nan)
    return pd.Series(result, index=series.index)


CONE_COLUMNS: tuple[str, ...] = ("Minimum", "P10", "Médiane", "P90", "Maximum", "Actuelle", "Rang (%)")


def volatility_cone(df: pd.DataFrame, horizons: tuple[int, ...] = CONE_HORIZONS) -> pd.DataFrame:
    """Distribution empirique de sigma_YZ par horizon sur tout l'historique fourni (en %)."""
    rows: dict[int, list[float]] = {}
    for horizon in horizons:
        series = yang_zhang(df, horizon).volatility.dropna()
        if len(series) < 2:
            continue
        current = float(series.iloc[-1])
        q = np.percentile(series, [0, 10, 50, 90, 100])
        rank = float((series <= current).mean() * 100.0)
        rows[horizon] = [*(q * 100.0), current * 100.0, rank]
    cone = pd.DataFrame.from_dict(rows, orient="index", columns=list(CONE_COLUMNS))
    cone.index.name = "Horizon (jours)"
    return cone


class Regime(StrEnum):
    STRESS = "Stress"
    COMPRESSION = "Compression"
    NORMAL = "Normal"
    UNKNOWN = "Historique insuffisant"


def classify_regime(ratio: float, percentile: float) -> Regime:
    if np.isnan(ratio) or np.isnan(percentile):
        return Regime.UNKNOWN
    if ratio > TENSION_HIGH or percentile >= PERCENTILE_STRESS:
        return Regime.STRESS
    if ratio < TENSION_LOW or percentile <= PERCENTILE_CALM:
        return Regime.COMPRESSION
    return Regime.NORMAL


@dataclass(frozen=True)
class Snapshot:
    """Photographie des indicateurs à la dernière séance (vol en décimal)."""

    date: pd.Timestamp
    close: float
    daily_return: float
    vol_yz_21: float
    vol_cc_21: float
    vol_yz_252: float
    tension_ratio: float
    percentile_rank: float
    overnight_share: float
    regime: Regime


def _last(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else float("nan")


def snapshot(df: pd.DataFrame) -> Snapshot:
    yz21 = yang_zhang(df, SHORT_WINDOW)
    yz252 = yang_zhang(df, LONG_WINDOW).volatility
    ratio = _last(tension_ratio(df))
    rank = _last(rolling_percentile_rank(yz21.volatility))
    return Snapshot(
        date=df.index[-1],
        close=float(df["Close"].iloc[-1]),
        daily_return=float(df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1.0),
        vol_yz_21=_last(yz21.volatility),
        vol_cc_21=_last(close_to_close(df, SHORT_WINDOW)),
        vol_yz_252=_last(yz252),
        tension_ratio=ratio,
        percentile_rank=rank,
        overnight_share=_last(yz21.overnight_share),
        regime=classify_regime(ratio, rank),
    )
