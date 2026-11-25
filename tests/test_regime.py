import numpy as np
import pandas as pd
import pytest

from voldash.regime import (
    Regime,
    classify_regime,
    rolling_percentile_rank,
    snapshot,
    tension_ratio,
    volatility_cone,
)


def test_percentile_rank_uses_only_past_window() -> None:
    series = pd.Series([1.0, 5.0, 3.0, 4.0, 2.0])
    ranks = rolling_percentile_rank(series, lookback=3)
    assert ranks.iloc[:3].isna().all()
    assert ranks.iloc[3] == pytest.approx(200 / 3)  # 4 vs [1, 5, 3] -> 2/3
    assert ranks.iloc[4] == pytest.approx(0.0)  # 2 vs [5, 3, 4] -> 0/3


def test_percentile_rank_ignores_incomplete_windows() -> None:
    series = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0])
    ranks = rolling_percentile_rank(series, lookback=3)
    assert np.isnan(ranks.iloc[3])
    assert ranks.iloc[4] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("ratio", "percentile", "expected"),
    [
        (1.30, 50.0, Regime.STRESS),
        (1.00, 95.0, Regime.STRESS),
        (0.70, 50.0, Regime.COMPRESSION),
        (1.00, 5.0, Regime.COMPRESSION),
        (1.00, 50.0, Regime.NORMAL),
        (float("nan"), 50.0, Regime.UNKNOWN),
    ],
)
def test_classify_regime(ratio: float, percentile: float, expected: Regime) -> None:
    assert classify_regime(ratio, percentile) is expected


def test_cone_quantiles_are_ordered(gbm_with_gaps: pd.DataFrame) -> None:
    cone = volatility_cone(gbm_with_gaps)
    assert list(cone.index) == [10, 21, 63, 126, 252]
    ordered = cone[["Minimum", "P10", "Médiane", "P90", "Maximum"]].to_numpy()
    assert (np.diff(ordered, axis=1) >= 0).all()
    # Le cône se resserre avec l'horizon (erreur d'échantillonnage ~ 1/sqrt(2N)).
    width = cone["P90"] - cone["P10"]
    assert width.loc[252] < width.loc[10]


def test_snapshot_on_stationary_series(gbm_with_gaps: pd.DataFrame) -> None:
    snap = snapshot(gbm_with_gaps)
    assert 0.0 <= snap.percentile_rank <= 100.0
    assert snap.tension_ratio == pytest.approx(tension_ratio(gbm_with_gaps).iloc[-1])
    assert 0.0 < snap.overnight_share < 1.0
    assert snap.vol_yz_252 == pytest.approx(0.36, rel=0.15)
