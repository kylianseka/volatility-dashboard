import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_ohlc
from voldash.estimators import (
    all_estimators,
    close_to_close,
    garman_klass,
    parkinson,
    rogers_satchell,
    yang_zhang,
)

INTRADAY, OVERNIGHT = 0.30, 0.20
TOTAL = float(np.sqrt(INTRADAY**2 + OVERNIGHT**2))
WINDOW = 1000


def test_close_to_close_uses_252_annualisation() -> None:
    a = 0.01
    closes = 100.0 * np.exp(np.cumsum([0.0] + [a if i % 2 == 0 else -a for i in range(20)]))
    df = pd.DataFrame({"Close": closes}, index=pd.bdate_range("2024-01-01", periods=21))
    expected = a * np.sqrt(20 / 19) * np.sqrt(252)
    assert close_to_close(df, 20).iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_yang_zhang_recovers_total_volatility(gbm_with_gaps: pd.DataFrame) -> None:
    assert yang_zhang(gbm_with_gaps, WINDOW).volatility.iloc[-1] == pytest.approx(TOTAL, rel=0.08)
    assert close_to_close(gbm_with_gaps, WINDOW).iloc[-1] == pytest.approx(TOTAL, rel=0.08)


def test_range_estimators_miss_overnight_gaps(gbm_with_gaps: pd.DataFrame) -> None:
    # Parkinson / GK / RS ne voient que la séance : ils mesurent ~30 %, pas ~36 %.
    for estimator in (parkinson, garman_klass, rogers_satchell):
        value = estimator(gbm_with_gaps, WINDOW).iloc[-1]
        assert value == pytest.approx(INTRADAY, rel=0.08)
        assert value < TOTAL * 0.92


def test_overnight_share_matches_simulation(gbm_with_gaps: pd.DataFrame) -> None:
    share = yang_zhang(gbm_with_gaps, WINDOW).overnight_share.iloc[-1]
    assert share == pytest.approx(OVERNIGHT**2 / TOTAL**2, abs=0.05)


def test_rogers_satchell_is_drift_invariant() -> None:
    trending = simulate_ohlc(days=1200, intraday_sigma=0.25, overnight_sigma=0.0, drift=3.0, seed=11)
    rs = rogers_satchell(trending, 1000).iloc[-1]
    pk = parkinson(trending, 1000).iloc[-1]
    assert rs == pytest.approx(0.25, rel=0.08)
    assert pk > rs  # Parkinson assimile la tendance à de la volatilité


def test_no_lookahead_bias(gbm_with_gaps: pd.DataFrame) -> None:
    cut = 600
    full = all_estimators(gbm_with_gaps, 21)
    truncated = all_estimators(gbm_with_gaps.iloc[:cut], 21)
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated)


def test_zero_range_day_is_finite() -> None:
    df = simulate_ohlc(days=60, intraday_sigma=0.2, overnight_sigma=0.1)
    df.iloc[30, :4] = df.iloc[30]["Open"]  # séance plate : H = L = O = C
    result = all_estimators(df, 21)
    assert np.isfinite(result.dropna().to_numpy()).all()


def test_yang_zhang_rejects_window_below_two(gbm_with_gaps: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        yang_zhang(gbm_with_gaps, 1)
