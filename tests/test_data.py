import pandas as pd
import pytest

from voldash.data import validate_ohlcv

AFTER_CLOSE = pd.Timestamp("2024-01-10 20:00")


def make_frame(rows: list[tuple[str, float, float, float, float, int]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([r[0] for r in rows])
    data = [r[1:] for r in rows]
    return pd.DataFrame(data, index=index, columns=["Open", "High", "Low", "Close", "Volume"])


def test_valid_bars_pass_untouched() -> None:
    raw = make_frame([("2024-01-02", 10, 11, 9, 10.5, 100), ("2024-01-03", 10.5, 12, 10, 11, 100)])
    clean, report = validate_ohlcv(raw, as_of=AFTER_CLOSE)
    assert len(clean) == 2
    assert report.rows_rejected == 0


def test_rejections_are_counted() -> None:
    raw = make_frame(
        [
            ("2024-01-02", 10, 11, 9, 10.5, 100),
            ("2024-01-03", 10, 9.5, 9, 10, 100),  # High < Open
            ("2024-01-04", 0, 11, 9, 10, 100),  # prix nul
            ("2024-01-05", 10, 10, 10, 10, 0),  # pas de transaction
            ("2024-01-08", 10, 11, 9, 10, 100),
            ("2024-01-08", 10, 11, 9, 10.2, 100),  # doublon -> garde le dernier
        ]
    )
    clean, report = validate_ohlcv(raw, as_of=AFTER_CLOSE)
    assert report.rejected == {
        "date_dupliquee": 1,
        "prix_non_positif": 1,
        "barre_incoherente": 1,
        "seance_sans_transaction": 1,
    }
    assert report.rows_in == 6 and report.rows_out == 2
    assert clean.loc["2024-01-08", "Close"] == pytest.approx(10.2)


def test_partial_session_is_dropped_before_close() -> None:
    raw = make_frame([("2024-01-09", 10, 11, 9, 10, 100), ("2024-01-10", 10, 10.5, 9.8, 10.1, 50)])
    clean, report = validate_ohlcv(raw, as_of=pd.Timestamp("2024-01-10 11:00"))
    assert list(clean.index) == [pd.Timestamp("2024-01-09")]
    assert report.rejected == {"seance_en_cours": 1}


def test_missing_column_raises() -> None:
    raw = make_frame([("2024-01-02", 10, 11, 9, 10.5, 100)]).drop(columns="Volume")
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate_ohlcv(raw)
