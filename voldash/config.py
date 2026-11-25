"""Constantes métier partagées par le moteur et le dashboard."""

from typing import Final

TRADING_DAYS: Final[int] = 252

CONE_HORIZONS: Final[tuple[int, ...]] = (10, 21, 63, 126, 252)

SHORT_WINDOW: Final[int] = 21
LONG_WINDOW: Final[int] = 252

TENSION_HIGH: Final[float] = 1.20
TENSION_LOW: Final[float] = 0.80
PERCENTILE_STRESS: Final[float] = 90.0
PERCENTILE_CALM: Final[float] = 10.0

# Tickers Yahoo Finance vérifiés (les codes ENR1n.DE / S92G.DE du plan directeur
# sont des RIC Reuters et ne renvoient aucune donnée sur Yahoo).
PRESET_TICKERS: Final[dict[str, str]] = {
    "Siemens Energy AG": "ENR.DE",
    "SMA Solar Technology AG": "S92.DE",
    "Airbus SE": "AIR.PA",
    "LVMH SE": "MC.PA",
}

LOOKBACK_PERIODS: Final[tuple[str, ...]] = ("1y", "3y", "5y", "10y", "max")
DEFAULT_LOOKBACK: Final[str] = "5y"
