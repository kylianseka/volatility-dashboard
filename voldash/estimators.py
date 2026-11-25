"""Estimateurs de volatilité réalisée annualisée sur fenêtre glissante.

Toutes les fenêtres sont rétrospectives (jamais `center=True`) : la valeur
en t n'utilise que les séances <= t. Le facteur d'annualisation est 252
(jours ouvrés actions), pas 365.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from voldash.config import TRADING_DAYS

LN2 = float(np.log(2.0))


def _log_ratios(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "hl": np.log(df["High"] / df["Low"]),
        "ho": np.log(df["High"] / df["Open"]),
        "lo": np.log(df["Low"] / df["Open"]),
        "co": np.log(df["Close"] / df["Open"]),
        # Gap overnight : ouverture du jour vs clôture de la veille.
        "oc_prev": np.log(df["Open"] / df["Close"].shift(1)),
    }


def _annualize(daily_variance: pd.Series) -> pd.Series:
    # Garman-Klass peut produire une variance moyenne légèrement négative sur
    # des séries très calmes : on plafonne à 0 plutôt que de produire des NaN.
    return np.sqrt(daily_variance.clip(lower=0.0) * TRADING_DAYS)


def close_to_close(df: pd.DataFrame, window: int) -> pd.Series:
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    return _annualize(log_returns.rolling(window).var(ddof=1))


def parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    hl = _log_ratios(df)["hl"]
    return _annualize((hl**2 / (4.0 * LN2)).rolling(window).mean())


def garman_klass(df: pd.DataFrame, window: int) -> pd.Series:
    r = _log_ratios(df)
    daily = 0.5 * r["hl"] ** 2 - (2.0 * LN2 - 1.0) * r["co"] ** 2
    return _annualize(daily.rolling(window).mean())


def _rogers_satchell_daily(r: dict[str, pd.Series]) -> pd.Series:
    return r["ho"] * (r["ho"] - r["co"]) + r["lo"] * (r["lo"] - r["co"])


def rogers_satchell(df: pd.DataFrame, window: int) -> pd.Series:
    return _annualize(_rogers_satchell_daily(_log_ratios(df)).rolling(window).mean())


@dataclass(frozen=True)
class YangZhang:
    """Volatilité Yang-Zhang et ses composantes de variance (annualisées)."""

    volatility: pd.Series
    overnight_var: pd.Series
    open_close_var: pd.Series
    rogers_satchell_var: pd.Series
    k: float

    @property
    def overnight_share(self) -> pd.Series:
        """Part de la variance totale expliquée par les gaps d'ouverture."""
        total = self.volatility**2
        return (self.overnight_var / total).where(total > 0)


def yang_zhang(df: pd.DataFrame, window: int) -> YangZhang:
    if window < 2:
        raise ValueError("La fenêtre Yang-Zhang doit être >= 2.")
    r = _log_ratios(df)
    overnight = r["oc_prev"].rolling(window).var(ddof=1) * TRADING_DAYS
    open_close = r["co"].rolling(window).var(ddof=1) * TRADING_DAYS
    rs = _rogers_satchell_daily(r).rolling(window).mean() * TRADING_DAYS
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    total = overnight + k * open_close + (1.0 - k) * rs
    return YangZhang(
        volatility=np.sqrt(total.clip(lower=0.0)),
        overnight_var=overnight,
        open_close_var=open_close,
        rogers_satchell_var=rs,
        k=k,
    )


ESTIMATOR_LABELS: dict[str, str] = {
    "cc": "Close-to-Close",
    "parkinson": "Parkinson",
    "garman_klass": "Garman-Klass",
    "rogers_satchell": "Rogers-Satchell",
    "yang_zhang": "Yang-Zhang",
}


def all_estimators(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Les 5 estimateurs sur la même fenêtre, en décimal (0.25 = 25 %)."""
    return pd.DataFrame(
        {
            "cc": close_to_close(df, window),
            "parkinson": parkinson(df, window),
            "garman_klass": garman_klass(df, window),
            "rogers_satchell": rogers_satchell(df, window),
            "yang_zhang": yang_zhang(df, window).volatility,
        },
        index=df.index,
    )
