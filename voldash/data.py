"""Ingestion des cours OHLCV, contrôle qualité et cache local.

Le pipeline est idempotent : chaque exécution re-télécharge l'historique complet,
le valide, puis réécrit le même fichier parquet. Si le fournisseur est
indisponible, le dernier snapshot validé sert de repli (mode hors ligne).
"""

import logging
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
MARKET_TZ = "Europe/Paris"
SESSION_CLOSE = time(17, 45)


class DataUnavailableError(RuntimeError):
    """Aucune donnée exploitable (ni fournisseur, ni cache)."""


@dataclass(frozen=True)
class QualityReport:
    """Bilan du contrôle qualité : rien n'est supprimé sans être compté."""

    rows_in: int
    rows_out: int
    rejected: dict[str, int] = field(default_factory=dict)
    flagged: dict[str, int] = field(default_factory=dict)

    @property
    def rows_rejected(self) -> int:
        return self.rows_in - self.rows_out


@dataclass(frozen=True)
class MarketData:
    ticker: str
    prices: pd.DataFrame
    report: QualityReport
    source: str  # "yahoo" ou "cache"


def validate_ohlcv(raw: pd.DataFrame, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, QualityReport]:
    """Valide le schéma et la cohérence des barres OHLC.

    Règles de rejet (chaque rejet est compté) :
    - doublons de date, prix manquants ou non strictement positifs ;
    - barres incohérentes (High < max(Open, Close) ou Low > min(Open, Close)) ;
    - séances sans transaction (volume nul et O = H = L = C) ;
    - séance du jour non clôturée (barre partielle qui biaiserait l'amplitude).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex.")

    df = raw.loc[:, list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df.sort_index()
    rows_in = len(df)
    rejected: dict[str, int] = {}

    def reject(name: str, mask: pd.Series) -> None:
        nonlocal df
        count = int(mask.sum())
        if count:
            rejected[name] = count
            df = df.loc[~mask]

    reject("date_dupliquee", pd.Series(df.index.duplicated(keep="last"), index=df.index))
    prices = df[["Open", "High", "Low", "Close"]]
    reject("prix_manquant", prices.isna().any(axis=1))
    prices = df[["Open", "High", "Low", "Close"]]
    reject("prix_non_positif", (prices <= 0).any(axis=1))
    incoherent = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    reject("barre_incoherente", incoherent)
    flat = (df["High"] == df["Low"]) & (df["Open"] == df["Close"]) & (df["Open"] == df["High"])
    reject("seance_sans_transaction", flat & (df["Volume"].fillna(0) == 0))

    now = as_of if as_of is not None else pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None)
    if now.time() < SESSION_CLOSE:
        reject("seance_en_cours", pd.Series(df.index.normalize() >= now.normalize(), index=df.index))

    flagged = {
        "amplitude_nulle": int((df["High"] == df["Low"]).sum()),
        "volume_nul": int((df["Volume"] == 0).sum()),
        "gap_superieur_20pct": int((np.log(df["Open"] / df["Close"].shift(1)).abs() > np.log(1.2)).sum()),
    }
    flagged = {name: count for name, count in flagged.items() if count}

    report = QualityReport(rows_in=rows_in, rows_out=len(df), rejected=rejected, flagged=flagged)
    logger.info(
        "Validation OHLCV : %d lignes lues, %d conservées, rejets=%s, alertes=%s",
        report.rows_in, report.rows_out, report.rejected, report.flagged,
    )
    return df, report


def _cache_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"{safe}_{period}.parquet"


def _download(ticker: str, period: str) -> pd.DataFrame:
    # auto_adjust=True : cours ajustés des splits et dividendes, sinon un split
    # apparaîtrait comme un faux gap d'ouverture.
    return yf.download(
        ticker, period=period, interval="1d", auto_adjust=True,
        progress=False, multi_level_index=False, threads=False,
    )


def load_market_data(ticker: str, period: str) -> MarketData:
    """Télécharge, valide et met en cache ; repli sur le cache si le réseau échoue."""
    ticker = ticker.strip().upper()
    cache_file = _cache_path(ticker, period)

    try:
        raw = _download(ticker, period)
    except Exception as err:  # yfinance lève des exceptions hétérogènes
        logger.warning("Téléchargement échoué pour %s : %s", ticker, err)
        raw = pd.DataFrame()

    if not raw.empty:
        prices, report = validate_ohlcv(raw)
        if prices.empty:
            raise DataUnavailableError(f"Aucune séance valide pour '{ticker}'.")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_file)
        return MarketData(ticker=ticker, prices=prices, report=report, source="yahoo")

    if cache_file.exists():
        logger.warning("Utilisation du cache local %s", cache_file.name)
        prices, report = validate_ohlcv(pd.read_parquet(cache_file))
        return MarketData(ticker=ticker, prices=prices, report=report, source="cache")

    raise DataUnavailableError(f"Aucune donnée disponible pour le symbole '{ticker}'.")
