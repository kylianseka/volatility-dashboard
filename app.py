"""Dashboard Streamlit d'analyse de volatilité — point d'entrée : `streamlit run app.py`."""

import logging
import re

import numpy as np
import pandas as pd
import streamlit as st

from voldash.charts import comparison_figure, cone_figure, estimators_figure, regime_figure
from voldash.config import (
    DEFAULT_LOOKBACK,
    LOOKBACK_PERIODS,
    PRESET_TICKERS,
    SHORT_WINDOW,
    TENSION_HIGH,
)
from voldash.data import DataUnavailableError, MarketData, load_market_data
from voldash.estimators import ESTIMATOR_LABELS, all_estimators, yang_zhang
from voldash.regime import Regime, Snapshot, rolling_percentile_rank, snapshot, tension_ratio, volatility_cone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
REGIME_ICONS: dict[Regime, str] = {
    Regime.STRESS: ":red[● Stress]",
    Regime.COMPRESSION: ":green[● Compression]",
    Regime.NORMAL: ":blue[● Normal]",
    Regime.UNKNOWN: ":gray[● Historique insuffisant]",
}

st.set_page_config(page_title="Volatilité Actions", page_icon=":material/monitoring:", layout="wide")


@st.cache_data(ttl=3600, max_entries=100, show_spinner="Chargement des cours…")
def get_market_data(ticker: str, period: str) -> MarketData:
    return load_market_data(ticker, period)


def pct(value: float, digits: int = 1) -> str:
    return "n/d" if np.isnan(value) else f"{value * 100:.{digits}f} %"


def interpretation(snap: Snapshot, cone: pd.DataFrame) -> list[str]:
    """Lecture en langage gérant des indicateurs (règles du plan directeur §8)."""
    lines = [
        f"La volatilité Yang-Zhang 21 j est de **{pct(snap.vol_yz_21)}**, plus élevée que "
        f"**{snap.percentile_rank:.0f} %** des séances de l'année écoulée."
    ]
    if not np.isnan(snap.tension_ratio):
        if snap.tension_ratio > TENSION_HIGH:
            lines.append(
                f"Ratio court/long terme à **{snap.tension_ratio:.2f}** (> 1,20) : choc de volatilité récent "
                "non encore absorbé par la moyenne annuelle → envisager une **réduction de la taille de ligne**."
            )
        elif snap.tension_ratio < 0.80:
            lines.append(
                f"Ratio court/long terme à **{snap.tension_ratio:.2f}** (< 0,80) : compression anormale, "
                "souvent suivie d'une reprise de la variance."
            )
        else:
            lines.append(f"Ratio court/long terme à **{snap.tension_ratio:.2f}** : pas de rupture de régime.")
    if SHORT_WINDOW in cone.index and snap.vol_yz_21 * 100 > cone.loc[SHORT_WINDOW, "P90"]:
        lines.append("La vol 21 j dépasse le **P90 historique** du cône → **élargir les stop-loss**.")
    gap = snap.vol_yz_21 - snap.vol_cc_21
    lines.append(
        f"Gaps d'ouverture : **{pct(snap.overnight_share, 0)}** de la variance 21 j. "
        f"Écart Yang-Zhang − Close-to-Close : **{gap * 100:+.1f} pts**"
        + (" (le Close-to-Close sous-estime le risque)." if gap > 0 else " (le Close-to-Close le surestime ici).")
    )
    return lines


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Paramètres")
    preset_name = st.selectbox("Action", list(PRESET_TICKERS), index=1)
    custom = st.text_input("…ou ticker Yahoo Finance", placeholder="ex. SAP.DE, ASML.AS").strip().upper()
    lookback = st.selectbox("Historique", LOOKBACK_PERIODS, index=LOOKBACK_PERIODS.index(DEFAULT_LOOKBACK))
    window = st.slider("Fenêtre de volatilité (jours ouvrés)", min_value=5, max_value=126, value=SHORT_WINDOW)
    st.caption(
        "Le ratio de tension et le rang de percentile restent calculés sur 21 j / 252 j "
        "quelle que soit la fenêtre choisie, pour rester comparables."
    )

if custom and not TICKER_PATTERN.match(custom):
    st.error("Ticker invalide : lettres, chiffres et `.-^=` uniquement.")
    st.stop()

ticker = custom or PRESET_TICKERS[preset_name]
display_name = custom or preset_name

try:
    market = get_market_data(ticker, lookback)
except DataUnavailableError as err:
    st.error(str(err))
    st.stop()

prices = market.prices
if len(prices) < 300:
    st.warning("Historique court (< 300 séances) : le ratio 252 j et le rang de percentile seront incomplets.")

snap = snapshot(prices)
estimators = all_estimators(prices, window)
cone = volatility_cone(prices)
yz_window = estimators["yang_zhang"]
yz_clean = yz_window.dropna()
p10, p90 = (float(np.percentile(yz_clean, q)) for q in (10, 90))

# ---------------------------------------------------------------- En-tête
st.title(f"{display_name}")
st.markdown(
    f"`{ticker}` · dernière séance clôturée : **{snap.date:%d/%m/%Y}** · {len(prices)} séances · "
    f"source : {'Yahoo Finance' if market.source == 'yahoo' else 'cache local (hors ligne)'} · "
    f"régime : {REGIME_ICONS[snap.regime]}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dernier cours", f"{snap.close:,.2f}", f"{snap.daily_return * 100:+.2f} %", border=True)
k2.metric(
    "Vol. YZ 21 j", pct(snap.vol_yz_21), f"{(snap.vol_yz_21 - snap.vol_cc_21) * 100:+.1f} pts vs CC",
    delta_color="off", border=True, help="Estimateur de référence : intègre gaps overnight, amplitude intraday, invariant au drift.",
)
k3.metric(
    "Rang 252 j", f"{snap.percentile_rank:.0f} %" if not np.isnan(snap.percentile_rank) else "n/d",
    border=True, help="Part des 252 dernières séances où la vol 21 j était inférieure ou égale au niveau actuel.",
)
k4.metric(
    "Ratio 21 j / 252 j", f"{snap.tension_ratio:.2f}" if not np.isnan(snap.tension_ratio) else "n/d",
    border=True, help="> 1,20 : choc de volatilité · < 0,80 : compression anormale.",
)
k5.metric(
    "Part overnight", pct(snap.overnight_share, 0), border=True,
    help="σ²(ouverture vs clôture veille) / σ²YZ sur 21 j : poids du risque de gap.",
)

with st.container(border=True):
    st.markdown("**Lecture rapide**")
    for line in interpretation(snap, cone):
        st.markdown(f"- {line}")

tab_regime, tab_cone, tab_est, tab_comp, tab_method = st.tabs(
    ["Évolution & régimes", "Cône de volatilité", "Comparatif estimateurs", "Comparaison des titres", "Méthode & qualité"]
)

with tab_regime:
    st.plotly_chart(
        regime_figure(prices["Close"], yz_window, estimators["cc"], tension_ratio(prices), p10, p90, window),
        key="regime",
    )
    st.caption(
        f"Bandes P10/P90 calculées sur toute la période affichée ({lookback}). "
        "Zone rouge : stress historique · zone verte : accalmie."
    )

with tab_cone:
    left, right = st.columns([3, 2])
    left.plotly_chart(cone_figure(cone), key="cone")
    right.markdown("**Distribution empirique de σYZ par horizon (%)**")
    right.dataframe(cone.round(1), column_config={"Rang (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
    right.caption(
        "Le cône se resserre avec l'horizon : l'erreur d'échantillonnage décroît (≈ σ/√2N) et la variance "
        "revient vers sa moyenne. Un point rouge au-dessus de P90 sur les horizons courts signale un stress anormal."
    )

with tab_est:
    st.plotly_chart(estimators_figure(estimators.dropna(how="all")), key="estimators")
    last = estimators.dropna().iloc[-1] if not estimators.dropna().empty else pd.Series(dtype=float)
    table = pd.DataFrame(
        {
            "Dernière valeur (%)": (last * 100).round(1),
            "Données": ["C", "H, L", "O, H, L, C", "O, H, L, C", "C veille, O, H, L, C"],
            "Gaps overnight": ["Oui", "Non", "Non", "Non", "Oui"],
            "Robuste au drift": ["Non", "Non", "Non", "Oui", "Oui"],
        }
    ).rename(index=ESTIMATOR_LABELS)
    st.dataframe(table)
    st.caption(
        "Parkinson, Garman-Klass et Rogers-Satchell ne voient que la séance : ils sous-estiment le risque total "
        "d'un titre exposé aux gaps d'ouverture. Yang-Zhang combine overnight, open-to-close et Rogers-Satchell."
    )

with tab_comp:
    st.markdown("Indicateurs à la dernière séance pour les titres pré-configurés (fenêtres 21 j / 252 j).")
    rows: dict[str, dict[str, object]] = {}
    curves: dict[str, pd.Series] = {}
    for name, symbol in PRESET_TICKERS.items():
        try:
            other = get_market_data(symbol, lookback).prices
        except DataUnavailableError:
            continue
        s = snapshot(other)
        rows[name] = {
            "Ticker": symbol,
            "σYZ 21 j (%)": round(s.vol_yz_21 * 100, 1),
            "σCC 21 j (%)": round(s.vol_cc_21 * 100, 1),
            "σYZ 252 j (%)": round(s.vol_yz_252 * 100, 1),
            "Ratio R": round(s.tension_ratio, 2),
            "Rang 252 j (%)": round(s.percentile_rank, 0),
            "Part overnight (%)": round(s.overnight_share * 100, 0),
            "Régime": s.regime.value,
        }
        curves[name] = yang_zhang(other, SHORT_WINDOW).volatility.dropna()
    st.dataframe(pd.DataFrame.from_dict(rows, orient="index"))
    st.plotly_chart(comparison_figure(curves), key="comparison")

with tab_method:
    st.markdown(
        """
**Source** : Yahoo Finance via `yfinance`, cours quotidiens **ajustés** des splits et dividendes
(sinon un split crée un faux gap). Cache Streamlit 1 h + snapshot parquet local en repli hors ligne.

**Estimateurs** (annualisation √252, fenêtres glissantes strictement rétrospectives) :
- Close-to-Close : écart-type des rendements log de clôture.
- Parkinson : amplitude haut/bas, ignore gaps et drift.
- Garman-Klass : amplitude + open-to-close, ignore gaps.
- Rogers-Satchell : invariant au drift, ignore gaps.
- **Yang-Zhang** : σ²O + k·σ²C + (1−k)·σ²RS, avec k = 0,34 / (1,34 + (N+1)/(N−1)) — référence retenue.

**Régime** : ratio R = σYZ(21 j)/σYZ(252 j) ; rang de percentile de la vol 21 j sur les 252 séances
précédentes ; cône = quantiles de σYZ sur 10/21/63/126/252 j sur tout l'historique chargé.
"""
    )
    report = market.report
    st.markdown("**Contrôle qualité des données chargées**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes reçues", report.rows_in)
    c2.metric("Lignes conservées", report.rows_out)
    c3.metric("Lignes rejetées", report.rows_rejected)
    st.write("Rejets :", report.rejected or "aucun", " · Alertes (conservées) :", report.flagged or "aucune")
    st.download_button(
        "Télécharger les séries calculées (CSV)",
        data=prices.join(estimators.add_prefix("vol_")).assign(
            ratio_tension=tension_ratio(prices),
            rang_percentile_252=rolling_percentile_rank(yang_zhang(prices, SHORT_WINDOW).volatility),
        ).to_csv().encode("utf-8"),
        file_name=f"volatilite_{ticker}_{lookback}.csv",
        mime="text/csv",
    )
