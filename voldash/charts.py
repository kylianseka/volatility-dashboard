"""Figures Plotly du dashboard (aucune logique de calcul ici)."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from voldash.config import TENSION_HIGH, TENSION_LOW
from voldash.estimators import ESTIMATOR_LABELS

PRICE_COLOR = "#1f3b5c"
YZ_COLOR = "#c0392b"
CC_COLOR = "#7f8c8d"
STRESS_COLOR = "rgba(192, 57, 43, 0.10)"
CALM_COLOR = "rgba(39, 174, 96, 0.10)"
BAND_COLOR = "rgba(31, 59, 92, 0.12)"
ESTIMATOR_COLORS: dict[str, str] = {
    "cc": "#95a5a6",
    "parkinson": "#2e86c1",
    "garman_klass": "#17a589",
    "rogers_satchell": "#d68910",
    "yang_zhang": "#c0392b",
}
LAYOUT = dict(template="plotly_white", hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))


def regime_figure(
    close: pd.Series,
    vol_yz: pd.Series,
    vol_cc: pd.Series,
    ratio: pd.Series,
    p10: float,
    p90: float,
    window: int,
) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.38, 0.38, 0.24],
        subplot_titles=("Cours de clôture ajusté", f"Volatilité annualisée ({window} j)", "Ratio de tension R = σYZ(21j) / σYZ(252j)"),
    )
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Cours", line=dict(color=PRICE_COLOR, width=1.5)), row=1, col=1)

    fig.add_hrect(y0=p90 * 100, y1=max(p90, float(vol_yz.max())) * 100, fillcolor=STRESS_COLOR, line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=p10 * 100, fillcolor=CALM_COLOR, line_width=0, row=2, col=1)
    fig.add_trace(go.Scatter(x=vol_cc.index, y=vol_cc * 100, name="Close-to-Close", line=dict(color=CC_COLOR, width=1, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=vol_yz.index, y=vol_yz * 100, name="Yang-Zhang", line=dict(color=YZ_COLOR, width=2)), row=2, col=1)
    fig.add_hline(y=p90 * 100, line_dash="dash", line_color=YZ_COLOR, annotation_text=f"P90 {p90:.0%}", annotation_position="top left", row=2, col=1)
    fig.add_hline(y=p10 * 100, line_dash="dash", line_color="#27ae60", annotation_text=f"P10 {p10:.0%}", annotation_position="bottom left", row=2, col=1)

    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio R", line=dict(color=PRICE_COLOR, width=1.5)), row=3, col=1)
    fig.add_hline(y=TENSION_HIGH, line_dash="dash", line_color=YZ_COLOR, row=3, col=1)
    fig.add_hline(y=TENSION_LOW, line_dash="dash", line_color="#27ae60", row=3, col=1)

    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(height=760, showlegend=True, legend=dict(orientation="h", y=1.06, x=0), **LAYOUT)
    return fig


def cone_figure(cone: pd.DataFrame) -> go.Figure:
    x = cone.index.astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cone["Maximum"], name="Max", line=dict(color=CC_COLOR, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=x, y=cone["Minimum"], name="Min", line=dict(color=CC_COLOR, dash="dot", width=1), fill="tonexty", fillcolor="rgba(127,140,141,0.06)"))
    fig.add_trace(go.Scatter(x=x, y=cone["P90"], name="P90", line=dict(color=PRICE_COLOR, width=1)))
    fig.add_trace(go.Scatter(x=x, y=cone["P10"], name="P10", line=dict(color=PRICE_COLOR, width=1), fill="tonexty", fillcolor=BAND_COLOR))
    fig.add_trace(go.Scatter(x=x, y=cone["Médiane"], name="Médiane", line=dict(color=PRICE_COLOR, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=cone["Actuelle"], name="Actuelle", mode="lines+markers", line=dict(color=YZ_COLOR, width=3), marker=dict(size=10)))
    fig.update_xaxes(type="log", tickvals=list(x), ticktext=[f"{h} j" for h in x], title_text="Horizon de calcul (jours ouvrés)")
    fig.update_yaxes(title_text="Volatilité Yang-Zhang annualisée (%)")
    fig.update_layout(height=480, legend=dict(orientation="h", y=1.08, x=0), **{**LAYOUT, "hovermode": "x"})
    return fig


def estimators_figure(estimators: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, label in ESTIMATOR_LABELS.items():
        width = 2.5 if key == "yang_zhang" else 1.2
        fig.add_trace(go.Scatter(x=estimators.index, y=estimators[key] * 100, name=label, line=dict(color=ESTIMATOR_COLORS[key], width=width)))
    fig.update_yaxes(title_text="Volatilité annualisée (%)")
    fig.update_layout(height=480, legend=dict(orientation="h", y=1.08, x=0), **LAYOUT)
    return fig


def comparison_figure(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=values.index, y=values * 100, name=name, line=dict(width=1.5)))
    fig.update_yaxes(title_text="σYZ 21 j annualisée (%)")
    fig.update_layout(height=420, legend=dict(orientation="h", y=1.1, x=0), **LAYOUT)
    return fig
