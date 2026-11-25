# Dashboard d'analyse de volatilité — Gestion actions

Application Streamlit qui permet à un gérant de comprendre en quelques secondes :

1. **comment la volatilité d'un titre a évolué** (série Yang-Zhang, bandes P10/P90, ratio de tension) ;
2. **si la situation actuelle est inhabituelle** par rapport à son historique (rang de percentile 252 j, cône de volatilité).

Titres pré-configurés : Siemens Energy (`ENR.DE`), SMA Solar (`S92.DE`), Airbus (`AIR.PA`), LVMH (`MC.PA`). Le champ libre accepte n'importe quel ticker Yahoo Finance.

La démarche, les choix et l'usage de l'IA sont détaillés dans **[docs/DEMARCHE.md](docs/DEMARCHE.md)**.

## Lancer l'application

Prérequis : Python 3.12+, accès internet (Yahoo Finance). Aucune clé d'API n'est nécessaire.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L'application s'ouvre sur <http://localhost:8501>. Sous macOS / Linux, remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python`.

## Lancer les tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Ils sont 22 et valident les estimateurs sur des séries simulées dont la volatilité est connue : annualisation √252, absence de biais d'anticipation, robustesse au drift, part overnight, séance plate. Ils couvrent aussi le rang de percentile, la classification de régime et le contrôle qualité des données.

## Contenu du dashboard

| Zone | Ce qu'elle montre |
|---|---|
| **KPI** | Dernier cours, σ Yang-Zhang 21 j (et écart avec Close-to-Close), rang de percentile 252 j, ratio 21 j / 252 j, part de variance overnight |
| **Lecture rapide** | Interprétation automatique en langage gérant (réduction de ligne si R > 1,20, élargissement des stops si σ > P90) |
| Évolution & régimes | Prix, σYZ vs σCC avec zones de stress / accalmie, ratio de tension dans le temps |
| Cône de volatilité | Min / P10 / médiane / P90 / max de σYZ sur 10–252 j et position actuelle |
| Comparatif estimateurs | Les 5 estimateurs superposés et leurs propriétés |
| Comparaison des titres | Tableau et courbes σYZ 21 j des 4 titres |
| Méthode & qualité | Formules, bilan du contrôle qualité, export CSV des séries calculées |

## Structure

```
app.py                  Interface Streamlit (présentation uniquement)
voldash/
  config.py             Constantes métier (252 j, horizons, seuils, tickers)
  data.py               Ingestion yfinance, validation OHLCV, cache parquet
  estimators.py         Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
  regime.py             Ratio de tension, rang de percentile, cône, classification
  charts.py             Figures Plotly
tests/                  Tests pytest (données simulées, sans réseau)
docs/DEMARCHE.md        Démarche, choix, usage de l'IA, limites
```
