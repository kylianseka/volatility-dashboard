# Démarche de travail

> Test technique Sanso Longchamp AM : dashboard d'analyse de volatilité d'une action européenne.
> Données arrêtées à la séance du 24/11/2026.

## 1. Comprendre le besoin

Le sujet pose deux questions de gérant :

| Question | Réponse dans le dashboard |
|---|---|
| Comment la volatilité a-t-elle évolué ? | Série σ Yang-Zhang avec bandes P10/P90, ratio de tension 21 j / 252 j dans le temps |
| La situation actuelle est-elle particulière ? | Rang de percentile sur 252 séances, cône de volatilité, badge de régime, « Lecture rapide » |

J'ai choisi de **répondre d'abord à ces deux questions, en haut de page** (KPI et lecture rapide en texte). Les onglets servent à creuser ensuite. Un gérant doit avoir l'essentiel sans cliquer.

## 2. Choix structurants

| Sujet | Choix | Pourquoi |
|---|---|---|
| Source | Yahoo Finance (`yfinance`), cours **ajustés** | Gratuit, sans clé, couvre les actions européennes. L'ajustement évite qu'un split apparaisse comme un faux gap. |
| Période | 5 ans par défaut (1 à 10 ans ou max au choix) | Le cône a besoin de 3 à 5 ans pour que les quantiles 252 j aient un sens. |
| Estimateur de référence | **Yang-Zhang** | Seul estimateur qui intègre à la fois les gaps d'ouverture et l'amplitude intraday, et qui reste robuste à la tendance. |
| Indicateurs complémentaires | Ratio R = σ21/σ252, rang de percentile 252 j, part de variance overnight, comparaison avec Close-to-Close | Ils transforment un chiffre de vol en diagnostic : est-ce élevé, est-ce récent, d'où vient le risque ? |
| Techno | Python, pandas/NumPy, Streamlit, Plotly | Le plus court chemin vers une application testable avec `streamlit run` ; calcul vectorisé (environ 30 ms pour 5 ans). |
| Hors périmètre | Modèles prédictifs (HAR, LSTM, Transformers), données fondamentales FMP | Le besoin est descriptif (« simple »). Une prévision non validée ajouterait du bruit, pas de la décision. |

## 3. Architecture

```
yfinance ──► data.validate_ohlcv ──► cache parquet ──► estimators / regime ──► charts ──► app.py
             (rejets comptés et loggés)   (repli hors ligne)   (fonctions pures, testées)
```

- **Séparation des responsabilités** : le calcul (`estimators.py`, `regime.py`) ne connaît pas Streamlit. Il est testable sans interface ni réseau.
- **Qualité des données avant calcul** : contrôle du schéma, puis rejet des doublons, des prix manquants ou ≤ 0, des barres incohérentes (High < max(O, C)…), des séances sans transaction et de **la séance du jour non clôturée**. Chaque rejet est compté, loggé et affiché dans l'onglet « Méthode & qualité ». Aucune ligne ne disparaît en silence.
- **Pipeline idempotent** : chaque chargement ré-écrit le même snapshot parquet. Si Yahoo est indisponible, le dernier snapshot validé est utilisé et la source est signalée dans l'en-tête.
- **Cache** : `@st.cache_data(ttl=3600)` sur l'ingestion uniquement. Les calculs sont assez rapides pour ne pas être cachés.

## 4. Usage de l'IA et ses limites

Le code a été écrit avec un assistant IA (Claude Code), à partir du sujet et d'un « plan directeur » lui-même généré par IA. Le point clé de la démarche : **tout ce que l'IA affirme a été vérifié avant d'être utilisé**. Voici ce que la vérification a fait apparaître dans le plan directeur.

| # | Affirmation ou code du plan IA | Constat | Correction |
|---|---|---|---|
| 1 | Tickers `ENR1n.DE`, `S92G.DE` | Codes Reuters (RIC) : **Yahoo ne renvoie aucune donnée** | `ENR.DE`, `S92.DE`, vérifiés par téléchargement |
| 2 | Rang de percentile « sur 252 jours » | Le code le calculait **sur tout l'historique chargé** | Fenêtre glissante stricte des 252 séances précédentes (`rolling_percentile_rank`) |
| 3 | Ratio de tension σ21/σ252 | Le code utilisait **la fenêtre du curseur** au lieu de 21 j, alors que le KPI affichait « 21j / 252j » | Fenêtres 21/252 fixes, indépendantes du curseur |
| 4 | Données du jour | La **barre de la séance en cours** (incomplète) était incluse : amplitude et volume sous-estimés | Exclue avant 17 h 45 (heure de Paris) |
| 5 | « ln(H/L) : division par zéro si H = L » | Faux : ln(1) = 0 est fini. Le vrai risque est un prix ≤ 0 ou une barre incohérente. | Validation des barres, plus un test dédié à la séance plate |
| 6 | Valeurs négatives de Rogers-Satchell | Mathématiquement impossible si la barre est cohérente (chaque terme ≥ 0) | Garanti en amont par la validation |
| 7 | Étude de cas : σYZ 38,4 % / 56,2 %, « sous-estimation **systématique** par σCC »… | Chiffres **non reproductibles**. Sur données réelles, σCC est **au-dessus** de σYZ pour Siemens Energy (79,0 % contre 67,7 %). | Chiffres recalculés (voir §5), affirmation retirée |
| 8 | Benchmark DL 2000–2025, tests de Diebold-Mariano, p < 0,05 | Aucune source, aucun code : non vérifiable | Non repris |
| 9 | API Streamlit (`refresh_mode`, `use_container_width`) | `refresh_mode` existe bien dans la version installée. `use_container_width` est déprécié. | Signatures vérifiées sur Streamlit 1.67 installé |

À retenir : **ces erreurs ne font pas planter le code**. Les erreurs 2, 3 et 4 produisent des chiffres plausibles mais faux, présentés au gérant avec la même assurance que des chiffres justes. Seules la relecture critique et des tests sur données à volatilité connue les détectent.

Ce qui a été fait pour s'en protéger :

- **Tests sur séries simulées à volatilité connue** (mouvement brownien avec gaps) : Yang-Zhang retrouve la vol totale ; Parkinson, Garman-Klass et Rogers-Satchell ne retrouvent que la vol intraday ; Rogers-Satchell reste juste avec une forte tendance alors que Parkinson surestime.
- **Test anti-lookahead** : le calcul sur un historique tronqué doit donner exactement les mêmes valeurs passées.
- **Vérification des API** par lecture des signatures installées, pas de mémoire.

## 5. Résultats sur les titres du sujet (24/11/2026, fenêtres 21 j / 252 j)

| Indicateur | Siemens Energy | SMA Solar |
|---|---|---|
| σ Yang-Zhang 21 j | 67,7 % | 79,5 % |
| σ Close-to-Close 21 j | 79,0 % | 70,0 % |
| σ Yang-Zhang 252 j | 54,5 % | 69,2 % |
| Ratio R (21 j / 252 j) | **1,24 → stress** | 1,15 → normal haut |
| Rang percentile 252 j | 84 % | 77 % |
| Part de variance overnight | 40 % | 33 % |

Lecture :

- **SMA Solar** a la volatilité absolue la plus forte, mais elle est proche de sa norme annuelle (R = 1,15).
- **Siemens Energy** connaît un **choc récent** (R > 1,20) : selon les règles du plan, réduire le poids de la ligne.
- Les deux titres ont une part importante de risque de gap (un tiers à 40 % de la variance) : un stop-loss serré à l'ouverture serait exécuté avec glissement.

Les valeurs bougent chaque jour : le dashboard les recalcule à chaque chargement.

## 6. Limites connues et pistes

- **Source** : Yahoo Finance est une API non officielle, sans SLA. En production, il faudrait un fournisseur contractuel (Bloomberg, Refinitiv, FMP payant) derrière la même interface `load_market_data`.
- **Données quotidiennes uniquement** : pas de volatilité réalisée intraday (5 min), qui serait plus précise.
- **Quantiles sur fenêtres chevauchantes** : les observations du cône sont autocorrélées. Les percentiles sont descriptifs, pas des probabilités.
- **Seuils 1,20 / 0,80 / P90** : ce sont des conventions du plan, pas des seuils calibrés. Ils seraient à valider par backtest sur le portefeuille.
- **Pistes** : comparaison avec la volatilité implicite (prime de risque), modèle HAR-RV pour une prévision à 1–22 j évaluée en QLIKE, alertes automatiques (e-mail ou Teams) sur franchissement de R > 1,20.
