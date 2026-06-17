# MPP Predictor ⚽📈

Modèle prédictif pour **Mon Petit Prono** (Coupe du Monde). L'objectif n'est
pas de deviner le vainqueur théorique d'un match, mais de **maximiser
l'espérance de points** selon le barème MPP — en exploitant des variables
physiques sous-estimées par les bookmakers (fatigue, manque de rythme) et la
psychologie de la communauté de parieurs.

> **Philosophie : Rasoir d'Ockham.** Une structure simple et robuste, peu de
> paramètres (tous justifiés et externalisés en config), pas de surapprentissage.
> Le modèle s'articule autour de deux **Super-Index dynamiques** par équipe
> (Attaque & Défense) qui alimentent une **loi de Poisson**, suivie d'un module
> de **décision** qui optimise l'espérance de points.

## Architecture du pipeline

```
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│  Raw Data   │──▶│  Data Tank   │──▶│ Moteur Poisson  │──▶│  Décision MPP    │
│ (ingestion) │   │ (Super-Index)│   │ (matrice scores)│   │ (max E[points])  │
└─────────────┘   └──────────────┘   └─────────────────┘   └──────────────────┘
   scores, météo     IAG / IDG par      P(i,j) scores         prono recommandé
   minutes club       équipe/match        exacts                ≠ score probable
```

| Couche | Module | Rôle |
|--------|--------|------|
| **1. Raw Data** | `ingestion/` | Accès aux sources (scores, minutes club, météo) derrière une interface stable |
| **2. Data Tank** | `features/` | Consolidation en Index d'Attaque (IAG) et de Défense (IDG) dynamiques |
| **3. Moteur** | `model/` | Index → espérances de buts (λ) → matrice de scores via Poisson |
| **4. Décision** | `decision/` | Choix du prono qui maximise l'espérance de points MPP |

## L'Index d'Attaque Global (IAG)

Combinaison linéaire pondérée et normalisée de **3 critères** :

```
IAG = w_D · D  +  w_F · F  +  w_C · C        (w_D + w_F + w_C = 1)
```

- **D — Dynamique offensive récente** *(poids 0.50)* : buts marqués sur une
  fenêtre glissante, amortis par récence (`λ^rang`) et pondérés par la force
  adverse (Elo). Le signal brut de production offensive.

- **F — Fraîcheur des cadres** *(poids 0.30)* : courbe en U inversé sur la
  charge annuelle des stars offensives. Une seule gaussienne capture le
  sous-régime (manque de rythme) **et** le surrégime (fatigue) :

  ```
  f(m) = exp( −(m − m_opt)² / (2σ²) )      m_opt ≈ 38 matchs, σ ≈ 15
  ```

  C'est l'avantage informationnel : les bookmakers sous-pondèrent la fatigue
  accumulée club → sélection.

- **C — Contexte** *(poids 0.20)* : multiplicateur borné (météo, repos
  différentiel, enjeu de poule). Volontairement simple.

L'Index de Défense suit la structure miroir. L'espérance de buts d'une équipe
combine son IAG et l'IDG adverse :
`λ = base · IAG · IDG_adverse`.

## Le module de décision : le vrai edge

Sur MPP, le score le **plus probable** n'est pas toujours le **plus rentable**.
Pour chaque pronostic candidat `(h, a)`, on calcule :

```
E[points | (h,a)] = Σ P(i,j) · points(prono=(h,a), réel=(i,j))
```

et on prend l'argmax. Un nul stratégique ou un score serré peut battre un gros
score populaire. Un hook `popularity_penalty` permet d'intégrer la psychologie
de la communauté (décourager les scores trop consensuels).

## Installation

```bash
git clone https://github.com/<toi>/mpp-predictor.git
cd mpp-predictor
pip install -e ".[dev]"
```

## Démarrage rapide

```bash
# 1. Installer
pip install -e ".[dev]"

# 2. Télécharger le dataset de résultats (Windows PowerShell : voir plus bas)
curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv \
    -o data/raw/results.csv

# 3. Prédire un vrai match (1er = domicile, 2e = extérieur, noms en anglais)
python -m examples.predict_match "Portugal" "DR Congo"

# 4. Tester la performance sur les vraies Coupes du Monde
python -m examples.run_backtest

# 5. Recalibrer les paramètres sur le backtest
python -m examples.calibrate

# 6. (option) Activer le critère fatigue pour un match
python -m examples.make_fatigue_template "Portugal" "DR Congo"
#    puis remplir data/fatigue_overrides.csv et relancer predict_match

# Lancer les tests
pytest
```

## Interface web

Le projet inclut une page web (dossier `web/`) pour pronostiquer sans terminal :

```bash
pip install -e ".[web]"
python -m web.app          # puis ouvrir http://localhost:5000
```

Pour la mettre **en ligne** (Render, Railway), voir [`DEPLOYMENT.md`](DEPLOYMENT.md).
À noter : Netlify / GitHub Pages ne conviennent pas (pas de backend Python).


Sous **Windows PowerShell**, remplacer la commande de téléchargement par :
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" -OutFile "data\raw\results.csv"
```


## Sources de données

| Critère | Source recommandée | Accès |
|---------|-------------------|-------|
| **D** (résultats) | football-data.org + eloratings.net | API gratuite / scraping |
| **F** (minutes club) | Dataset Kaggle FBref `players_data` | CSV hebdo, sans clé |
| **C** (météo) | Open-Meteo | API gratuite, sans clé |

> Voir `src/mpp_predictor/ingestion/loaders.py` pour les détails et la stratégie
> MVP. On commence simple (CSV pour F), on enrichit ensuite.

## Configuration

Tous les paramètres (poids, fenêtres, sweet spot de fatigue, barème MPP) vivent
dans [`config/params.yaml`](config/params.yaml). Aucune valeur magique dans le
code. Modifier le modèle = éditer un YAML.

## Structure du projet

```
src/mpp_predictor/
├── config.py              # chargement + validation (poids → somme = 1)
├── ingestion/loaders.py   # accès aux sources de données
├── features/
│   ├── models.py          # objets métier (TeamSnapshot, KeyPlayer…)
│   └── attack_index.py    # IAG : critères D, F, C
├── model/
│   ├── expected_goals.py  # index → λ
│   └── poisson_engine.py  # λ → matrice de scores
└── decision/
    └── mpp_optimizer.py    # matrice → prono optimal MPP
```

## Résultats du backtest

Testé sur **276 matchs réels de Coupe du Monde (2010+)**, le modèle ne voyant
que l'historique antérieur à chaque match (aucune fuite de données) :

| Version | Points totaux | Moyenne/match | Scores exacts |
|---------|---------------|---------------|---------------|
| **Modèle + Elo + Dixon-Coles** | **253** | **0.917** | 33 (12.0 %) |
| Modèle de base (Elo neutre) | 234 | 0.848 | 32 (11.6 %) |
| Baseline « toujours 2-1 » | 207 | 0.750 | — |
| Baseline « toujours 1-1 » | 158 | 0.572 | — |

L'ajout du classement Elo (force réelle des équipes) fait gagner ~19 points.
Le modèle bat largement les deux baselines naïves.

```bash
# Télécharger le dataset (~3.5 Mo, 49k matchs depuis 1872)
curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv \
    -o data/raw/results.csv
python -m examples.run_backtest    # tester sur les vraies Coupes du Monde
python -m examples.calibrate       # trouver les meilleurs paramètres
```

## Améliorations intégrées

- **Classement Elo** (`features/elo.py`) : force réelle de chaque équipe,
  calculée en rejouant tout l'historique. Module l'espérance de buts. Intensité
  réglable (`elo.strength`) et **calibrée** sur le backtest.
- **Correction Dixon-Coles** (`model/poisson_engine.py`) : ajuste les scores
  serrés que la loi de Poisson simple sous-estime. Paramètre `rho` calibré.
- **Calibration automatique** (`decision/calibration.py`, `examples/calibrate.py`) :
  recherche les meilleurs paramètres par descente de coordonnées, au lieu de
  les deviner. Les chiffres décident.
- **Critère fatigue (F)** : activable via un fichier `data/fatigue_overrides.csv`.
  Un générateur de modèle (`examples/make_fatigue_template.py`) prépare le
  fichier à remplir pour n'importe quel match.

### Honnêteté sur le critère fatigue

L'idée originale — pénaliser les sélections dont les stars sont cramées par une
saison de club surchargée — est implémentée (courbe en U inversé), mais elle
fonctionne en **saisie manuelle** : les minutes en club par joueur ne sont pas
dans le dataset de résultats internationaux et ne sont pas téléchargeables
automatiquement de façon fiable (FBref limite le scraping, Kaggle exige un
compte). La voie automatique complète (mapping joueurs ↔ clubs via FBref) reste
un chantier ouvert, marqué dans la roadmap. Le critère fatigue est un
ajustement fin, pas le moteur principal du modèle (c'est l'Elo qui domine).

## Feuille de route

- [x] Index d'Attaque (D, F, C) + moteur Poisson + décision MPP
- [x] Index de Défense (structure miroir)
- [x] Loader de résultats réels + backtest sur vraies Coupes du Monde
- [x] Classement Elo branché et calibré
- [x] Correction Dixon-Coles
- [x] Calibration automatique des paramètres
- [x] Critère fatigue activable (saisie manuelle assistée)
- [ ] Critère fatigue 100 % automatique (mapping joueurs ↔ clubs via FBref)
- [ ] Modèle de popularité communautaire MPP (exploiter les biais des parieurs)

## Avertissement

Projet personnel à but pédagogique et ludique. Les paris comportent un risque ;
aucune garantie de gain n'est fournie ou suggérée.

---
*Architecture en 4 couches · Python 3.10+ · NumPy / SciPy / pandas*
