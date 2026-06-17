"""Couche d'ingestion — Raw Data.

Cette couche documente et encapsule l'accès aux sources de données. Elle est
volontairement découplée du reste : le modèle ne sait pas D'OÙ viennent les
données, seulement comment elles sont structurées (cf. features.models).

Sources recommandées (toutes vérifiées comme accessibles) :

1. RÉSULTATS DES SÉLECTIONS (critère D)
   - football-data.org : API REST, tier gratuit (quota limité). Résultats
     internationaux, compétitions majeures.
   - Elo des sélections : eloratings.net (scraping, structure stable).

2. MINUTES / MATCHS JOUÉS EN CLUB (critère F)
   - Dataset Kaggle FBref "Football Players Stats 2025-2026" : CSV mis à jour
     chaque semaine, colonnes MP / Starts / Min par joueur sur les 5 grands
     championnats. Le chemin le plus simple — aucun scraping, aucune clé.
   - Alternative scraping : librairie worldfootballR (R) ou scraping direct
     FBref (tables rendues dynamiquement -> plus délicat).

3. CONTEXTE (critère C)
   - Météo : Open-Meteo (gratuit, sans clé) — historique et prévisions par
     coordonnées de ville hôte.
   - Repos / calendrier : déduit des dates de matchs.

STRATÉGIE MVP : commencer par le dataset Kaggle (CSV) pour F, l'API
football-data pour D, Open-Meteo pour C. Brancher chaque "loader" derrière
l'interface ci-dessous pour pouvoir changer de source sans toucher au modèle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class PlayerLoadStats(Protocol):
    """Contrat minimal d'un chargeur de stats joueurs (critère F)."""

    def load(self) -> pd.DataFrame:
        """Renvoie un DataFrame avec au moins : player, squad, min, mp."""
        ...


class FBrefCsvLoader:
    """Charge le dataset FBref (CSV Kaggle) des minutes joueurs.

    Télécharge le CSV depuis Kaggle manuellement (ou via l'API Kaggle) et
    pointe ce loader dessus. On normalise les noms de colonnes utiles.
    """

    REQUIRED_COLUMNS = ("Player", "Squad", "Min", "MP")

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans le CSV FBref : {missing}. "
                "Vérifie que tu utilises bien le dataset 'players_data'."
            )
        out = df[list(self.REQUIRED_COLUMNS)].copy()
        out.columns = ["player", "squad", "minutes", "matches_played"]
        return out
