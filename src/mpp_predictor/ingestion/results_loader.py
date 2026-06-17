"""Chargement des résultats internationaux réels et construction des snapshots.

Source : dataset martj42/international_results (CSV public, ~49k matchs depuis
1872). Colonnes : date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral.

Ce module fait deux choses :
1. Charger le CSV en DataFrame propre (dates parsées, types corrects).
2. Reconstruire, pour une équipe à une date donnée, son TeamSnapshot à partir
   de ses N derniers matchs — exactement ce que la couche features attend.

Note : ce dataset ne contient pas les minutes en club des joueurs (critère F).
Le backtest fonctionne donc avec la fraîcheur neutre par défaut. Le critère F
se branchera via le CSV FBref séparé (cf. loaders.FBrefCsvLoader).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..features.models import PlayedMatch, TeamSnapshot

# Elo par défaut quand on n'a pas de table Elo dédiée. On approxime ici par une
# valeur neutre ; un vrai pipeline brancherait eloratings.net.
DEFAULT_ELO = 1500.0


def load_results(csv_path: Path) -> pd.DataFrame:
    """Charge et nettoie le CSV de résultats internationaux."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    expected = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV résultats : {missing}")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def build_snapshot(
    df: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
    n_matches: int = 10,
    elo_lookup: dict[str, float] | None = None,
) -> TeamSnapshot:
    """Reconstruit le TeamSnapshot d'une équipe juste avant une date donnée.

    On ne regarde QUE les matchs strictement antérieurs à ``as_of`` pour éviter
    toute fuite de données (data leakage) — crucial pour un backtest honnête.

    Args:
        df: résultats chargés via load_results.
        team: nom de l'équipe (tel qu'écrit dans le CSV).
        as_of: date du match à prédire ; on prend les matchs d'avant.
        n_matches: profondeur de l'historique récent.
        elo_lookup: table optionnelle nom -> Elo. Sinon Elo neutre.
    """
    elo_lookup = elo_lookup or {}
    past = df[
        (df["date"] < as_of)
        & ((df["home_team"] == team) | (df["away_team"] == team))
    ].tail(n_matches)

    recent: list[PlayedMatch] = []
    for _, row in past.iterrows():
        is_home = row["home_team"] == team
        gf = row["home_score"] if is_home else row["away_score"]
        ga = row["away_score"] if is_home else row["home_score"]
        opponent = row["away_team"] if is_home else row["home_team"]
        days_ago = (as_of - row["date"]).days
        recent.append(
            PlayedMatch(
                goals_for=int(gf),
                goals_against=int(ga),
                opponent_elo=elo_lookup.get(opponent, DEFAULT_ELO),
                days_ago=int(days_ago),
            )
        )

    return TeamSnapshot(
        name=team,
        elo=elo_lookup.get(team, DEFAULT_ELO),
        recent_matches=recent,
        key_players=[],   # pas de données club ici -> fraîcheur neutre
        context=None,
    )
