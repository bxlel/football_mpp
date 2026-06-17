"""Système de classement Elo pour les sélections nationales.

L'Elo donne à chaque équipe une note de force qui évolue match après match :
- battre une équipe plus forte rapporte beaucoup de points,
- perdre contre plus faible en coûte beaucoup,
- un nul rapproche les deux notes.

On calcule les Elo en rejouant TOUT l'historique chronologiquement. À la fin,
on dispose d'une note par équipe à chaque instant, qu'on injecte dans les
Super-Index à la place de l'Elo neutre 1500.

Formule standard adaptée au football (inspirée de World Football Elo Ratings) :
    note_attendue_A = 1 / (1 + 10^((Elo_B - Elo_A) / 400))
    nouveau_Elo_A   = Elo_A + K · G · (résultat_réel_A - note_attendue_A)
où K est un facteur de vitesse, G un bonus selon l'écart de buts, et
résultat_réel vaut 1 (victoire), 0.5 (nul) ou 0 (défaite).
"""

from __future__ import annotations

import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 65.0  # bonus de points Elo pour l'équipe à domicile


def _k_factor(tournament: str) -> float:
    """Importance du match : un Mondial pèse plus qu'un amical."""
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if "world cup qualif" in t or "uefa euro" in t or "copa" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _goal_bonus(goal_diff: int) -> float:
    """Bonus G : une victoire large compte plus (plafonné)."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0  # croissance douce au-delà


def compute_elo_history(df: pd.DataFrame) -> dict[str, float]:
    """Rejoue tout l'historique et renvoie l'Elo FINAL de chaque équipe.

    Args:
        df: résultats triés par date (issu de load_results).

    Returns:
        Dictionnaire {nom_équipe: Elo courant} après le dernier match connu.
    """
    elo: dict[str, float] = {}

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        hs, as_ = int(row.home_score), int(row.away_score)
        neutral = bool(getattr(row, "neutral", False))

        ra = elo.get(home, INITIAL_ELO)
        rb = elo.get(away, INITIAL_ELO)

        # Avantage du terrain sauf si terrain neutre.
        ra_adj = ra + (0.0 if neutral else HOME_ADVANTAGE)

        expected_home = 1.0 / (1.0 + 10 ** ((rb - ra_adj) / 400.0))

        if hs > as_:
            score_home = 1.0
        elif hs < as_:
            score_home = 0.0
        else:
            score_home = 0.5

        k = _k_factor(getattr(row, "tournament", ""))
        g = _goal_bonus(hs - as_)
        change = k * g * (score_home - expected_home)

        elo[home] = ra + change
        elo[away] = rb - change

    return elo


def win_probability(elo_a: float, elo_b: float, home_advantage: bool = True) -> float:
    """Probabilité que A batte B selon leurs Elo (pour info / sanity check)."""
    ra = elo_a + (HOME_ADVANTAGE if home_advantage else 0.0)
    return 1.0 / (1.0 + 10 ** ((elo_b - ra) / 400.0))
