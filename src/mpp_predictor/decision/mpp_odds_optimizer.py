"""Optimiseur Mon Petit Prono avec les VRAIES cotes 1N2.

Le système MPP réel (contrairement au barème simplifié 3/2/1) :
  - chaque issue 1N2 rapporte une cote en points (ex: Portugal-RDC = 34/140/170) ;
  - un BONUS s'ajoute si le score exact est trouvé (~20 pts, variable selon le
    nombre de parieurs sur ce score, mais on le prend comme paramètre).

Points gagnés avec un pronostic (ph, pa) si le vrai score est (ah, aa) :
  - si même score exact      -> cote_du_résultat + bonus_exact
  - sinon si même résultat   -> cote_du_résultat
  - sinon                    -> 0

L'outil calcule, pour CHAQUE score candidat, l'espérance de points :

    E[(ph,pa)] = Σ_{ah,aa} P(ah,aa) · points((ph,pa) , (ah,aa))

et renvoie le candidat qui maximise cette espérance. C'est ce qui répond à
l'hésitation « ce gros score improbable vaut-il le coup ? » : le modèle
tranche en points espérés, pas au feeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.poisson_engine import ScoreMatrix


@dataclass(frozen=True)
class MppOdds:
    """Cotes MPP d'un match (points par issue 1N2) + bonus de score exact."""

    home_win: float   # points si victoire domicile (ex: 34)
    draw: float       # points si match nul (ex: 140)
    away_win: float   # points si victoire extérieur (ex: 170)
    exact_bonus: float = 20.0  # bonus ajouté si le score exact est trouvé


@dataclass(frozen=True)
class MppRecommendation:
    """Pronostic recommandé selon les cotes MPP réelles."""

    home_goals: int
    away_goals: int
    expected_points: float
    most_likely_score: tuple[int, int]
    most_likely_prob: float
    # Top candidats avec leur espérance, pour comparer les options.
    ranking: list[tuple[int, int, float]]


def _outcome(h: int, a: int) -> int:
    return (h > a) - (h < a)


def _odds_for_outcome(ph: int, pa: int, odds: MppOdds) -> float:
    o = _outcome(ph, pa)
    if o > 0:
        return odds.home_win
    if o < 0:
        return odds.away_win
    return odds.draw


def recommend_with_odds(
    score_matrix: ScoreMatrix, odds: MppOdds
) -> MppRecommendation:
    """Trouve le pronostic maximisant l'espérance de points MPP réelle."""
    matrix = score_matrix.matrix
    size = matrix.shape[0]

    # Espérance de points pour chaque pronostic candidat (ph, pa).
    results: list[tuple[int, int, float]] = []
    for ph in range(size):
        for pa in range(size):
            base_odds = _odds_for_outcome(ph, pa, odds)
            exp_points = 0.0
            for ah in range(size):
                for aa in range(size):
                    p = matrix[ah, aa]
                    if p <= 0.0:
                        continue
                    if _outcome(ph, pa) == _outcome(ah, aa):
                        pts = base_odds
                        if (ph, pa) == (ah, aa):
                            pts += odds.exact_bonus
                        exp_points += p * pts
            results.append((ph, pa, exp_points))

    results.sort(key=lambda x: x[2], reverse=True)
    best_h, best_a, best_pts = results[0]

    ml_i, ml_j = score_matrix.most_likely_score()
    return MppRecommendation(
        home_goals=best_h,
        away_goals=best_a,
        expected_points=best_pts,
        most_likely_score=(ml_i, ml_j),
        most_likely_prob=float(matrix[ml_i, ml_j]),
        ranking=results[:8],
    )
