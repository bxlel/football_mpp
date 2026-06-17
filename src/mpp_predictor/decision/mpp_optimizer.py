"""Module de décision Mon Petit Prono.

Idée centrale, et tout l'intérêt du projet : sur MPP on ne gagne pas en
prédisant le score le PLUS PROBABLE, mais celui qui MAXIMISE l'espérance de
points selon le barème. Ces deux scores diffèrent souvent.

Pour chaque pronostic candidat (h, a), l'espérance de points vaut :

    E[points | (h,a)] = Σ_{i,j} P(i,j) · points(prono=(h,a), réel=(i,j))

On balaie tous les candidats de la matrice et on prend l'argmax. Le barème
(score exact / différence de buts / bon résultat) vient de la config, donc
s'adapte à ta ligue.

Le levier "psychologie de la communauté" se branche ici : si beaucoup de
parieurs convergent sur un gros score populaire, viser un score serré mais
probable peut être plus rentable. On expose un hook `popularity_penalty`
optionnel pour moduler les candidats trop consensuels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..model.poisson_engine import ScoreMatrix


@dataclass(frozen=True)
class Recommendation:
    """Pronostic recommandé et son espérance de points."""

    home_goals: int
    away_goals: int
    expected_points: float
    most_likely_score: tuple[int, int]
    most_likely_prob: float


def _outcome(h: int, a: int) -> int:
    """+1 victoire domicile, 0 nul, -1 victoire extérieur."""
    return (h > a) - (h < a)


def _points_for(
    prono: tuple[int, int], actual: tuple[int, int], scoring: dict[str, int]
) -> int:
    """Points marqués si on pronostique `prono` et que le réel est `actual`."""
    ph, pa = prono
    ah, aa = actual
    if (ph, pa) == (ah, aa):
        return scoring["exact_score"]
    if (ph - pa) == (ah - aa):
        return scoring["goal_difference"]
    if _outcome(ph, pa) == _outcome(ah, aa):
        return scoring["correct_outcome"]
    return 0


def recommend_prediction(
    score_matrix: ScoreMatrix,
    cfg: Config,
    popularity_penalty: Callable[[int, int], float] | None = None,
) -> Recommendation:
    """Trouve le pronostic qui maximise l'espérance de points MPP.

    Args:
        score_matrix: matrice de probabilités issue de Poisson.
        cfg: config (pour le barème).
        popularity_penalty: fonction optionnelle (h, a) -> facteur dans ]0, 1]
            appliqué à l'espérance d'un candidat pour décourager les scores
            trop consensuels dans la communauté. None = pas de pénalité.

    Returns:
        Le pronostic recommandé, distinct du score le plus probable quand le
        barème le justifie.
    """
    scoring = cfg.section("mpp_scoring")
    matrix = score_matrix.matrix
    size = matrix.shape[0]

    best: Recommendation | None = None
    for ph in range(size):
        for pa in range(size):
            exp_points = 0.0
            for ah in range(size):
                for aa in range(size):
                    p = matrix[ah, aa]
                    if p > 0.0:
                        exp_points += p * _points_for((ph, pa), (ah, aa), scoring)

            if popularity_penalty is not None:
                exp_points *= popularity_penalty(ph, pa)

            if best is None or exp_points > best.expected_points:
                ml_i, ml_j = score_matrix.most_likely_score()
                best = Recommendation(
                    home_goals=ph,
                    away_goals=pa,
                    expected_points=exp_points,
                    most_likely_score=(ml_i, ml_j),
                    most_likely_prob=float(matrix[ml_i, ml_j]),
                )

    assert best is not None  # la matrice est non vide
    return best
