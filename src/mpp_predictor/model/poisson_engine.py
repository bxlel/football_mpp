"""Moteur prédictif — Loi de Poisson.

À partir des espérances de buts (λ_home, λ_away), on construit la matrice des
probabilités de scores exacts sous l'hypothèse d'indépendance des deux Poisson :

    P(home = i, away = j) = Poisson(i ; λ_home) · Poisson(j ; λ_away)

C'est l'approche standard (Maher, 1982) : simple, robuste, peu de paramètres —
fidèle au Rasoir d'Ockham. L'indépendance est une approximation connue ; on
peut plus tard corriger les scores nuls/serrés (modèle de Dixon-Coles) sans
changer l'interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class ScoreMatrix:
    """Matrice (max_goals+1)x(max_goals+1) de probabilités de scores exacts.

    L'entrée [i, j] est P(domicile marque i, extérieur marque j).
    """

    matrix: np.ndarray
    lambda_home: float
    lambda_away: float

    def prob_exact(self, home_goals: int, away_goals: int) -> float:
        return float(self.matrix[home_goals, away_goals])

    def prob_home_win(self) -> float:
        return float(np.tril(self.matrix, -1).sum())

    def prob_draw(self) -> float:
        return float(np.trace(self.matrix))

    def prob_away_win(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    def most_likely_score(self) -> tuple[int, int]:
        i, j = np.unravel_index(np.argmax(self.matrix), self.matrix.shape)
        return int(i), int(j)


def build_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 8,
    dixon_coles_rho: float = 0.0,
) -> ScoreMatrix:
    """Construit la matrice de scores à partir des deux espérances de buts.

    Args:
        lambda_home, lambda_away: espérances de buts.
        max_goals: taille de la matrice.
        dixon_coles_rho: paramètre de correction des scores serrés. 0 = Poisson
            pur ; valeur typique négative (~-0.05 à -0.15) pour favoriser les
            nuls serrés.
    """
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Les espérances de buts doivent être strictement positives.")

    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, lambda_home)
    away_probs = poisson.pmf(goals, lambda_away)

    # Produit extérieur = matrice conjointe sous indépendance.
    matrix = np.outer(home_probs, away_probs)

    # --- Correction Dixon-Coles ---
    # Poisson simple sous-estime les scores serrés (0-0, 1-1) et surestime
    # 1-0 / 0-1. On applique le facteur tau de Dixon & Coles (1997) aux quatre
    # cases de bas score. rho < 0 augmente les nuls serrés.
    if dixon_coles_rho != 0.0:
        tau = _dixon_coles_tau(lambda_home, lambda_away, dixon_coles_rho)
        matrix[0, 0] *= tau[0]
        matrix[0, 1] *= tau[1]
        matrix[1, 0] *= tau[2]
        matrix[1, 1] *= tau[3]

    # Renormalise (troncature de la queue + correction DC).
    matrix /= matrix.sum()

    return ScoreMatrix(matrix=matrix, lambda_home=lambda_home, lambda_away=lambda_away)


def _dixon_coles_tau(lh: float, la: float, rho: float) -> tuple[float, float, float, float]:
    """Facteurs de correction pour les scores (0,0),(0,1),(1,0),(1,1)."""
    return (
        1.0 - lh * la * rho,   # 0-0
        1.0 + lh * rho,        # 0-1
        1.0 + la * rho,        # 1-0
        1.0 - rho,             # 1-1
    )
