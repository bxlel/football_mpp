"""Pont entre les Super-Index et les espérances de buts (λ) de Poisson.

L'espérance de buts d'une équipe dépend de SA force offensive et de la
faiblesse défensive ADVERSE. Forme multiplicative classique autour d'une
moyenne de référence (buts moyens par équipe et par match en tournoi) :

    λ_équipe = base · IAG_équipe · IDG_adverse

où IAG est l'Index d'Attaque et IDG l'Index de Défense adverse (un IDG élevé
= défense permissive). Tant que la couche défense n'est pas branchée, on passe
un IDG neutre de 1.0.
"""

from __future__ import annotations

# Buts moyens marqués par une équipe sur un match de phase finale.
# Référence empirique (~2.6 buts/match au total -> ~1.3 par équipe).
BASE_GOALS = 1.30


def indices_to_lambda(
    attack_index: float,
    opponent_defense_index: float = 1.0,
    base_goals: float = BASE_GOALS,
) -> float:
    """Transforme des index normalisés en espérance de buts pour Poisson."""
    lam = base_goals * attack_index * opponent_defense_index
    # Garde-fou : Poisson exige λ > 0.
    return max(lam, 0.05)
