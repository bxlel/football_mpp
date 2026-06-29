"""Courbe de forme physique (fitness) en fonction de la charge de matchs.

Au lieu d'un optimum ponctuel (« exactement 38 matchs »), on modélise un
PLATEAU : une zone de forme optimale (un intervalle) où le rendement vaut 1.0,
encadrée par deux pentes qui chutent vers le sous-régime (manque de rythme) et
le surrégime (fatigue). C'est plus fidèle à la physiologie : un joueur est au
top sur une plage, pas sur une valeur unique.

    fitness(m) = 1.0                              si lo <= m <= hi   (plateau)
               = exp(-((m - lo)^2)/(2σ²))         si m < lo          (rouille)
               = exp(-((m - hi)^2)/(2σ²))         si m > hi          (fatigue)

Cette fonction est partagée par l'attaque et la défense pour rester cohérente.
"""

from __future__ import annotations

import math


def fitness(m: float, lo: float, hi: float, sigma: float) -> float:
    """Rendement de forme dans ]0, 1] pour une charge de `m` matchs/an.

    Args:
        m: nombre de matchs joués par le joueur sur l'année glissante.
        lo, hi: bornes de l'intervalle de forme optimale (plateau à 1.0).
        sigma: largeur des pentes de chaque côté du plateau.
    """
    if lo <= m <= hi:
        return 1.0
    if m < lo:
        return math.exp(-((m - lo) ** 2) / (2 * sigma**2))
    return math.exp(-((m - hi) ** 2) / (2 * sigma**2))
