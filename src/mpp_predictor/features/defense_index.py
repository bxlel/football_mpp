"""Index de Défense Global (IDG) — structure miroir de l'attaque.

    IDG = w_D · D  +  w_F · F  +  w_C · C

Même logique que l'Index d'Attaque, mais côté défensif. Un IDG ÉLEVÉ signifie
une défense PERMISSIVE (qui encaisse) — c'est cohérent avec son usage dans
expected_goals : λ_adverse = base · IAG_adverse · IDG_équipe.

- D — Buts ENCAISSÉS récemment, amortis par récence, pondérés par la force
  offensive adverse (encaisser contre une attaque faible est plus grave).
- F — Fraîcheur des cadres DÉFENSIFS (défenseurs + gardien). Même gaussienne
  en U inversé : un bloc défensif fatigué OU manquant de rythme craque.
- C — Contexte : un bloc bas compact (équipe en sous-régime) réduit les buts
  encaissés ; chaleur/altitude jouent aussi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config
from .fitness import fitness
from .models import TeamSnapshot


@dataclass(frozen=True)
class DefenseIndexBreakdown:
    """Détail du calcul de l'Index de Défense."""

    dynamics: float
    freshness: float
    context: float
    weighted_total: float


def _conceded_dynamics(team: TeamSnapshot, cfg: Config) -> float:
    """Buts encaissés récents, amortis et pondérés par la force adverse.

    On normalise pour qu'une valeur ~1.0 représente une défense moyenne.
    Plus la valeur est haute, plus la défense encaisse.
    """
    section = cfg.section("defense_index", "dynamics")
    n = int(section["n_recent_matches"])
    lam = float(section["time_decay_lambda"])
    use_strength = bool(section["use_opponent_strength"])

    matches = sorted(team.recent_matches, key=lambda m: m.days_ago)[:n]
    if not matches:
        return 1.0

    weighted = 0.0
    weight_sum = 0.0
    for rank, match in enumerate(matches):
        decay = lam**rank
        # Encaisser contre une équipe faible (Elo bas) est plus pénalisant :
        # on divise par la force adverse normalisée.
        strength = (match.opponent_elo / 1500.0) if use_strength else 1.0
        adjusted = match.goals_against / strength if strength > 0 else match.goals_against
        weighted += adjusted * decay
        weight_sum += decay

    return weighted / weight_sum if weight_sum > 0 else 1.0


def _defensive_freshness(team: TeamSnapshot, cfg: Config) -> float:
    """Plateau de forme sur la charge des cadres défensifs.

    Renvoie un facteur de permissivité : 1.0 = défense fraîche et rodée, >1.0
    si fatiguée ou en manque de rythme (elle encaisse alors davantage).
    """
    section = cfg.section("defense_index", "freshness")
    lo = float(section["optimal_low"])
    hi = float(section["optimal_high"])
    sigma = float(section["sigma"])
    k = int(section["key_players_count"])

    players = team.defensive_players[:k]
    if not players:
        return 1.0

    scores = [fitness(p.club_matches_last_year, lo, hi, sigma) for p in players]
    rendement = sum(scores) / len(scores)
    # Inverse : rendement défensif faible -> permissivité élevée.
    return 2.0 - rendement


def _defensive_context(team: TeamSnapshot, cfg: Config) -> float:
    """Contexte défensif borné, centré sur 1.0.

    Un match sans enjeu ou un bloc bas (équipe en sous-régime) réduit les buts
    encaissés ; chaleur extrême fatigue la défense sur la durée.
    """
    section = cfg.section("defense_index", "context")
    lo = float(section.get("min_factor", 0.85))
    hi = float(section.get("max_factor", 1.15))

    factor = 1.0
    ctx = team.context
    if ctx is not None:
        if ctx.is_dead_rubber:
            factor += 0.03  # relâchement défensif
        if ctx.temperature_celsius is not None and ctx.temperature_celsius > 30:
            factor += 0.03
    return max(lo, min(hi, factor))


def compute_defense_index(team: TeamSnapshot, cfg: Config) -> DefenseIndexBreakdown:
    """Calcule l'Index de Défense Global (IDG). Valeur haute = permissive."""
    weights = cfg.defense_weights

    d = _conceded_dynamics(team, cfg)
    f = _defensive_freshness(team, cfg)
    c = _defensive_context(team, cfg)

    total = weights.dynamics * d + weights.freshness * f + weights.context * c

    return DefenseIndexBreakdown(
        dynamics=d, freshness=f, context=c, weighted_total=total
    )
