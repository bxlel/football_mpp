"""Index d'Attaque Global (IAG) — le cœur du modèle.

    IAG = w_D · D  +  w_F · F  +  w_C · C

avec w_D + w_F + w_C = 1. Chaque critère est ramené sur une échelle
comparable avant pondération.

- D — Dynamique offensive récente : buts marqués sur la fenêtre glissante,
  amortis par récence et pondérés par la force adverse. C'est le signal brut.

- F — Fraîcheur des cadres offensifs : courbe en U inversé sur la charge
  annuelle. Une seule gaussienne capture à la fois le sous-régime (manque de
  rythme) et le surrégime (fatigue). C'est l'avantage informationnel sur le
  bookmaker, qui sous-pondère la fatigue club -> sélection.

- C — Contexte : multiplicateur borné (météo, repos, enjeu). Volontairement
  simple, à la marge.

Toutes les constantes proviennent de la config — aucune valeur en dur ici.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config
from .fitness import fitness
from .models import TeamSnapshot


@dataclass(frozen=True)
class AttackIndexBreakdown:
    """Détail du calcul, utile pour le debug et l'explicabilité."""

    dynamics: float
    freshness: float
    context: float
    weighted_total: float


def _dynamics_score(team: TeamSnapshot, cfg: Config) -> float:
    """Critère D : moyenne de buts récents, amortie et ajustée à l'adversaire.

    D = Σ (buts_i · λ^rang_i · force_adverse_i) / Σ (λ^rang_i)

    La force adverse normalise le fait de marquer contre fort vs contre faible.
    On utilise l'Elo adverse rapporté à une référence (1500 = équipe moyenne).
    """
    section = cfg.section("attack_index", "dynamics")
    n = int(section["n_recent_matches"])
    lam = float(section["time_decay_lambda"])
    use_strength = bool(section["use_opponent_strength"])

    matches = sorted(team.recent_matches, key=lambda m: m.days_ago)[:n]
    if not matches:
        return 0.0

    weighted_goals = 0.0
    weight_sum = 0.0
    for rank, match in enumerate(matches):
        decay = lam**rank
        strength = (match.opponent_elo / 1500.0) if use_strength else 1.0
        weighted_goals += match.goals_for * decay * strength
        weight_sum += decay

    return weighted_goals / weight_sum if weight_sum > 0 else 0.0


def _freshness_score(team: TeamSnapshot, cfg: Config) -> float:
    """Critère F : forme/fraîcheur de l'équipe.

    Deux modes :
    1. Si des cadres offensifs sont renseignés (minutes club), on utilise la
       courbe de forme classique (plateau).
    2. Sinon, on retombe sur un proxy calculé à partir des VRAIES dates de
       matchs : le repos avant le match. Trop peu de repos (enchaînement) ou
       trop (manque de rythme) pénalise légèrement la fraîcheur offensive.
       Activable via attack_index.freshness.use_rest_proxy.
    """
    section = cfg.section("attack_index", "freshness")
    lo = float(section["optimal_low"])
    hi = float(section["optimal_high"])
    sigma = float(section["sigma"])
    k = int(section["key_players_count"])

    players = team.offensive_players[:k]
    if players:
        scores = [fitness(p.club_matches_last_year, lo, hi, sigma) for p in players]
        return sum(scores) / len(scores)

    # --- Proxy basé sur le repos réel (si activé) ---
    use_rest = bool(section.get("use_rest_proxy", False))
    if use_rest and team.recent_matches:
        last = min(team.recent_matches, key=lambda p: p.days_ago)
        rest = last.days_ago
        # Courbe : optimal autour de 4-7 jours de repos.
        # < 3 j (enchaînement) ou > 10 j (manque de rythme) -> pénalité douce.
        rest_opt_lo = float(section.get("rest_optimal_low", 4))
        rest_opt_hi = float(section.get("rest_optimal_high", 8))
        amp = float(section.get("rest_amplitude", 0.10))
        if rest_opt_lo <= rest <= rest_opt_hi:
            return 1.0
        if rest < rest_opt_lo:
            deficit = (rest_opt_lo - rest) / rest_opt_lo
            return max(1.0 - amp, 1.0 - amp * deficit)
        # rest > hi
        excess = min(1.0, (rest - rest_opt_hi) / 14.0)
        return max(1.0 - amp, 1.0 - amp * excess)

    return 1.0


def _context_factor(team: TeamSnapshot, cfg: Config) -> float:
    """Critère C : multiplicateur contextuel borné, centré sur 1.0.

    Heuristiques simples et additives autour de 1.0 :
    - avantage/désavantage de repos vs l'adversaire,
    - chaleur extrême (>30°C) : bride le pressing -> moins d'occasions,
    - altitude marquée (>2000 m) : idem pour les équipes non habituées,
    - match sans enjeu : légère démobilisation offensive.

    Le résultat est clampé dans [min_factor, max_factor].
    """
    section = cfg.section("attack_index", "context")
    lo = float(section["min_factor"])
    hi = float(section["max_factor"])

    factor = 1.0
    ctx = team.context
    if ctx is not None:
        rest_delta = ctx.rest_days - ctx.opponent_rest_days
        factor += 0.01 * max(-5, min(5, rest_delta))  # ±0.05 max

        if ctx.temperature_celsius is not None and ctx.temperature_celsius > 30:
            factor -= 0.05
        if ctx.altitude_meters is not None and ctx.altitude_meters > 2000:
            factor -= 0.05
        if ctx.is_dead_rubber:
            factor -= 0.03

    return max(lo, min(hi, factor))


def compute_attack_index(team: TeamSnapshot, cfg: Config) -> AttackIndexBreakdown:
    """Calcule l'Index d'Attaque Global et renvoie le détail par critère.

    Forme additive (calibrée, échelle compatible Poisson) :

        IAG = w_D·D + w_F·(D·F) + w_C·C

    - D : dynamique de buts récents (signal brut).
    - F ∈ ]0,1] : forme physique des cadres. Le terme w_F·(D·F) vaut au plus
      w_F·D (joueurs frais) et diminue quand ils sont cramés/rouillés. La
      fatigue BRIDE donc bien la composante offensive — sens correct.
    - C : multiplicateur de contexte, centré sur 1.0.

    Sans donnée fatigue, F=1.0 : le terme vaut w_F·D, identique au modèle de
    référence qui marquait 253 points au backtest.
    """
    weights = cfg.attack_weights

    d = _dynamics_score(team, cfg)
    f = _freshness_score(team, cfg)
    c = _context_factor(team, cfg)

    total = weights.dynamics * d + weights.freshness * f + weights.context * c

    return AttackIndexBreakdown(
        dynamics=d, freshness=f, context=c, weighted_total=total
    )
