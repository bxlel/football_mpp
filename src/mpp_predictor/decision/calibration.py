"""Calibration par recherche sur grille (grid search).

Au lieu de deviner les paramètres, on teste systématiquement des combinaisons
et on garde celle qui maximise les points MPP sur le backtest. C'est la méthode
honnête : les chiffres décident, pas l'intuition.

Paramètres calibrés :
  - elo.strength        : intensité de l'effet Elo
  - poisson.dixon_coles_rho : correction des scores serrés
  - poids D / F / C de l'attaque (on fait varier la part de la dynamique)

On évite le surapprentissage en gardant une grille volontairement grossière
(principe Ockham) : on cherche une bonne région, pas la 4e décimale.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from ..config import Config
from .backtester import run_backtest


@dataclass
class CalibrationResult:
    """Meilleure combinaison trouvée et sa performance."""

    elo_strength: float
    dixon_coles_rho: float
    attack_dynamics_weight: float
    points: int
    exact_scores: int
    combos_tested: int


def _make_config(base: Config, elo_s: float, rho: float, w_dyn: float) -> Config:
    """Crée une config dérivée avec les paramètres donnés.

    Les poids restants (F et C) sont répartis proportionnellement pour que la
    somme reste 1.0 (contrainte validée par IndexWeights).
    """
    raw = copy.deepcopy(base.raw)
    raw["elo"]["strength"] = elo_s
    raw["poisson"]["dixon_coles_rho"] = rho

    # On garde le ratio F:C d'origine et on ajuste pour sommer à 1 avec w_dyn.
    orig = base.raw["attack_index"]["weights"]
    rest = orig["freshness"] + orig["context"]
    remaining = 1.0 - w_dyn
    f = remaining * (orig["freshness"] / rest)
    c = remaining * (orig["context"] / rest)
    raw["attack_index"]["weights"] = {"dynamics": w_dyn, "freshness": f, "context": c}
    # Défense : on aligne le même poids de dynamique pour la cohérence.
    draw = raw["defense_index"]["weights"]
    drest = draw["freshness"] + draw["context"]
    draw["dynamics"] = w_dyn
    draw["freshness"] = remaining * (base.raw["defense_index"]["weights"]["freshness"] / drest)
    draw["context"] = remaining * (base.raw["defense_index"]["weights"]["context"] / drest)

    return Config(raw=raw)


def calibrate(
    df: pd.DataFrame,
    base_cfg: Config,
    test_matches: pd.DataFrame,
    elo_strengths: list[float] | None = None,
    rhos: list[float] | None = None,
    dynamics_weights: list[float] | None = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Calibration par descente de coordonnées (rapide).

    Plutôt que tester toutes les combinaisons (lent), on optimise un paramètre
    à la fois en gardant les autres fixés au meilleur courant. Quelques passes
    suffisent à trouver une bonne région — fidèle au principe Ockham et bien
    plus rapide qu'une grille complète.
    """
    elo_strengths = elo_strengths or [0.4, 0.6, 0.8, 1.0, 1.2]
    rhos = rhos or [0.0, -0.03, -0.05, -0.08]
    dynamics_weights = dynamics_weights or [0.40, 0.50, 0.60, 0.70]

    # Valeurs de départ.
    best_elo = base_cfg.raw["elo"]["strength"]
    best_rho = base_cfg.raw["poisson"]["dixon_coles_rho"]
    best_dyn = base_cfg.raw["attack_index"]["weights"]["dynamics"]

    def score(elo_s, rho, w_dyn):
        cfg = _make_config(base_cfg, elo_s, rho, w_dyn)
        return run_backtest(df, cfg, test_matches=test_matches)

    best_result = score(best_elo, best_rho, best_dyn)
    best_points = best_result.model_points
    tested = 1

    # Une passe sur chaque axe (suffisant et rapide).
    for elo_s in elo_strengths:
        r = score(elo_s, best_rho, best_dyn)
        tested += 1
        if r.model_points > best_points:
            best_points, best_elo, best_result = r.model_points, elo_s, r
            if verbose:
                print(f"  elo.strength={elo_s} -> {r.model_points} pts")

    for rho in rhos:
        r = score(best_elo, rho, best_dyn)
        tested += 1
        if r.model_points > best_points:
            best_points, best_rho, best_result = r.model_points, rho, r
            if verbose:
                print(f"  rho={rho} -> {r.model_points} pts")

    for w_dyn in dynamics_weights:
        r = score(best_elo, best_rho, w_dyn)
        tested += 1
        if r.model_points > best_points:
            best_points, best_dyn, best_result = r.model_points, w_dyn, r
            if verbose:
                print(f"  poids_dynamique={w_dyn} -> {r.model_points} pts")

    return CalibrationResult(
        elo_strength=best_elo,
        dixon_coles_rho=best_rho,
        attack_dynamics_weight=best_dyn,
        points=best_points,
        exact_scores=best_result.exact_scores,
        combos_tested=tested,
    )
