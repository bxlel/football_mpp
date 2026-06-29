"""Pipeline de prédiction partagé — une seule source de vérité.

Tous les scripts (web, predict_match, predict_mpp, backtest) doivent produire
EXACTEMENT le même pronostic pour un match donné. Pour éviter que l'un d'eux
oublie un paramètre calibré (bug qu'on a déjà rencontré), ils passent tous par
cette fonction unique.
"""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..decision.backtester import _elo_attack_multiplier
from ..features.attack_index import compute_attack_index
from ..features.defense_index import compute_defense_index
from ..features.elo import INITIAL_ELO
from ..ingestion.results_loader import build_snapshot
from .expected_goals import indices_to_lambda
from .poisson_engine import ScoreMatrix, build_score_matrix


def _cfg(cfg: Config, *keys, default):
    try:
        return cfg.section(*keys)
    except KeyError:
        return default


def predict_match(
    cfg: Config,
    df: pd.DataFrame,
    elo: dict[str, float],
    home: str,
    away: str,
    as_of: pd.Timestamp,
    n_matches: int = 10,
    home_key_players=None,
    away_key_players=None,
) -> tuple[ScoreMatrix, float, float, float, float]:
    """Calcule la matrice de scores d'un match avec TOUS les params calibrés.

    Returns:
        (matrice, lambda_home, lambda_away, elo_home, elo_away)
    """
    home_snap = build_snapshot(df, home, as_of, n_matches=n_matches, elo_lookup=elo)
    away_snap = build_snapshot(df, away, as_of, n_matches=n_matches, elo_lookup=elo)

    # Injection optionnelle des cadres (critère fatigue).
    if home_key_players:
        from ..features.models import TeamSnapshot
        home_snap = TeamSnapshot(name=home_snap.name, elo=home_snap.elo,
                                 recent_matches=home_snap.recent_matches,
                                 key_players=home_key_players, context=home_snap.context)
    if away_key_players:
        from ..features.models import TeamSnapshot
        away_snap = TeamSnapshot(name=away_snap.name, elo=away_snap.elo,
                                 recent_matches=away_snap.recent_matches,
                                 key_players=away_key_players, context=away_snap.context)

    iag_h = compute_attack_index(home_snap, cfg).weighted_total
    iag_a = compute_attack_index(away_snap, cfg).weighted_total
    idg_h = compute_defense_index(home_snap, cfg).weighted_total
    idg_a = compute_defense_index(away_snap, cfg).weighted_total

    elo_h = elo.get(home, INITIAL_ELO)
    elo_a = elo.get(away, INITIAL_ELO)
    strength = _cfg(cfg, "elo", "strength", default=0.8)
    pred_ha = _cfg(cfg, "elo", "prediction_home_advantage", default=None)
    adj_h = _elo_attack_multiplier(elo_h, elo_a, is_home=True, strength=strength, home_advantage=pred_ha)
    adj_a = _elo_attack_multiplier(elo_a, elo_h, is_home=False, strength=strength, home_advantage=pred_ha)

    base_goals = _cfg(cfg, "poisson", "base_goals", default=1.30)
    lam_h = indices_to_lambda(iag_h, opponent_defense_index=idg_a, base_goals=base_goals) * adj_h
    lam_a = indices_to_lambda(iag_a, opponent_defense_index=idg_h, base_goals=base_goals) * adj_a

    matrix = build_score_matrix(
        lam_h, lam_a,
        max_goals=cfg.section("poisson", "max_goals"),
        dixon_coles_rho=_cfg(cfg, "poisson", "dixon_coles_rho", default=0.0),
        nb_dispersion=_cfg(cfg, "poisson", "nb_dispersion", default=0.0),
        bivariate_cov=_cfg(cfg, "poisson", "bivariate_cov", default=0.0),
        draw_boost=_cfg(cfg, "poisson", "draw_boost", default=1.0),
    )
    return matrix, lam_h, lam_a, elo_h, elo_a
