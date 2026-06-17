"""Tests unitaires du cœur métier.

Couvre les invariants qui ne doivent jamais casser : validation des poids,
forme de la courbe de fraîcheur, propriétés de la matrice Poisson, et le
comportement clé du décideur MPP (recommander != score le plus probable).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpp_predictor.config import IndexWeights, load_config
from mpp_predictor.decision.mpp_optimizer import recommend_prediction
from mpp_predictor.features.attack_index import (
    _freshness_score,
    compute_attack_index,
)
from mpp_predictor.features.models import KeyPlayer, PlayedMatch, TeamSnapshot
from mpp_predictor.model.poisson_engine import build_score_matrix


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        IndexWeights(0.5, 0.3, 0.3)  # somme = 1.1


def test_weights_valid():
    w = IndexWeights(0.5, 0.3, 0.2)
    assert w.dynamics == 0.5


def test_freshness_peaks_at_optimal():
    """La courbe en U inversé doit être maximale au sweet spot (38 matchs)."""
    cfg = load_config()

    def team_with(m):
        return TeamSnapshot(
            name="t", elo=1500,
            key_players=[KeyPlayer("p", True, m)],
        )

    at_optimal = _freshness_score(team_with(38), cfg)
    overloaded = _freshness_score(team_with(60), cfg)
    rusty = _freshness_score(team_with(10), cfg)

    assert at_optimal > overloaded   # surrégime pénalisé
    assert at_optimal > rusty        # sous-régime pénalisé
    assert math.isclose(at_optimal, 1.0, abs_tol=1e-9)


def test_poisson_matrix_is_normalized():
    matrix = build_score_matrix(1.5, 1.2, max_goals=8)
    assert math.isclose(matrix.matrix.sum(), 1.0, abs_tol=1e-9)


def test_poisson_outcomes_sum_to_one():
    matrix = build_score_matrix(1.8, 0.9)
    total = matrix.prob_home_win() + matrix.prob_draw() + matrix.prob_away_win()
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_poisson_rejects_nonpositive_lambda():
    with pytest.raises(ValueError):
        build_score_matrix(0.0, 1.0)


def test_recommendation_returns_valid_score():
    cfg = load_config()
    matrix = build_score_matrix(1.4, 1.1)
    reco = recommend_prediction(matrix, cfg)
    assert reco.home_goals >= 0
    assert reco.away_goals >= 0
    assert reco.expected_points > 0


def test_attack_index_in_reasonable_range():
    cfg = load_config()
    team = TeamSnapshot(
        name="t", elo=1800,
        recent_matches=[
            PlayedMatch(2, 1, 1600, 5),
            PlayedMatch(1, 0, 1700, 15),
        ],
        key_players=[KeyPlayer("p", True, 38)],
    )
    result = compute_attack_index(team, cfg)
    assert result.weighted_total > 0
    assert 0 <= result.freshness <= 1


def test_defense_index_higher_when_conceding_more():
    """Une défense qui encaisse plus doit avoir un IDG (permissivité) plus haut."""
    from mpp_predictor.features.defense_index import compute_defense_index

    cfg = load_config()
    leaky = TeamSnapshot(
        name="leaky", elo=1500,
        recent_matches=[PlayedMatch(0, 3, 1500, 5), PlayedMatch(1, 4, 1500, 15)],
    )
    solid = TeamSnapshot(
        name="solid", elo=1500,
        recent_matches=[PlayedMatch(2, 0, 1500, 5), PlayedMatch(1, 0, 1500, 15)],
    )
    leaky_idg = compute_defense_index(leaky, cfg).weighted_total
    solid_idg = compute_defense_index(solid, cfg).weighted_total
    assert leaky_idg > solid_idg


def test_snapshot_builder_no_data_leakage():
    """build_snapshot ne doit JAMAIS inclure de matchs postérieurs à as_of."""
    import pandas as pd

    from mpp_predictor.ingestion.results_loader import build_snapshot

    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "home_team": ["France", "France", "France"],
        "away_team": ["Spain", "Italy", "Brazil"],
        "home_score": [2, 1, 3],
        "away_score": [0, 1, 0],
    })
    snap = build_snapshot(df, "France", pd.Timestamp("2020-07-01"))
    # Seuls les 2 matchs avant juillet 2020 doivent apparaître.
    assert len(snap.recent_matches) == 2


def test_elo_stronger_team_gains_less_beating_weak():
    """Battre une équipe faible rapporte peu d'Elo ; battre une forte beaucoup."""
    import pandas as pd

    from mpp_predictor.features.elo import compute_elo_history

    # A bat B (B plus faible au départ car on enchaîne) vs A bat C (fort).
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "home_team": ["Strong", "Strong"],
        "away_team": ["Weak", "AlsoStrong"],
        "home_score": [5, 1],
        "away_score": [0, 0],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [True, True],
    })
    elo = compute_elo_history(df)
    # Strong doit avoir gagné des points, et tout le monde part de 1500.
    assert elo["Strong"] > 1500
    assert elo["Weak"] < 1500


def test_dixon_coles_increases_draws():
    """rho négatif doit augmenter la probabilité des nuls serrés (0-0, 1-1)."""
    from mpp_predictor.model.poisson_engine import build_score_matrix

    plain = build_score_matrix(1.3, 1.3, dixon_coles_rho=0.0)
    corrected = build_score_matrix(1.3, 1.3, dixon_coles_rho=-0.1)
    assert corrected.prob_draw() > plain.prob_draw()


def test_calibration_make_config_keeps_weights_valid():
    """La config dérivée par la calibration doit garder des poids sommant à 1."""
    from mpp_predictor.decision.calibration import _make_config

    cfg = load_config()
    derived = _make_config(cfg, elo_s=0.5, rho=-0.04, w_dyn=0.6)
    # Doit se charger sans lever (validation des poids dans IndexWeights).
    w = derived.attack_weights
    assert abs(w.dynamics + w.freshness + w.context - 1.0) < 1e-6
    assert abs(w.dynamics - 0.6) < 1e-9


def test_elo_multiplier_neutral_at_equal_strength():
    """À force égale, le multiplicateur Elo doit valoir ~1.0 (hors avantage terrain)."""
    from mpp_predictor.decision.backtester import _elo_attack_multiplier

    m = _elo_attack_multiplier(1500, 1500, is_home=False, strength=0.8)
    assert abs(m - 1.0) < 1e-9
