"""Backtesting — mesurer la performance du modèle sur des matchs réels passés.

Principe : pour chaque match d'une période de test, on reconstruit l'état des
deux équipes à partir de LEUR HISTORIQUE ANTÉRIEUR uniquement (pas de fuite de
données), on fait tourner le pipeline complet, on compare le prono recommandé
au vrai résultat, et on cumule les points MPP.

On compare le modèle à deux baselines pour vérifier qu'il apporte un edge :
- "always_1_1" : toujours pronostiquer 1-1 (un grand classique paresseux).
- "always_2_1" : toujours pronostiquer 2-1 (le score le plus fréquent).

Si le modèle ne bat pas ces baselines, il ne sert à rien — c'est le test de
vérité.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Config
from ..decision.mpp_optimizer import _points_for, recommend_prediction
from ..features.attack_index import compute_attack_index
from ..features.defense_index import compute_defense_index
from ..ingestion.results_loader import build_snapshot
from ..model.expected_goals import indices_to_lambda
from ..model.poisson_engine import build_score_matrix


@dataclass
class BacktestResult:
    """Synthèse d'un backtest."""

    n_matches: int
    model_points: int
    model_avg: float
    baseline_1_1_points: int
    baseline_2_1_points: int
    exact_scores: int  # nb de scores exacts trouvés par le modèle

    def summary(self) -> str:
        return (
            f"Matchs testés        : {self.n_matches}\n"
            f"Points MODÈLE        : {self.model_points} "
            f"(moyenne {self.model_avg:.3f}/match)\n"
            f"Scores exacts        : {self.exact_scores} "
            f"({self.exact_scores / self.n_matches:.1%})\n"
            f"Baseline toujours 1-1: {self.baseline_1_1_points} "
            f"(moyenne {self.baseline_1_1_points / self.n_matches:.3f}/match)\n"
            f"Baseline toujours 2-1: {self.baseline_2_1_points} "
            f"(moyenne {self.baseline_2_1_points / self.n_matches:.3f}/match)"
        )


def _elo_attack_multiplier(
    elo_for: float, elo_against: float, is_home: bool, strength: float = 0.8,
    home_advantage: float | None = None,
) -> float:
    """Multiplicateur d'attaque selon l'écart de force Elo.

    Une équipe nettement plus forte que son adversaire marque davantage. On
    convertit l'écart Elo en probabilité de victoire, recentrée autour de 1.0.
    `strength` règle l'intensité de l'effet (0 = neutre). Borné pour rester stable.
    `home_advantage` : bonus terrain pour la prédiction (None = valeur globale ;
    0 conseillé pour des matchs neutres comme la Coupe du Monde).
    """
    from ..features.elo import HOME_ADVANTAGE

    ha = HOME_ADVANTAGE if home_advantage is None else home_advantage
    bonus = ha if is_home else 0.0
    exp = 1.0 / (1.0 + 10 ** ((elo_against - (elo_for + bonus)) / 400.0))
    # exp=0.5 -> 1.0 quelle que soit strength ; s'écarte selon strength.
    mult = 1.0 + strength * (exp - 0.5) * 2.0
    return max(0.5, min(1.8, mult))


def _update_elo(elo, home, away, actual, row, initial, home_adv) -> None:
    """Met à jour l'Elo des deux équipes après un match (cohérent avec elo.py)."""
    from ..features.elo import _goal_bonus, _k_factor

    hs, as_ = actual
    neutral = bool(row.get("neutral", False)) if hasattr(row, "get") else False
    ra = elo.get(home, initial)
    rb = elo.get(away, initial)
    ra_adj = ra + (0.0 if neutral else home_adv)
    expected_home = 1.0 / (1.0 + 10 ** ((rb - ra_adj) / 400.0))
    score_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
    tournament = row["tournament"] if "tournament" in row else ""
    change = _k_factor(tournament) * _goal_bonus(hs - as_) * (score_home - expected_home)
    elo[home] = ra + change
    elo[away] = rb - change


def run_backtest(
    df: pd.DataFrame,
    cfg: Config,
    test_matches: pd.DataFrame,
    history_depth: int = 10,
    min_history: int = 5,
) -> BacktestResult:
    """Fait tourner le modèle sur un ensemble de matchs de test.

    Args:
        df: TOUT l'historique (sert à reconstruire les snapshots).
        cfg: configuration.
        test_matches: sous-ensemble de df à prédire (la période de test).
        history_depth: profondeur d'historique par équipe.
        min_history: nb minimal de matchs passés requis, sinon on saute
            (on ne peut rien prédire sans historique).
    """
    scoring = cfg.section("mpp_scoring")
    max_goals = cfg.section("poisson", "max_goals")
    try:
        dc_rho = cfg.section("poisson", "dixon_coles_rho")
    except KeyError:
        dc_rho = 0.0
    try:
        elo_strength = cfg.section("elo", "strength")
    except KeyError:
        elo_strength = 0.8
    try:
        base_goals = cfg.section("poisson", "base_goals")
    except KeyError:
        base_goals = 1.30
    try:
        pred_ha = cfg.section("elo", "prediction_home_advantage")
    except KeyError:
        pred_ha = None
    try:
        nb_disp = cfg.section("poisson", "nb_dispersion")
    except KeyError:
        nb_disp = 0.0
    try:
        biv_cov = cfg.section("poisson", "bivariate_cov")
    except KeyError:
        biv_cov = 0.0
    try:
        draw_boost = cfg.section("poisson", "draw_boost")
    except KeyError:
        draw_boost = 1.0

    # --- Elo sans fuite de données ---
    # On calcule l'Elo à partir de TOUT l'historique antérieur à la période de
    # test, puis on le met à jour au fil des matchs de test (dans l'ordre des
    # dates). Ainsi chaque prédiction n'utilise que le passé.
    from ..features.elo import compute_elo_history, INITIAL_ELO, HOME_ADVANTAGE

    test_start = test_matches["date"].min()
    pre_test = df[df["date"] < test_start]
    elo = compute_elo_history(pre_test)

    model_pts = 0
    base_11_pts = 0
    base_21_pts = 0
    exact = 0
    counted = 0

    # On parcourt les matchs de test dans l'ordre chronologique (sécurité).
    for _, row in test_matches.sort_values("date").iterrows():
        as_of = row["date"]
        home, away = row["home_team"], row["away_team"]
        actual = (int(row["home_score"]), int(row["away_score"]))

        home_snap = build_snapshot(df, home, as_of, history_depth, elo_lookup=elo)
        away_snap = build_snapshot(df, away, as_of, history_depth, elo_lookup=elo)

        # On exige assez d'historique des deux côtés.
        if (len(home_snap.recent_matches) < min_history
                or len(away_snap.recent_matches) < min_history):
            _update_elo(elo, home, away, actual, row, INITIAL_ELO, HOME_ADVANTAGE)
            continue

        # Index des deux équipes.
        iag_home = compute_attack_index(home_snap, cfg).weighted_total
        iag_away = compute_attack_index(away_snap, cfg).weighted_total
        idg_home = compute_defense_index(home_snap, cfg).weighted_total
        idg_away = compute_defense_index(away_snap, cfg).weighted_total

        # On module l'attaque par la différence de force Elo : une équipe bien
        # plus forte marque davantage. Facteur multiplicatif borné.
        elo_home = elo.get(home, INITIAL_ELO)
        elo_away = elo.get(away, INITIAL_ELO)
        adj_home = _elo_attack_multiplier(elo_home, elo_away, is_home=True, strength=elo_strength, home_advantage=pred_ha)
        adj_away = _elo_attack_multiplier(elo_away, elo_home, is_home=False, strength=elo_strength, home_advantage=pred_ha)

        # λ de chaque équipe = son attaque × permissivité adverse × ajust. Elo.
        lam_home = indices_to_lambda(iag_home, opponent_defense_index=idg_away, base_goals=base_goals) * adj_home
        lam_away = indices_to_lambda(iag_away, opponent_defense_index=idg_home, base_goals=base_goals) * adj_away

        matrix = build_score_matrix(lam_home, lam_away, max_goals=max_goals,
                                     dixon_coles_rho=dc_rho, nb_dispersion=nb_disp,
                                     bivariate_cov=biv_cov, draw_boost=draw_boost)
        reco = recommend_prediction(matrix, cfg)

        model_pts += _points_for((reco.home_goals, reco.away_goals), actual, scoring)
        if (reco.home_goals, reco.away_goals) == actual:
            exact += 1
        base_11_pts += _points_for((1, 1), actual, scoring)
        base_21_pts += _points_for((2, 1), actual, scoring)
        counted += 1

        # Mise à jour de l'Elo APRÈS la prédiction (le match devient du passé).
        _update_elo(elo, home, away, actual, row, INITIAL_ELO, HOME_ADVANTAGE)

    if counted == 0:
        raise ValueError("Aucun match avec assez d'historique pour backtester.")

    return BacktestResult(
        n_matches=counted,
        model_points=model_pts,
        model_avg=model_pts / counted,
        baseline_1_1_points=base_11_pts,
        baseline_2_1_points=base_21_pts,
        exact_scores=exact,
    )
