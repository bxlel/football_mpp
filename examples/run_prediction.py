"""Exemple bout-en-bout : du snapshot d'équipe au pronostic MPP recommandé.

Lance :  python -m examples.run_prediction
(depuis la racine du repo, après `pip install -e .`)

Ce script utilise des données fictives mais réalistes pour montrer le pipeline
complet sans dépendre d'une source externe. Il illustre surtout le point clé :
le score recommandé peut différer du score le plus probable.
"""

from __future__ import annotations

from mpp_predictor.config import load_config
from mpp_predictor.decision.mpp_optimizer import recommend_prediction
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.models import (
    KeyPlayer,
    MatchContext,
    PlayedMatch,
    TeamSnapshot,
)
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix


def demo() -> None:
    cfg = load_config()

    # --- Équipe A : grande nation, stars potentiellement en surrégime ---
    team_a = TeamSnapshot(
        name="Équipe A",
        elo=1950.0,
        recent_matches=[
            PlayedMatch(goals_for=3, goals_against=0, opponent_elo=1500, days_ago=5),
            PlayedMatch(goals_for=2, goals_against=1, opponent_elo=1700, days_ago=12),
            PlayedMatch(goals_for=1, goals_against=1, opponent_elo=1850, days_ago=30),
            PlayedMatch(goals_for=4, goals_against=2, opponent_elo=1400, days_ago=60),
        ],
        key_players=[
            KeyPlayer("Star 1", is_offensive=True, club_matches_last_year=58),  # cramé
            KeyPlayer("Star 2", is_offensive=True, club_matches_last_year=52),
            KeyPlayer("Star 3", is_offensive=True, club_matches_last_year=40),
        ],
        context=MatchContext(rest_days=3, opponent_rest_days=4, temperature_celsius=34),
    )

    # --- Équipe B : nation modeste, bloc bas, joueurs frais ---
    team_b = TeamSnapshot(
        name="Équipe B",
        elo=1620.0,
        recent_matches=[
            PlayedMatch(goals_for=1, goals_against=0, opponent_elo=1550, days_ago=6),
            PlayedMatch(goals_for=0, goals_against=0, opponent_elo=1600, days_ago=20),
            PlayedMatch(goals_for=1, goals_against=2, opponent_elo=1900, days_ago=40),
        ],
        key_players=[
            KeyPlayer("Joueur 1", is_offensive=True, club_matches_last_year=36),
            KeyPlayer("Joueur 2", is_offensive=True, club_matches_last_year=39),
        ],
        context=MatchContext(rest_days=4, opponent_rest_days=3, temperature_celsius=34),
    )

    iag_a = compute_attack_index(team_a, cfg)
    iag_b = compute_attack_index(team_b, cfg)

    print(f"--- Index d'Attaque ---")
    print(f"{team_a.name}: D={iag_a.dynamics:.2f} F={iag_a.freshness:.2f} "
          f"C={iag_a.context:.2f} -> IAG={iag_a.weighted_total:.3f}")
    print(f"{team_b.name}: D={iag_b.dynamics:.2f} F={iag_b.freshness:.2f} "
          f"C={iag_b.context:.2f} -> IAG={iag_b.weighted_total:.3f}")

    # Index de défense neutre tant que la couche n'est pas branchée.
    lambda_a = indices_to_lambda(iag_a.weighted_total, opponent_defense_index=1.0)
    lambda_b = indices_to_lambda(iag_b.weighted_total, opponent_defense_index=1.0)

    matrix = build_score_matrix(lambda_a, lambda_b,
                                max_goals=cfg.section("poisson", "max_goals"))

    print(f"\n--- Poisson (λ_A={lambda_a:.2f}, λ_B={lambda_b:.2f}) ---")
    print(f"P(victoire A) = {matrix.prob_home_win():.1%}")
    print(f"P(nul)        = {matrix.prob_draw():.1%}")
    print(f"P(victoire B) = {matrix.prob_away_win():.1%}")

    reco = recommend_prediction(matrix, cfg)
    print(f"\n--- Décision MPP ---")
    print(f"Score le plus probable : {reco.most_likely_score[0]}-"
          f"{reco.most_likely_score[1]} (P={reco.most_likely_prob:.1%})")
    print(f"Prono RECOMMANDÉ (max E[points]) : "
          f"{reco.home_goals}-{reco.away_goals} "
          f"(E[points]={reco.expected_points:.2f})")

    _tight_match_demo(cfg)


def _tight_match_demo(cfg) -> None:
    """Match très serré : illustre quand le prono optimal s'écarte du score
    le plus probable (le vrai intérêt du module de décision)."""
    print("\n" + "=" * 50)
    print("Cas d'école : match serré entre deux équipes proches")
    print("=" * 50)

    # Deux équipes de niveau quasi identique -> beaucoup de scores plausibles,
    # le 0-0/1-1 pèse lourd dans l'espérance de points.
    matrix = build_score_matrix(1.05, 0.95,
                                max_goals=cfg.section("poisson", "max_goals"))
    reco = recommend_prediction(matrix, cfg)

    print(f"P(victoire dom.) = {matrix.prob_home_win():.1%} | "
          f"P(nul) = {matrix.prob_draw():.1%} | "
          f"P(victoire ext.) = {matrix.prob_away_win():.1%}")
    print(f"Score le plus probable : {reco.most_likely_score[0]}-"
          f"{reco.most_likely_score[1]} (P={reco.most_likely_prob:.1%})")
    print(f"Prono RECOMMANDÉ       : {reco.home_goals}-{reco.away_goals} "
          f"(E[points]={reco.expected_points:.2f})")
    if (reco.home_goals, reco.away_goals) != reco.most_likely_score:
        print(">> Le prono optimal DIFFÈRE du score le plus probable : "
              "c'est exactement l'edge MPP.")


if __name__ == "__main__":
    demo()
