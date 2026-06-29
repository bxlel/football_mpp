"""Prédiction avec les VRAIES cotes MPP (1N2).

Usage :
    python -m examples.predict_mpp "Portugal" "DR Congo" 34 140 170

Les 3 nombres = cotes MPP pour [victoire domicile] [nul] [victoire extérieur].
Exemple Portugal-RDC : 34 140 170.

Le script calcule les probabilités (Elo + Poisson, comme d'habitude), puis les
croise avec tes cotes MPP pour te dire quel pronostic rapporte le plus de points
EN MOYENNE — et comment se classent les autres options.

Option : ajouter le bonus de score exact en 6e argument (défaut 20).
    python -m examples.predict_mpp "Portugal" "DR Congo" 34 140 170 25
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from mpp_predictor.config import load_config
from mpp_predictor.decision.backtester import _elo_attack_multiplier
from mpp_predictor.decision.mpp_odds_optimizer import MppOdds, recommend_with_odds
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.defense_index import compute_defense_index
from mpp_predictor.features.elo import INITIAL_ELO, compute_elo_history
from mpp_predictor.ingestion.results_loader import build_snapshot, load_results
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"


def main() -> None:
    if len(sys.argv) < 6:
        print('Usage : python -m examples.predict_mpp "Dom" "Ext" COTE_1 COTE_N COTE_2 [bonus]')
        print('Exemple : python -m examples.predict_mpp "Portugal" "DR Congo" 34 140 170')
        raise SystemExit(1)

    home, away = sys.argv[1], sys.argv[2]
    try:
        c_home, c_draw, c_away = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    except ValueError:
        raise SystemExit("Les cotes doivent être des nombres. Ex : 34 140 170")
    bonus = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0

    if not CSV_PATH.exists():
        raise SystemExit(f"Fichier introuvable : {CSV_PATH}. Télécharge results.csv d'abord.")

    cfg = load_config()
    df = load_results(CSV_PATH)
    teams = set(df["home_team"]) | set(df["away_team"])
    for t in (home, away):
        if t not in teams:
            sugg = sorted(x for x in teams if t.lower() in x.lower())[:6]
            raise SystemExit(f"Équipe inconnue : {t}. Suggestions : {sugg}")

    elo = compute_elo_history(df)
    as_of = pd.Timestamp(datetime.now().date())
    hs = build_snapshot(df, home, as_of, 10, elo_lookup=elo)
    aws = build_snapshot(df, away, as_of, 10, elo_lookup=elo)

    iag_h = compute_attack_index(hs, cfg).weighted_total
    iag_a = compute_attack_index(aws, cfg).weighted_total
    idg_h = compute_defense_index(hs, cfg).weighted_total
    idg_a = compute_defense_index(aws, cfg).weighted_total

    eh, ea = elo.get(home, INITIAL_ELO), elo.get(away, INITIAL_ELO)
    try:
        strength = cfg.section("elo", "strength")
    except KeyError:
        strength = 0.8
    adj_h = _elo_attack_multiplier(eh, ea, is_home=True, strength=strength)
    adj_a = _elo_attack_multiplier(ea, eh, is_home=False, strength=strength)

    lam_h = indices_to_lambda(iag_h, opponent_defense_index=idg_a) * adj_h
    lam_a = indices_to_lambda(iag_a, opponent_defense_index=idg_h) * adj_a

    def _dc(key, default=0.0):
        try:
            return cfg.section("poisson", key)
        except KeyError:
            return default
    matrix = build_score_matrix(lam_h, lam_a,
                                max_goals=cfg.section("poisson", "max_goals"),
                                dixon_coles_rho=_dc("dixon_coles_rho"),
                                nb_dispersion=_dc("nb_dispersion"),
                                bivariate_cov=_dc("bivariate_cov"),
                                draw_boost=_dc("draw_boost", 1.0))

    odds = MppOdds(home_win=c_home, draw=c_draw, away_win=c_away, exact_bonus=bonus)
    reco = recommend_with_odds(matrix, odds)

    print("\n" + "=" * 56)
    print(f"  {home} vs {away}  —  cotes MPP {c_home:.0f}/{c_draw:.0f}/{c_away:.0f}")
    print("=" * 56)
    print(f"Probabilités : Dom {matrix.prob_home_win():.0%} | "
          f"Nul {matrix.prob_draw():.0%} | Ext {matrix.prob_away_win():.0%}")
    print(f"Score le plus probable : {reco.most_likely_score[0]}-{reco.most_likely_score[1]}"
          f" ({reco.most_likely_prob:.0%})")
    print(f"\n>>> PRONO MPP RECOMMANDÉ : {reco.home_goals}-{reco.away_goals}")
    print(f"    espérance = {reco.expected_points:.1f} points par match\n")

    print("Classement des meilleurs paris (espérance de points) :")
    print(f"  {'score':>6} | {'pts espérés':>11}")
    print("  " + "-" * 22)
    for h, a, pts in reco.ranking:
        marker = "  <-- recommandé" if (h, a) == (reco.home_goals, reco.away_goals) else ""
        print(f"  {h}-{a:<4} | {pts:>11.1f}{marker}")

    print("\nRappel : c'est une aide à la décision. La forte cote d'un score")
    print("improbable peut le rendre rentable, mais le risque reste réel.")


if __name__ == "__main__":
    main()
