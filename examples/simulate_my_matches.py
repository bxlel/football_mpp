"""
Simule tous les matchs de la CDM 2026 joués jusqu'au 17 juin.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from mpp_predictor.config import load_config
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix
from mpp_predictor.decision.mpp_optimizer import recommend_prediction
from mpp_predictor.ingestion.results_loader import build_snapshot, load_results
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.defense_index import compute_defense_index
from mpp_predictor.features.elo import compute_elo_history

def main():
    cfg = load_config()
    df = load_results(Path("data/raw/results.csv"))
    df['date'] = pd.to_datetime(df['date'])
    elo = compute_elo_history(df)
    
    # On récupère TOUS les matchs joués de la CDM 2026 jusqu'au 17 juin
    test = df[
        (df["tournament"] == "FIFA World Cup") & 
        (df["date"] >= "2026-06-11") & 
        (df["date"] <= "2026-06-17") &
        (df["home_score"].notna())
    ]
    
    report = []
    
    for _, match in test.iterrows():
        home, away = match['home_team'], match['away_team']
        as_of = match['date']
        
        home_snap = build_snapshot(df, home, as_of, n_matches=10, elo_lookup=elo)
        away_snap = build_snapshot(df, away, as_of, n_matches=10, elo_lookup=elo)
        
        iag_h, iag_a = compute_attack_index(home_snap, cfg).weighted_total, compute_attack_index(away_snap, cfg).weighted_total
        idg_h, idg_a = compute_defense_index(home_snap, cfg).weighted_total, compute_defense_index(away_snap, cfg).weighted_total
        lam_h, lam_a = indices_to_lambda(iag_h, idg_a), indices_to_lambda(iag_a, idg_h)
        
        matrix = build_score_matrix(lam_h, lam_a, max_goals=cfg.section("poisson", "max_goals"))
        reco = recommend_prediction(matrix, cfg)
        
        report.append({
            "Match": f"{home} vs {away}",
            "Score_Réel": f"{int(match['home_score'])}-{int(match['away_score'])}",
            "Modele_Prono": f"{reco.home_goals}-{reco.away_goals}",
            "Modele_Cote": f"{reco.expected_points:.2f}"
        })
    
    pd.DataFrame(report).to_csv("simulation_complete_2026.csv", index=False)
    print(f"✅ Simulation terminée ! {len(test)} matchs analysés.")
    print("Ouvre 'simulation_complete_2026.csv' pour le rapport complet.")

if __name__ == "__main__":
    main()