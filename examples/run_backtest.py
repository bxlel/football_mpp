"""
Script de backtest complet pour toutes les Coupes du Monde depuis 2002.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from mpp_predictor.config import load_config
from mpp_predictor.decision.backtester import run_backtest
from mpp_predictor.ingestion.results_loader import load_results

CSV_PATH = Path("data/raw/results.csv")

def main():
    if not CSV_PATH.exists():
        print("Erreur : Fichier data/raw/results.csv introuvable.")
        return

    cfg = load_config()
    df = load_results(CSV_PATH)
    
    # Conversion forcée en datetime
    df['date'] = pd.to_datetime(df['date'])
    
    editions = [2002, 2006, 2010, 2014, 2018, 2022, 2026]
    results_history = []

    print(f"{'Année':<10} | {'Matchs':<8} | {'Moy. Pts':<10} | {'Scores Exacts'}")
    print("-" * 50)

    for annee in editions:
        test = df[
            (df["tournament"] == "FIFA World Cup") & 
            (df["date"].dt.year == annee)
        ]
        
        if len(test) == 0: 
            continue
        
        res = run_backtest(df, cfg, test_matches=test)
        results_history.append(res.model_points / len(test))
        
        # Calcul manuel du pourcentage ici
        pct_exact = (res.exact_scores / len(test)) * 100
        
        print(f"{annee:<10} | {len(test):<8} | {res.model_points/len(test):<10.3f} | {pct_exact:.1f}%")

    if results_history:
        avg_perf = sum(results_history) / len(results_history)
        print("-" * 50)
        print(f"Performance moyenne sur 2002-2026 : {avg_perf:.3f} pts/match")

if __name__ == "__main__":
    main()