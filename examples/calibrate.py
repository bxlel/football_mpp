"""Calibration automatique complète du modèle (recherche sur grille).

Usage :
    python -m examples.calibrate

Teste de nombreuses combinaisons de paramètres (intensité Elo, correction
Dixon-Coles, équilibre des poids) sur le backtest, et affiche la meilleure.
Reporte ensuite les valeurs trouvées dans config/params.yaml.
"""

from __future__ import annotations

from pathlib import Path

from mpp_predictor.config import load_config
from mpp_predictor.decision.calibration import calibrate
from mpp_predictor.ingestion.results_loader import load_results

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Fichier introuvable : {CSV_PATH}")

    base_cfg = load_config()
    df = load_results(CSV_PATH)
    test = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2010-01-01")]

    print(f"Calibration sur {len(test)} matchs de Coupe du Monde.")
    print("Recherche en cours (peut prendre 1-2 minutes)...\n")

    best = calibrate(df, base_cfg, test_matches=test)

    print("\n" + "=" * 50)
    print("MEILLEURE COMBINAISON TROUVÉE")
    print("=" * 50)
    print(f"  Combinaisons testées   : {best.combos_tested}")
    print(f"  elo.strength           : {best.elo_strength}")
    print(f"  poisson.dixon_coles_rho: {best.dixon_coles_rho}")
    print(f"  poids dynamique attaque: {best.attack_dynamics_weight}")
    print(f"  -> Points              : {best.points}")
    print(f"  -> Scores exacts       : {best.exact_scores}")
    print("\nReporte ces valeurs dans config/params.yaml.")


if __name__ == "__main__":
    main()
