"""Prédire un VRAI match à venir entre deux équipes.

Usage :
    python -m examples.predict_match "DR Congo" "Portugal"

Le script reconstruit l'état récent des deux équipes à partir du CSV de
résultats, fait tourner le modèle complet, et affiche le prono recommandé pour
Mon Petit Prono.

ATTENTION aux noms d'équipes : ils doivent être écrits comme dans le dataset
(en anglais). Quelques exemples utiles pour la Coupe du Monde 2026 :
    "DR Congo"  (PAS "Congo" seul, qui est un autre pays !)
    "Portugal", "Colombia", "Uzbekistan", "France", "Brazil", "Spain",
    "Argentina", "Germany", "England", "Morocco", "Senegal", "United States"

Si tu te trompes de nom, le script te propose les noms ressemblants trouvés.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from mpp_predictor.config import load_config
from mpp_predictor.decision.mpp_optimizer import recommend_prediction
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.defense_index import compute_defense_index
from mpp_predictor.features.elo import compute_elo_history, INITIAL_ELO
from mpp_predictor.features.models import KeyPlayer, TeamSnapshot
from mpp_predictor.ingestion.results_loader import build_snapshot, load_results
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"
FATIGUE_PATH = Path(__file__).resolve().parents[1] / "data" / "fatigue_overrides.csv"


def _load_fatigue(team: str) -> list[KeyPlayer]:
    """Charge la charge des cadres d'une équipe depuis le fichier optionnel.

    Le fichier data/fatigue_overrides.csv est facultatif. S'il existe, il active
    le critère F (fatigue) pour les équipes qui y figurent. Sinon, fraîcheur
    neutre. Voir data/fatigue_overrides.example.csv pour le format.
    """
    if not FATIGUE_PATH.exists():
        return []
    fdf = pd.read_csv(FATIGUE_PATH)
    rows = fdf[fdf["team"].str.lower() == team.lower()]
    players = []
    for _, r in rows.iterrows():
        players.append(KeyPlayer(
            name=str(r["player"]),
            is_offensive=bool(r["is_offensive"]),
            club_matches_last_year=int(r["club_matches_last_year"]),
        ))
    return players


def _find_team(df: pd.DataFrame, name: str) -> str | None:
    """Vérifie qu'une équipe existe ; sinon propose des noms proches."""
    teams = set(df["home_team"]) | set(df["away_team"])
    if name in teams:
        return name
    # Recherche tolérante : noms qui contiennent le texte tapé.
    suggestions = sorted(t for t in teams if name.lower() in t.lower())
    if suggestions:
        print(f"⚠️  '{name}' introuvable. Tu voulais dire l'un de ceux-ci ?")
        for s in suggestions[:8]:
            print(f"     - {s}")
    else:
        print(f"⚠️  '{name}' introuvable et aucune suggestion. "
              "Vérifie l'orthographe (en anglais).")
    return None


def predict(home: str, away: str) -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"Fichier introuvable : {CSV_PATH}\n"
            "Télécharge-le d'abord avec la commande Invoke-WebRequest."
        )

    cfg = load_config()
    df = load_results(CSV_PATH)

    home_ok = _find_team(df, home)
    away_ok = _find_team(df, away)
    if home_ok is None or away_ok is None:
        raise SystemExit("Corrige les noms d'équipes et relance.")

    # On prédit "à partir de maintenant" : le modèle utilise tout le passé connu.
    as_of = pd.Timestamp(datetime.now().date())

    # Elo calculé sur tout l'historique.
    elo = compute_elo_history(df)

    home_snap = build_snapshot(df, home_ok, as_of, n_matches=10, elo_lookup=elo)
    away_snap = build_snapshot(df, away_ok, as_of, n_matches=10, elo_lookup=elo)

    # Critère fatigue (F) si le fichier d'overrides existe.
    home_fatigue = _load_fatigue(home_ok)
    away_fatigue = _load_fatigue(away_ok)
    if home_fatigue:
        home_snap = TeamSnapshot(
            name=home_snap.name, elo=home_snap.elo,
            recent_matches=home_snap.recent_matches,
            key_players=home_fatigue, context=home_snap.context,
        )
    if away_fatigue:
        away_snap = TeamSnapshot(
            name=away_snap.name, elo=away_snap.elo,
            recent_matches=away_snap.recent_matches,
            key_players=away_fatigue, context=away_snap.context,
        )

    if not home_snap.recent_matches or not away_snap.recent_matches:
        raise SystemExit("Pas assez d'historique récent pour une de ces équipes.")

    iag_home = compute_attack_index(home_snap, cfg).weighted_total
    iag_away = compute_attack_index(away_snap, cfg).weighted_total
    idg_home = compute_defense_index(home_snap, cfg).weighted_total
    idg_away = compute_defense_index(away_snap, cfg).weighted_total

    # Ajustement par la force Elo.
    elo_home = elo.get(home_ok, INITIAL_ELO)
    elo_away = elo.get(away_ok, INITIAL_ELO)
    from mpp_predictor.decision.backtester import _elo_attack_multiplier
    adj_home = _elo_attack_multiplier(elo_home, elo_away, is_home=True)
    adj_away = _elo_attack_multiplier(elo_away, elo_home, is_home=False)

    lam_home = indices_to_lambda(iag_home, opponent_defense_index=idg_away) * adj_home
    lam_away = indices_to_lambda(iag_away, opponent_defense_index=idg_home) * adj_away

    try:
        dc_rho = cfg.section("poisson", "dixon_coles_rho")
    except KeyError:
        dc_rho = 0.0
    matrix = build_score_matrix(lam_home, lam_away,
                                max_goals=cfg.section("poisson", "max_goals"),
                                dixon_coles_rho=dc_rho)
    reco = recommend_prediction(matrix, cfg)

    fatigue_note = ""
    if home_fatigue or away_fatigue:
        fatigue_note = " (critère fatigue actif)"

    print("\n" + "=" * 52)
    print(f"  PRÉDICTION : {home_ok}  vs  {away_ok}{fatigue_note}")
    print("=" * 52)
    print(f"Elo : {home_ok} {elo_home:.0f}  -  {elo_away:.0f} {away_ok}")
    print(f"Espérance de buts : {home_ok} {lam_home:.2f} - {lam_away:.2f} {away_ok}")
    print(f"\nProbabilités de résultat :")
    print(f"  Victoire {home_ok:<12} : {matrix.prob_home_win():.1%}")
    print(f"  Match nul            : {matrix.prob_draw():.1%}")
    print(f"  Victoire {away_ok:<12} : {matrix.prob_away_win():.1%}")
    print(f"\nScore le plus probable : "
          f"{reco.most_likely_score[0]}-{reco.most_likely_score[1]} "
          f"(probabilité {reco.most_likely_prob:.1%})")
    print(f"\n>>> PRONO RECOMMANDÉ pour MPP : "
          f"{reco.home_goals}-{reco.away_goals}")
    print(f"    (espérance de points : {reco.expected_points:.2f})")
    if (reco.home_goals, reco.away_goals) != reco.most_likely_score:
        print(f"    Note : différent du score le plus probable -> "
              "c'est l'optimisation MPP qui joue.")
    print()


def main() -> None:
    if len(sys.argv) != 3:
        print('Usage : python -m examples.predict_match "Équipe domicile" "Équipe extérieur"')
        print('Exemple : python -m examples.predict_match "Portugal" "DR Congo"')
        raise SystemExit(1)
    predict(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
