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


def _dc(cfg, key, default=0.0):
    """Lecture défensive d'un paramètre poisson de la config."""
    try:
        return cfg.section("poisson", key)
    except KeyError:
        return default


def _load_fatigue(team: str) -> list[KeyPlayer]:
    """Charge la charge des cadres d'une équipe depuis le fichier optionnel.

    Le fichier data/fatigue_overrides.csv est facultatif. S'il existe, il active
    le critère F (fatigue) pour les équipes qui y figurent. Sinon, fraîcheur
    neutre. Voir data/fatigue_overrides.example.csv pour le format.

    Deux formats acceptés :
    - colonne `club_matches_last_year` par joueur (saisie manuelle), OU
    - colonne `average_matches_played` (remplie automatiquement par
      magic_predict.py via Kaggle) : même valeur appliquée à tous les cadres.
    Si les deux existent, `average_matches_played` (auto) a la priorité dès
    qu'elle est renseignée.
    """
    if not FATIGUE_PATH.exists():
        return []
    fdf = pd.read_csv(FATIGUE_PATH)
    rows = fdf[fdf["team"].str.lower() == team.lower()]
    players = []
    has_auto = "average_matches_played" in fdf.columns
    for _, r in rows.iterrows():
        load = None
        # Priorité à la valeur automatique si elle est présente et valide.
        if has_auto and pd.notna(r.get("average_matches_played")):
            try:
                load = float(r["average_matches_played"])
            except (TypeError, ValueError):
                load = None
        # Sinon, valeur manuelle par joueur.
        if load is None and "club_matches_last_year" in fdf.columns:
            try:
                load = float(r["club_matches_last_year"])
            except (TypeError, ValueError):
                load = None
        if load is None:
            continue
        players.append(KeyPlayer(
            name=str(r["player"]),
            is_offensive=bool(r["is_offensive"]),
            club_matches_last_year=int(round(load)),
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
                                dixon_coles_rho=dc_rho,
                                nb_dispersion=_dc(cfg, "nb_dispersion"),
                                bivariate_cov=_dc(cfg, "bivariate_cov"),
                                draw_boost=_dc(cfg, "draw_boost", 1.0))
    reco = recommend_prediction(matrix, cfg)

    # --- FILTRE DE CONFIANCE ---
    # Basé sur l'analyse des 661 matchs de backtest :
    # - prob_prono < 8%  -> taux de hit = 0%  (ne jamais faire confiance)
    # - prob_prono 8-12% -> taux de hit ~28%  (prudence)
    # - prob_prono > 12% -> taux de hit ~46%  (confiance élevée)
    prob_prono = float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100
    if prob_prono < 8.0:
        confidence = "🔴 FAIBLE  — utilise ton jugement, le modèle est incertain"
    elif prob_prono < 12.0:
        confidence = "🟡 MOYENNE — le modèle trouve ce type de score ~28% du temps"
    else:
        confidence = "🟢 ÉLEVÉE  — le modèle trouve ce type de score ~46% du temps"

    is_same = (reco.home_goals, reco.away_goals) == reco.most_likely_score
    fatigue_note = " (fatigue active)" if (home_fatigue or away_fatigue) else ""

    print("\n" + "=" * 56)
    print(f"  {home_ok}  vs  {away_ok}{fatigue_note}")
    print("=" * 56)
    print(f"Force Elo : {elo_home:.0f} vs {elo_away:.0f}")
    print(f"Buts attendus : {lam_home:.2f} – {lam_away:.2f}")
    print()
    print(f"  Victoire {home_ok:<14} : {matrix.prob_home_win():.0%}")
    print(f"  Match nul              : {matrix.prob_draw():.0%}")
    print(f"  Victoire {away_ok:<14} : {matrix.prob_away_win():.0%}")
    print()
    print(f">>> PRONO : {reco.home_goals}-{reco.away_goals}")
    if is_same:
        print(f"    C'est aussi le score le plus probable ({prob_prono:.0f}%)")
    else:
        print(f"    Score le plus probable : {reco.most_likely_score[0]}-{reco.most_likely_score[1]} ({reco.most_likely_prob:.0%})")
        print(f"    Mais {reco.home_goals}-{reco.away_goals} rapporte plus de points en moyenne sur MPP")
    print()
    print(f"    {confidence}")
    if (reco.home_goals, reco.away_goals) != reco.most_likely_score:
        print(f"\n    Note : différent du score le plus probable -> "
              "optimisation MPP active.")
    print()


def main() -> None:
    if len(sys.argv) != 3:
        print('Usage : python -m examples.predict_match "Équipe domicile" "Équipe extérieur"')
        print('Exemple : python -m examples.predict_match "Portugal" "DR Congo"')
        raise SystemExit(1)
    predict(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
