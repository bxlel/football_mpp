"""Serveur web pour MPP Predictor.

Expose le modèle de prédiction via une petite page web et une API JSON.
Conçu pour tourner en local (développement) ET être déployé en ligne (Render,
Railway, etc.) sans changement de code.

Endpoints :
    GET  /                  -> la page web
    GET  /api/teams         -> liste des équipes connues (pour l'autocomplétion)
    POST /api/predict       -> {home, away} -> prédiction JSON

Lancement local :
    python -m web.app
puis ouvrir http://localhost:5000
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from mpp_predictor.config import load_config
from mpp_predictor.decision.backtester import _elo_attack_multiplier
from mpp_predictor.decision.mpp_optimizer import recommend_prediction
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.defense_index import compute_defense_index
from mpp_predictor.features.elo import INITIAL_ELO, compute_elo_history
from mpp_predictor.ingestion.results_loader import build_snapshot, load_results
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "results.csv"

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"))


@lru_cache(maxsize=1)
def _load_state():
    """Charge le dataset et calcule l'Elo une seule fois (mis en cache)."""
    cfg = load_config()
    df = load_results(CSV_PATH)
    elo = compute_elo_history(df)
    return cfg, df, elo


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/teams")
def teams():
    _, df, _ = _load_state()
    names = sorted(set(df["home_team"]) | set(df["away_team"]))
    return jsonify(names)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    home = (data.get("home") or "").strip()
    away = (data.get("away") or "").strip()
    if not home or not away:
        return jsonify({"error": "Indique deux équipes."}), 400

    cfg, df, elo = _load_state()
    all_teams = set(df["home_team"]) | set(df["away_team"])
    if home not in all_teams:
        return jsonify({"error": f"Équipe inconnue : {home}"}), 400
    if away not in all_teams:
        return jsonify({"error": f"Équipe inconnue : {away}"}), 400

    as_of = pd.Timestamp(pd.Timestamp.now().date())
    home_snap = build_snapshot(df, home, as_of, n_matches=10, elo_lookup=elo)
    away_snap = build_snapshot(df, away, as_of, n_matches=10, elo_lookup=elo)

    if not home_snap.recent_matches or not away_snap.recent_matches:
        return jsonify({"error": "Pas assez d'historique pour une de ces équipes."}), 400

    iag_home = compute_attack_index(home_snap, cfg).weighted_total
    iag_away = compute_attack_index(away_snap, cfg).weighted_total
    idg_home = compute_defense_index(home_snap, cfg).weighted_total
    idg_away = compute_defense_index(away_snap, cfg).weighted_total

    elo_home = elo.get(home, INITIAL_ELO)
    elo_away = elo.get(away, INITIAL_ELO)
    try:
        strength = cfg.section("elo", "strength")
    except KeyError:
        strength = 0.8
    adj_home = _elo_attack_multiplier(elo_home, elo_away, is_home=True, strength=strength)
    adj_away = _elo_attack_multiplier(elo_away, elo_home, is_home=False, strength=strength)

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

    # Top 5 scores les plus probables pour l'affichage.
    flat = []
    for i in range(matrix.matrix.shape[0]):
        for j in range(matrix.matrix.shape[1]):
            flat.append((i, j, float(matrix.matrix[i, j])))
    flat.sort(key=lambda x: x[2], reverse=True)
    top_scores = [{"home": i, "away": j, "prob": round(p * 100, 1)}
                  for i, j, p in flat[:5]]

    return jsonify({
        "home": home,
        "away": away,
        "elo_home": round(elo_home),
        "elo_away": round(elo_away),
        "lambda_home": round(lam_home, 2),
        "lambda_away": round(lam_away, 2),
        "prob_home_win": round(matrix.prob_home_win() * 100, 1),
        "prob_draw": round(matrix.prob_draw() * 100, 1),
        "prob_away_win": round(matrix.prob_away_win() * 100, 1),
        "most_likely": {"home": reco.most_likely_score[0],
                        "away": reco.most_likely_score[1],
                        "prob": round(reco.most_likely_prob * 100, 1)},
        "recommended": {"home": reco.home_goals, "away": reco.away_goals,
                        "expected_points": round(reco.expected_points, 2)},
        "top_scores": top_scores,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
