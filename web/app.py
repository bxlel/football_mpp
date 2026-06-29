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


def _cfg(cfg, *keys, default):
    """Accès défensif à la config (renvoie default si la clé manque)."""
    try:
        return cfg.section(*keys)
    except KeyError:
        return default


@lru_cache(maxsize=1)
def _load_state():
    """Charge le dataset et calcule l'Elo une seule fois (mis en cache)."""
    cfg = load_config()
    df = load_results(CSV_PATH)
    elo = compute_elo_history(df)
    return cfg, df, elo


@app.route("/health")
def health():
    """Réponse instantanée pour que Render détecte le port tout de suite."""
    return "ok", 200


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/teams")
def teams():
    _, df, _ = _load_state()
    from .flags import flag, accent
    names = sorted(set(df["home_team"]) | set(df["away_team"]))
    out = []
    for n in names:
        out.append({"name": n, "flag": flag(n), "accent": accent(n)})
    return jsonify(out)


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
    strength = _cfg(cfg, "elo", "strength", default=0.8)
    pred_ha = _cfg(cfg, "elo", "prediction_home_advantage", default=None)
    adj_home = _elo_attack_multiplier(elo_home, elo_away, is_home=True,
                                      strength=strength, home_advantage=pred_ha)
    adj_away = _elo_attack_multiplier(elo_away, elo_home, is_home=False,
                                      strength=strength, home_advantage=pred_ha)

    base_goals = _cfg(cfg, "poisson", "base_goals", default=1.30)
    lam_home = indices_to_lambda(iag_home, opponent_defense_index=idg_away, base_goals=base_goals) * adj_home
    lam_away = indices_to_lambda(iag_away, opponent_defense_index=idg_home, base_goals=base_goals) * adj_away

    dc_rho = _cfg(cfg, "poisson", "dixon_coles_rho", default=0.0)
    nb_disp = _cfg(cfg, "poisson", "nb_dispersion", default=0.0)
    biv_cov = _cfg(cfg, "poisson", "bivariate_cov", default=0.0)
    draw_boost = _cfg(cfg, "poisson", "draw_boost", default=1.0)
    max_goals = cfg.section("poisson", "max_goals")
    matrix = build_score_matrix(lam_home, lam_away, max_goals=max_goals,
                                dixon_coles_rho=dc_rho, nb_dispersion=nb_disp,
                                bivariate_cov=biv_cov, draw_boost=draw_boost)
    reco = recommend_prediction(matrix, cfg)

    # --- FACTEURS DE FIABILITÉ ---
    # Basés sur l'analyse des 113 scores exacts trouvés sur 669 matchs.
    # Chaque facteur indique si le match est dans une zone où le modèle a
    # historiquement bien (ou mal) performé.
    prono_score = (reco.home_goals, reco.away_goals)
    prob_prono_val = float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100
    total_prono = reco.home_goals + reco.away_goals
    is_draw_prono = reco.home_goals == reco.away_goals
    elo_gap = abs(elo_home - elo_away)

    reliability = []
    # 1. Probabilité du prono (le signal le plus fort)
    if prob_prono_val >= 12:
        reliability.append({"factor": "Probabilité du score élevée", "good": True,
                            "detail": f"{prob_prono_val:.0f}% — le modèle réussit ~46% du temps dans ce cas"})
    elif prob_prono_val < 8:
        reliability.append({"factor": "Probabilité du score faible", "good": False,
                            "detail": f"{prob_prono_val:.0f}% — en dessous de 8%, le modèle tombe rarement juste"})
    else:
        reliability.append({"factor": "Probabilité du score moyenne", "good": None,
                            "detail": f"{prob_prono_val:.0f}% — réussite historique ~28%"})
    # 2. Petit score (le modèle excelle sur les matchs à 1 but)
    if total_prono <= 1:
        reliability.append({"factor": "Match à petit score", "good": True,
                            "detail": "65% des scores exacts trouvés sont des matchs à 1 but (1-0 / 0-1)"})
    elif total_prono >= 4:
        reliability.append({"factor": "Match à gros score", "good": False,
                            "detail": "le modèle ne trouve quasiment jamais les matchs à 4+ buts"})
    # 3. Match nul (angle mort du modèle)
    if is_draw_prono:
        reliability.append({"factor": "Pronostic de match nul", "good": False,
                            "detail": "le modèle ne réussit que 5.8% des nuls — son point faible"})
    # 4. Écart Elo
    if 50 <= elo_gap <= 200:
        reliability.append({"factor": "Écart de force idéal", "good": True,
                            "detail": f"écart Elo de {elo_gap:.0f} — la zone où le modèle est le plus juste"})
    elif elo_gap < 50:
        reliability.append({"factor": "Match très équilibré", "good": False,
                            "detail": f"écart Elo de {elo_gap:.0f} — trop serré, issue difficile à prévoir"})

    # Verdict global
    n_good = sum(1 for r in reliability if r["good"] is True)
    n_bad = sum(1 for r in reliability if r["good"] is False)
    if n_good >= 2 and n_bad == 0:
        verdict = {"level": "high", "text": "Match à fort potentiel — c'est le type de match où le modèle brille"}
    elif n_bad >= 2:
        verdict = {"level": "low", "text": "Match piège — le modèle est ici sur son terrain le moins fiable"}
    else:
        verdict = {"level": "medium", "text": "Match standard — fiabilité moyenne du modèle"}

    # --- STATISTIQUES DÉTAILLÉES DU MODÈLE (analyse des 669 matchs) ---
    # Chaque catégorie montre le taux de réussite réel mesuré sur le backtest.
    # On marque "active" la tranche où se situe le match analysé.
    def mark(cond):
        return bool(cond)

    perf_breakdown = {
        "Probabilité du score recommandé": [
            {"label": "moins de 8%",   "rate": 0,  "n": "195 matchs", "active": mark(prob_prono_val < 8)},
            {"label": "8 à 10%",       "rate": 26, "n": "95 matchs",  "active": mark(8 <= prob_prono_val < 10)},
            {"label": "10 à 12%",      "rate": 31, "n": "84 matchs",  "active": mark(10 <= prob_prono_val < 12)},
            {"label": "12 à 15%",      "rate": 46, "n": "72 matchs",  "active": mark(12 <= prob_prono_val < 15)},
            {"label": "plus de 15%",   "rate": 70, "n": "11 matchs",  "active": mark(prob_prono_val >= 15)},
        ],
        "Nombre de buts du score": [
            {"label": "0 but (0-0)",   "rate": 9,  "n": "64 matchs",  "active": mark(total_prono == 0)},
            {"label": "1 but (1-0/0-1)","rate": 53, "n": "136 matchs", "active": mark(total_prono == 1)},
            {"label": "2 buts",        "rate": 16, "n": "186 matchs", "active": mark(total_prono == 2)},
            {"label": "3 buts",        "rate": 12, "n": "165 matchs", "active": mark(total_prono == 3)},
            {"label": "4 buts et +",   "rate": 0,  "n": "118 matchs", "active": mark(total_prono >= 4)},
        ],
        "Phase du tournoi": [
            {"label": "Phase de groupes", "rate": 19, "n": "492 matchs", "active": None},
            {"label": "Phases finales",   "rate": 11, "n": "177 matchs", "active": None},
        ],
        "Écart de force (Elo)": [
            {"label": "moins de 50",   "rate": 18, "n": "125 matchs", "active": mark(elo_gap < 50)},
            {"label": "50 à 100",      "rate": 24, "n": "122 matchs", "active": mark(50 <= elo_gap < 100)},
            {"label": "100 à 200",     "rate": 13, "n": "215 matchs", "active": mark(100 <= elo_gap < 200)},
            {"label": "200 à 500",     "rate": 15, "n": "190 matchs", "active": mark(200 <= elo_gap < 500)},
            {"label": "plus de 500",   "rate": 18, "n": "11 matchs",  "active": mark(elo_gap >= 500)},
        ],
        "Type de résultat": [
            {"label": "Victoire domicile", "rate": 20, "n": "272 matchs", "active": mark(reco.home_goals > reco.away_goals)},
            {"label": "Match nul",         "rate": 6,  "n": "172 matchs", "active": mark(is_draw_prono)},
            {"label": "Victoire extérieur","rate": 20, "n": "217 matchs", "active": mark(reco.home_goals < reco.away_goals)},
        ],
    }

    m = matrix.matrix

    # Top scores les plus probables.
    flat = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            flat.append((i, j, float(m[i, j])))
    flat.sort(key=lambda x: x[2], reverse=True)
    top_scores = [{"home": i, "away": j, "prob": round(p * 100, 1)}
                  for i, j, p in flat[:6]]

    # Grille de probabilités (heatmap) limitée à 0..5 buts pour la lisibilité.
    grid_n = min(6, m.shape[0])
    grid = [[round(float(m[i, j]) * 100, 1) for j in range(grid_n)]
            for i in range(grid_n)]

    # Distribution du nombre de buts par équipe (0..5+).
    def goal_dist(probs_axis):
        dist = [round(float(probs_axis[k]) * 100, 1) for k in range(min(6, len(probs_axis)))]
        return dist
    home_axis = m.sum(axis=1)
    away_axis = m.sum(axis=0)

    # Marchés dérivés.
    p_btts = float(m[1:, 1:].sum())  # les deux marquent
    p_over25 = 0.0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i + j >= 3:
                p_over25 += float(m[i, j])
    p_clean_home = float(m[:, 0].sum())   # home garde sa cage inviolée
    p_clean_away = float(m[0, :].sum())

    # Forme récente (résultats des derniers matchs, du plus récent au plus ancien).
    def recent_form(snap):
        ms = sorted(snap.recent_matches, key=lambda x: x.days_ago)
        out = []
        gf = ga = 0
        for pm in ms:
            gf += pm.goals_for
            ga += pm.goals_against
            if pm.goals_for > pm.goals_against:
                out.append("W")
            elif pm.goals_for < pm.goals_against:
                out.append("L")
            else:
                out.append("D")
        n = max(1, len(ms))
        return {
            "form": out[:8],
            "avg_scored": round(gf / n, 2),
            "avg_conceded": round(ga / n, 2),
            "played": len(ms),
        }

    # Face-à-face historique entre les deux équipes.
    h2h_mask = (
        ((df["home_team"] == home) & (df["away_team"] == away))
        | ((df["home_team"] == away) & (df["away_team"] == home))
    )
    h2h = df[h2h_mask].sort_values("date").tail(6)
    h2h_list = []
    h2h_home_wins = h2h_away_wins = h2h_draws = 0
    for _, r in h2h.iterrows():
        hh, aa = r["home_team"], r["away_team"]
        hs_, as_ = int(r["home_score"]), int(r["away_score"])
        # Normalise du point de vue de "home" (l'équipe domicile du match prédit).
        if hh == home:
            gf, ga = hs_, as_
        else:
            gf, ga = as_, hs_
        if gf > ga:
            h2h_home_wins += 1
        elif gf < ga:
            h2h_away_wins += 1
        else:
            h2h_draws += 1
        h2h_list.append({
            "date": r["date"].strftime("%Y") if hasattr(r["date"], "strftime") else str(r["date"])[:4],
            "label": f"{hh} {hs_}-{as_} {aa}",
        })

    # --- NOTES STYLE PES (0-100) dérivées des vraies données ---
    # On convertit les métriques réelles en notes lisibles facon jeu vidéo.
    def team_ratings(snap, iag, idg, lam_for, elo_val):
        ms = sorted(snap.recent_matches, key=lambda x: x.days_ago)
        n = max(1, len(ms))
        gf = sum(p.goals_for for p in ms) / n
        ga = sum(p.goals_against for p in ms) / n
        wins = sum(1 for p in ms if p.goals_for > p.goals_against)
        # Attaque : basée sur l'espérance de buts et la prod offensive récente
        attack = max(20, min(99, round(38 + lam_for * 20 + gf * 6)))
        # Défense : inverse des buts encaissés
        defense = max(20, min(99, round(85 - ga * 18)))
        # Forme : ratio de victoires récentes
        form_r = max(20, min(99, round(35 + (wins / n) * 60)))
        # Puissance : basée sur l'Elo (1500 = moyen, 2100 = top mondial)
        power = max(20, min(99, round((elo_val - 1300) / 8)))
        # Finition : prod offensive pure
        finishing = max(20, min(99, round(40 + gf * 18)))
        # Note globale
        overall = round((attack + defense + form_r + power) / 4)
        return {
            "attack": attack, "defense": defense, "form": form_r,
            "power": power, "finishing": finishing, "overall": overall,
        }

    stats_home = team_ratings(home_snap, iag_home, idg_home, lam_home, elo_home)
    stats_away = team_ratings(away_snap, iag_away, idg_away, lam_away, elo_away)

    from .flags import flag, accent

    return jsonify({
        "home": home,
        "away": away,
        "flags": {"home": flag(home), "away": flag(away)},
        "colors": {"home": {"c1": accent(home)}, "away": {"c1": accent(away)}},
        "team_stats": {"home": stats_home, "away": stats_away},
        "reliability": reliability,
        "verdict": verdict,
        "perf_breakdown": perf_breakdown,
        "elo_home": round(elo_home),
        "elo_away": round(elo_away),
        "elo_diff": round(elo_home - elo_away),
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
        "confidence": {
            "prob_prono": round(float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100, 1),
            "level": "high" if float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100 >= 12
                     else ("medium" if float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100 >= 8 else "low"),
            "hit_rate": 46 if float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100 >= 12
                        else (28 if float(matrix.matrix[reco.home_goals, reco.away_goals]) * 100 >= 8 else 0),
        },
        "top_scores": top_scores,
        "grid": grid,
        "goal_dist_home": goal_dist(home_axis),
        "goal_dist_away": goal_dist(away_axis),
        "markets": {
            "btts": round(p_btts * 100, 1),
            "over25": round(p_over25 * 100, 1),
            "under25": round((1 - p_over25) * 100, 1),
            "clean_sheet_home": round(p_clean_home * 100, 1),
            "clean_sheet_away": round(p_clean_away * 100, 1),
        },
        "form_home": recent_form(home_snap),
        "form_away": recent_form(away_snap),
        "h2h": {
            "matches": h2h_list,
            "home_wins": h2h_home_wins,
            "draws": h2h_draws,
            "away_wins": h2h_away_wins,
        },
    })


@lru_cache(maxsize=1)
def _compute_bilan():
    """Bilan HONNÊTE de la CdM 2026 : le modèle ne voit que le passé de chaque
    match (méthode backtest, sans tricher). Séparé poules / phases finales."""
    from mpp_predictor.features.elo import HOME_ADVANTAGE
    from mpp_predictor.decision.backtester import _update_elo
    from mpp_predictor.model.predict import predict_match

    cfg, df, _ = _load_state()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    wc = (df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")
             & (df.home_score.notna())].sort_values("date"))

    # numéro de match par équipe -> tour
    count = {}
    stage = []
    for _, m in wc.iterrows():
        count[m.home_team] = count.get(m.home_team, 0) + 1
        count[m.away_team] = count.get(m.away_team, 0) + 1
        mx = max(count[m.home_team], count[m.away_team])
        stage.append("poules" if mx <= 3 else "finales")
    wc = wc.assign(stage=stage)

    elo = compute_elo_history(df[df.date < wc.date.min()]) if not wc.empty else {}

    buckets = {"poules": [], "finales": []}
    for _, m in wc.iterrows():
        real = (int(m.home_score), int(m.away_score))
        hs = build_snapshot(df, m.home_team, m.date, 10, elo_lookup=elo)
        aws = build_snapshot(df, m.away_team, m.date, 10, elo_lookup=elo)
        if len(hs.recent_matches) >= 5 and len(aws.recent_matches) >= 5:
            mx, lh, la, eh, ea = predict_match(cfg, df, elo, m.home_team, m.away_team, m.date)
            reco = recommend_prediction(mx, cfg)
            prono = (reco.home_goals, reco.away_goals)
            same_result = ((prono[0] > prono[1]) == (real[0] > real[1])
                           and (prono[0] < prono[1]) == (real[0] < real[1])
                           and (prono[0] == prono[1]) == (real[0] == real[1]))
            same_diff = (prono[0] - prono[1]) == (real[0] - real[1])
            buckets[m.stage].append({
                "home": m.home_team, "away": m.away_team,
                "real": f"{real[0]}-{real[1]}", "prono": f"{prono[0]}-{prono[1]}",
                "exact": prono == real, "good_result": same_result,
                "good_diff": same_diff, "real_draw": real[0] == real[1],
            })
        _update_elo(elo, m.home_team, m.away_team, real, m, INITIAL_ELO, HOME_ADVANTAGE)

    def summarize(rows):
        n = len(rows)
        if n == 0:
            return {"n": 0, "played": False}
        ex = sum(r["exact"] for r in rows)
        gd = sum(r["good_diff"] and not r["exact"] for r in rows)
        gr = sum(r["good_result"] for r in rows)
        gr_only = sum(r["good_result"] and not r["good_diff"] and not r["exact"] for r in rows)
        pts = ex * 3 + gd * 2 + gr_only * 1
        real_draws = sum(r["real_draw"] for r in rows)
        draws_found = sum(r["real_draw"] and r["exact"] for r in rows)
        return {
            "n": n, "played": True,
            "exact": ex, "exact_pct": round(ex / n * 100, 1),
            "good_diff": gd,
            "good_result": gr, "good_result_pct": round(gr / n * 100, 1),
            "missed": n - gr,
            "points": pts, "ppm": round(pts / n, 2),
            "real_draws": real_draws, "draws_found": draws_found,
            "exact_list": [f"{r['home']} {r['real']} {r['away']}" for r in rows if r["exact"]],
        }

    return {
        "poules": summarize(buckets["poules"]),
        "finales": summarize(buckets["finales"]),
        "last_update": str(df["date"].max().date()),
    }


@app.route("/api/bilan")
def bilan():
    return jsonify(_compute_bilan())


# Affiches du Round of 32 (CdM 2026) — bracket officiel complet au 28/06/2026.
# Ordre exact du tableau (les paires consécutives se retrouvent au tour suivant).
ROUND_OF_32 = [
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("Belgium", "Senegal"),
    ("United States", "Bosnia and Herzegovina"),
    ("Spain", "Austria"),
    ("Portugal", "Croatia"),
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
    ("Australia", "Egypt"),
    ("Argentina", "Cape Verde"),
]


@app.route("/api/bracket")
def bracket():
    """Prédit chaque affiche connue des phases finales (à la date du jour)."""
    from mpp_predictor.model.predict import predict_match
    cfg, df, elo = _load_state()
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    today = pd.Timestamp(pd.Timestamp.now().date())
    from .flags import flag, accent

    def predict_one(home, away):
        mx, lh, la, eh, ea = predict_match(cfg, df, elo, home, away, today)
        reco = recommend_prediction(mx, cfg)
        prono = (reco.home_goals, reco.away_goals)
        if prono[0] > prono[1]:
            winner = home
        elif prono[0] < prono[1]:
            winner = away
        else:
            winner = home if eh >= ea else away
        elo_gap = abs(eh - ea)
        ot = "élevé" if elo_gap < 60 else ("moyen" if elo_gap < 120 else "faible")
        return {
            "home": home, "away": away,
            "home_flag": flag(home), "away_flag": flag(away),
            "home_c1": accent(home), "away_c1": accent(away),
            "elo_home": round(eh), "elo_away": round(ea), "elo_gap": round(elo_gap),
            "score": f"{prono[0]}-{prono[1]}", "winner": winner,
            "is_draw_pred": prono[0] == prono[1],
            "prob": round(float(mx.matrix[prono[0], prono[1]]) * 100, 1),
            "ot_risk": ot,
        }

    # Round of 32
    r32 = []
    for home, away in ROUND_OF_32:
        try:
            r32.append(predict_one(home, away))
        except Exception:
            r32.append(None)

    # Simulation des tours suivants en faisant avancer les vainqueurs.
    def next_round(prev):
        out = []
        for i in range(0, len(prev), 2):
            if i + 1 >= len(prev) or prev[i] is None or prev[i + 1] is None:
                out.append(None); continue
            h = prev[i]["winner"]; a = prev[i + 1]["winner"]
            try:
                out.append(predict_one(h, a))
            except Exception:
                out.append(None)
        return out

    r16 = next_round(r32)
    r8 = next_round(r16)
    r4 = next_round(r8)
    r2 = next_round(r4)

    champion = r2[0]["winner"] if r2 and r2[0] else None
    champion_flag = flag(champion) if champion else None

    NAMES = {16: "32es de finale", 8: "16es de finale", 4: "8es de finale",
             2: "Quarts de finale", 1: "Finale"}
    rounds = []
    for rd in [r32, r16, r8, r4, r2]:
        clean = [m for m in rd if m]
        if not clean:
            continue
        rounds.append({"name": NAMES.get(len(clean), f"Tour ({len(clean)})"),
                       "matches": clean})

    return jsonify({
        "rounds": rounds,
        "champion": champion,
        "champion_flag": champion_flag,
    })


@app.route("/bracket")
def bracket_page():
    return send_from_directory(app.static_folder, "bracket.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
