"""
Extraction COMPLÈTE des features sur les 661 matchs de backtest.

Ce script génère deux fichiers :
  - analysis/full_features.csv   : toutes les features pour les 661 matchs
  - analysis/hits_only.csv       : les 108 scores exacts uniquement

Features extraites :
  Identité du match, scores, Elo, forme récente, repos, journée,
  lambdas Poisson, probabilités de tous les scores proches, métriques
  de confiance du modèle, contexte historique des confrontations.

Usage :
    python -m examples.extract_features

Puis ouvre full_features.csv dans Excel pour explorer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "results.csv"
OUT_DIR  = ROOT / "analysis"


def run():
    OUT_DIR.mkdir(exist_ok=True)

    from mpp_predictor.config import load_config
    from mpp_predictor.ingestion.results_loader import load_results, build_snapshot
    from mpp_predictor.features.elo import compute_elo_history, INITIAL_ELO, HOME_ADVANTAGE
    from mpp_predictor.model.predict import predict_match
    from mpp_predictor.decision.mpp_optimizer import recommend_prediction
    from mpp_predictor.decision.backtester import _update_elo

    cfg = load_config()
    df  = load_results(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    SETS = {
        "FIFA World Cup": "2010-01-01",
        "UEFA Euro":      "2008-01-01",
        "Copa América":   "2011-01-01",
    }

    records = []

    for tour, since in SETS.items():
        test = (df[(df.tournament == tour) & (df.date >= since)
                   & (df.home_score.notna())]
                .sort_values("date"))
        pre  = df[df.date < test.date.min()]
        elo  = compute_elo_history(pre)

        # Numéro de journée par équipe dans ce tournoi
        team_match_count: dict[tuple, int] = {}

        for idx, m in test.iterrows():
            home, away = m.home_team, m.away_team
            date       = m.date
            year       = date.year

            # Compteur de journée
            key_h = (year, home); key_a = (year, away)
            team_match_count[key_h] = team_match_count.get(key_h, 0) + 1
            team_match_count[key_a] = team_match_count.get(key_a, 0) + 1
            matchday_h = team_match_count[key_h]
            matchday_a = team_match_count[key_a]
            matchday   = max(matchday_h, matchday_a)
            is_group   = int(matchday <= 3)
            is_ko      = int(matchday >= 4)

            # Snapshots
            hs  = build_snapshot(df, home, date, 10, elo_lookup=elo)
            aws = build_snapshot(df, away, date, 10, elo_lookup=elo)

            real = (int(m.home_score), int(m.away_score))
            _update_elo(elo, home, away, real, m, INITIAL_ELO, HOME_ADVANTAGE)

            if len(hs.recent_matches) < 5 or len(aws.recent_matches) < 5:
                continue

            # Prédiction complète
            mx, lh, la, eh, ea = predict_match(cfg, df, elo, home, away, date)
            reco = recommend_prediction(mx, cfg)
            mat  = mx.matrix
            prono = (reco.home_goals, reco.away_goals)
            hit   = int(prono == real)

            # Forme récente des deux équipes
            def form(snap, n=5):
                ms = sorted(snap.recent_matches, key=lambda x: x.days_ago)[:n]
                wins = sum(1 for x in ms if x.goals_for > x.goals_against)
                draws = sum(1 for x in ms if x.goals_for == x.goals_against)
                losses = sum(1 for x in ms if x.goals_for < x.goals_against)
                gf = sum(x.goals_for for x in ms)
                ga = sum(x.goals_against for x in ms)
                return wins, draws, losses, gf/max(1,len(ms)), ga/max(1,len(ms))

            wh,dh,lh_,gfh,gah = form(hs)
            wa,da,la_,gfa,gaa = form(aws)

            # Jours de repos (dernier match avant cette date)
            def rest(snap):
                ms = sorted(snap.recent_matches, key=lambda x: x.days_ago)
                return ms[0].days_ago if ms else 99

            rest_h = rest(hs)
            rest_a = rest(aws)

            # Face-à-face historique (avant cette date)
            h2h = df[
                ((df.home_team==home)&(df.away_team==away)|
                 (df.home_team==away)&(df.away_team==home))
                & (df.date < date)
            ]
            h2h_total = len(h2h)
            h2h_home_wins = len(h2h[(h2h.home_team==home)&(h2h.home_score>h2h.away_score)]) + \
                            len(h2h[(h2h.away_team==home)&(h2h.away_score>h2h.home_score)])
            h2h_draws = len(h2h[h2h.home_score==h2h.away_score])

            # Proba des scores autour du vrai score
            def p(i,j):
                if i<0 or j<0 or i>=mat.shape[0] or j>=mat.shape[1]: return 0.0
                return round(float(mat[i,j])*100,2)

            rh, ra = real
            prob_real   = p(rh, ra)
            prob_real_p1h = p(rh+1, ra)    # 1 but de plus à domicile
            prob_real_m1h = p(rh-1, ra)    # 1 but de moins à domicile
            prob_real_p1a = p(rh, ra+1)
            prob_real_m1a = p(rh, ra-1)
            prob_00 = p(0,0); prob_10 = p(1,0); prob_01 = p(0,1)
            prob_11 = p(1,1); prob_21 = p(2,1); prob_12 = p(1,2)
            prob_20 = p(2,0); prob_02 = p(0,2)

            # Métrique de confiance
            prob_prono = round(float(mat[prono[0], prono[1]])*100, 2)
            top1_prob  = round(float(mat.max())*100, 2)

            # Entropie de la distribution (mesure d'incertitude)
            import numpy as np
            flat = mat.flatten()
            flat = flat[flat > 0]
            entropy = -float(np.sum(flat * np.log(flat)))

            records.append({
                # Identité
                "tournament":       tour,
                "year":             year,
                "month":            date.month,
                "date":             date.strftime("%Y-%m-%d"),
                "home":             home,
                "away":             away,
                # Score réel
                "real_h":           rh,
                "real_a":           ra,
                "total_goals":      rh+ra,
                "goal_diff":        rh-ra,
                "goal_diff_abs":    abs(rh-ra),
                "is_draw":          int(rh==ra),
                "home_win":         int(rh>ra),
                "away_win":         int(rh<ra),
                # Prono modèle
                "prono_h":          prono[0],
                "prono_a":          prono[1],
                "hit":              hit,
                "correct_result":   int((prono[0]>prono[1])==(rh>ra) and (prono[0]<prono[1])==(rh<ra) and (prono[0]==prono[1])==(rh==ra)),
                # Elo
                "elo_home":         round(eh),
                "elo_away":         round(ea),
                "elo_diff":         round(eh-ea),
                "elo_diff_abs":     round(abs(eh-ea)),
                "elo_ratio":        round(eh/ea, 3),
                "elo_home_rank":    None,  # à enrichir si classement FIFA dispo
                # Lambdas
                "lam_home":         round(lh,3),
                "lam_away":         round(la,3),
                "lam_total":        round(lh+la,3),
                "lam_diff":         round(lh-la,3),
                "lam_diff_abs":     round(abs(lh-la),3),
                "lam_ratio":        round(lh/la if la>0 else 0, 3),
                # Probabilités des issues
                "prob_home_win":    round(mx.prob_home_win()*100,2),
                "prob_draw":        round(mx.prob_draw()*100,2),
                "prob_away_win":    round(mx.prob_away_win()*100,2),
                "prob_favorite_win":round(max(mx.prob_home_win(),mx.prob_away_win())*100,2),
                # Proba du vrai score et scores autour
                "prob_real_score":  prob_real,
                "prob_real_p1h":    prob_real_p1h,
                "prob_real_m1h":    prob_real_m1h,
                "prob_real_p1a":    prob_real_p1a,
                "prob_real_m1a":    prob_real_m1a,
                # Scores courants
                "prob_00":          prob_00,
                "prob_10":          prob_10,
                "prob_01":          prob_01,
                "prob_11":          prob_11,
                "prob_20":          prob_20,
                "prob_02":          prob_02,
                "prob_21":          prob_21,
                "prob_12":          prob_12,
                # Confiance modèle
                "prob_prono":       prob_prono,
                "top1_prob":        top1_prob,
                "entropy":          round(entropy,4),
                "confidence_score": round(prob_prono / entropy * 10, 4) if entropy>0 else 0,
                # Forme récente (5 derniers matchs)
                "home_wins_5":      wh, "home_draws_5": dh, "home_losses_5": lh_,
                "home_gf_avg_5":    round(gfh,2), "home_ga_avg_5": round(gah,2),
                "away_wins_5":      wa, "away_draws_5": da, "away_losses_5": la_,
                "away_gf_avg_5":    round(gfa,2), "away_ga_avg_5": round(gaa,2),
                "form_diff_wins":   wh-wa,
                "form_diff_gf":     round(gfh-gfa,2),
                # Repos
                "rest_home":        rest_h,
                "rest_away":        rest_a,
                "rest_diff":        rest_h-rest_a,
                # Journée
                "matchday":         matchday,
                "is_group_stage":   is_group,
                "is_knockout":      is_ko,
                # Face-à-face
                "h2h_total":        h2h_total,
                "h2h_home_wins":    h2h_home_wins,
                "h2h_draws":        h2h_draws,
                "h2h_home_winrate": round(h2h_home_wins/h2h_total,3) if h2h_total>0 else 0.5,
            })

    out = pd.DataFrame(records)
    out.to_csv(OUT_DIR / "full_features.csv", index=False)
    out[out.hit==1].to_csv(OUT_DIR / "hits_only.csv", index=False)

    print(f"✅ {len(out)} matchs extraits")
    print(f"✅ {out.hit.sum()} scores exacts (hits)")
    print(f"✅ {len(out.columns)} features par match")
    print(f"")
    print(f"Fichiers créés dans le dossier 'analysis/' :")
    print(f"  - full_features.csv  ({len(out)} lignes x {len(out.columns)} colonnes)")
    print(f"  - hits_only.csv      ({out.hit.sum()} lignes)")
    print(f"")
    print(f"TOP 5 features les plus corrélées avec les hits :")
    num_cols = out.select_dtypes(include='number').columns.tolist()
    num_cols = [c for c in num_cols if c != 'hit']
    corrs = out[num_cols].corrwith(out['hit']).abs().sort_values(ascending=False)
    for feat, corr in corrs.head(10).items():
        direction = "+" if out[num_cols].corrwith(out['hit'])[feat] > 0 else "-"
        print(f"  {direction} {feat:<30}: {corr:.3f}")


if __name__ == "__main__":
    run()
