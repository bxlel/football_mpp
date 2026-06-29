"""
BILAN COMPLET DE LA PHASE DE POULES — Coupe du Monde 2026.

Lance le modèle sur tous les matchs de poules joués et sort un maximum de
métriques : pronos de résultat, scores exacts, performance par type de match,
analyse des nuls, des gros scores, par confédération, etc.

Usage :
    python -m examples.bilan_poules

(Pense à mettre à jour results.csv avant de lancer.)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "results.csv"


def run():
    if not CSV_PATH.exists():
        raise SystemExit("results.csv manquant. Télécharge-le d'abord.")

    from mpp_predictor.config import load_config
    from mpp_predictor.ingestion.results_loader import load_results, build_snapshot
    from mpp_predictor.features.elo import compute_elo_history, INITIAL_ELO, HOME_ADVANTAGE
    from mpp_predictor.model.predict import predict_match
    from mpp_predictor.decision.mpp_optimizer import recommend_prediction
    from mpp_predictor.decision.backtester import _update_elo

    cfg = load_config()
    df = load_results(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    wc = (df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")
             & (df.home_score.notna())].sort_values("date"))

    # Numéro de match par équipe -> phase de poules = matchs 1 à 3.
    count = {}
    keep = []
    for idx, m in wc.iterrows():
        kh = m.home_team; ka = m.away_team
        count[kh] = count.get(kh, 0) + 1
        count[ka] = count.get(ka, 0) + 1
        md = max(count[kh], count[ka])
        keep.append(md <= 3)
    wc = wc[pd.Series(keep, index=wc.index)]

    if wc.empty:
        raise SystemExit("Aucun match de poules 2026 trouvé.")

    print("#" * 64)
    print("#" + "  BILAN PHASE DE POULES — COUPE DU MONDE 2026".center(62) + "#")
    print("#" * 64)
    print(f"Données jusqu'au : {df['date'].max().date()}")

    elo = compute_elo_history(df[df.date < wc.date.min()])

    records = []
    for _, m in wc.iterrows():
        hs = build_snapshot(df, m.home_team, m.date, 10, elo_lookup=elo)
        aws = build_snapshot(df, m.away_team, m.date, 10, elo_lookup=elo)
        real = (int(m.home_score), int(m.away_score))
        if len(hs.recent_matches) < 5 or len(aws.recent_matches) < 5:
            _update_elo(elo, m.home_team, m.away_team, real, m, INITIAL_ELO, HOME_ADVANTAGE)
            continue
        mx, lh, la, eh, ea = predict_match(cfg, df, elo, m.home_team, m.away_team, m.date)
        reco = recommend_prediction(mx, cfg)
        prono = (reco.home_goals, reco.away_goals)
        same_result = ((prono[0] > prono[1]) == (real[0] > real[1])
                       and (prono[0] < prono[1]) == (real[0] < real[1])
                       and (prono[0] == prono[1]) == (real[0] == real[1]))
        same_diff = (prono[0] - prono[1]) == (real[0] - real[1])
        records.append({
            "home": m.home_team, "away": m.away_team,
            "real_h": real[0], "real_a": real[1],
            "prono_h": prono[0], "prono_a": prono[1],
            "total": real[0] + real[1],
            "real_draw": real[0] == real[1],
            "pred_draw": prono[0] == prono[1],
            "exact": prono == real,
            "good_result": same_result,
            "good_diff": same_diff,
            "elo_diff": abs(eh - ea),
            "prob_prono": float(mx.matrix[prono[0], prono[1]]) * 100,
        })
        _update_elo(elo, m.home_team, m.away_team, real, m, INITIAL_ELO, HOME_ADVANTAGE)

    r = pd.DataFrame(records)
    n = len(r)

    # ------- 1. SCORE GLOBAL -------
    print("\n" + "=" * 64)
    print("  1. SCORE GLOBAL")
    print("=" * 64)
    pts = r.exact.sum() * 3 + ((~r.exact) & r.good_diff).sum() * 2 + \
          ((~r.exact) & (~r.good_diff) & r.good_result).sum() * 1
    print(f"Matchs analysés          : {n}")
    print(f"Points MPP (barème 3/2/1): {pts}  (moyenne {pts/n:.2f}/match)")
    print(f"  Scores exacts (3 pts)  : {r.exact.sum():>3}  ({r.exact.mean():.1%})")
    print(f"  Bonne diff. (2 pts)    : {((~r.exact)&r.good_diff).sum():>3}")
    print(f"  Bon résultat (1 pt)    : {((~r.exact)&(~r.good_diff)&r.good_result).sum():>3}")
    print(f"  Ratés (0 pt)           : {(~r.good_result).sum():>3}")
    print(f"  Bons résultats TOTAL   : {r.good_result.sum():>3}  ({r.good_result.mean():.1%})")

    # ------- 2. ANALYSE DES RÉSULTATS -------
    print("\n" + "=" * 64)
    print("  2. RÉSULTATS : RÉALITÉ vs MODÈLE")
    print("=" * 64)
    print(f"  Vrais nuls           : {r.real_draw.sum():>3} ({r.real_draw.mean():.0%})")
    print(f"  Nuls prédits         : {r.pred_draw.sum():>3} ({r.pred_draw.mean():.0%})")
    print(f"  Nuls bien trouvés    : {(r.real_draw & r.exact).sum()} exact, "
          f"{(r.real_draw & r.good_result).sum()} en résultat")
    draws = r[r.real_draw]
    print(f"  Taux de hit sur nuls : {draws.exact.mean():.1%}" if len(draws) else "")
    wins = r[~r.real_draw]
    print(f"  Taux de hit sur vict.: {wins.exact.mean():.1%}")

    # ------- 3. PERF PAR NB DE BUTS -------
    print("\n" + "=" * 64)
    print("  3. PERFORMANCE PAR NOMBRE DE BUTS (réel)")
    print("=" * 64)
    for g in range(0, 7):
        sub = r[r.total == g]
        if len(sub):
            print(f"  {g} but(s) : {len(sub):>2} matchs | "
                  f"{sub.exact.sum()} exacts ({sub.exact.mean():.0%}) | "
                  f"{sub.good_result.mean():.0%} bon résultat")

    # ------- 4. PERF PAR ÉCART ELO -------
    print("\n" + "=" * 64)
    print("  4. PERFORMANCE PAR ÉCART DE FORCE (Elo)")
    print("=" * 64)
    bins = pd.cut(r.elo_diff, [0, 50, 100, 200, 400, 9999],
                  labels=["0-50", "50-100", "100-200", "200-400", "400+"])
    for b, sub in r.groupby(bins, observed=True):
        print(f"  Écart {str(b):>8} : {len(sub):>2} matchs | "
              f"{sub.exact.sum()} exacts ({sub.exact.mean():.0%}) | "
              f"{sub.good_result.mean():.0%} bon résultat")

    # ------- 5. CONFIANCE DU MODÈLE -------
    print("\n" + "=" * 64)
    print("  5. PERFORMANCE PAR NIVEAU DE CONFIANCE")
    print("=" * 64)
    cb = pd.cut(r.prob_prono, [0, 8, 12, 100],
                labels=["faible (<8%)", "moyenne (8-12%)", "élevée (>12%)"])
    for b, sub in r.groupby(cb, observed=True):
        print(f"  Confiance {str(b):<16}: {len(sub):>2} matchs | "
              f"{sub.exact.sum()} exacts ({sub.exact.mean():.0%})")

    # ------- 6. LISTE DES SCORES EXACTS -------
    print("\n" + "=" * 64)
    print("  6. SCORES EXACTS TROUVÉS")
    print("=" * 64)
    for _, x in r[r.exact].iterrows():
        print(f"  [3pts] {x.home} {x.real_h}-{x.real_a} {x.away}")

    # ------- 7. LISTE DES NULS RATÉS -------
    print("\n" + "=" * 64)
    print("  7. NULS QUE LE MODÈLE A RATÉS (son point faible)")
    print("=" * 64)
    missed = r[r.real_draw & ~r.exact]
    for _, x in missed.iterrows():
        print(f"  {x.home} {x.real_h}-{x.real_a} {x.away}  "
              f"(prono {x.prono_h}-{x.prono_a})")

    print("\n" + "#" * 64)
    print("Résumé : le modèle est fort sur les vainqueurs, faible sur les nuls.")
    print("#" * 64)


if __name__ == "__main__":
    run()
