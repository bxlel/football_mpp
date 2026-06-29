"""
BILAN PHASE DE POULES — version "façon SITE".

Contrairement à bilan_poules.py qui rejoue l'histoire match par match (méthode
honnête de backtest), ce script calcule les pronos avec TOUTES les données
actuelles, exactement comme le fait le site web en ligne. C'est le bilan qui
reflète ce que tu as réellement vu sur le site.

NOTE : ce bilan est "optimiste" — le modèle profite de données postérieures
aux matchs. Ce n'est pas une mesure scientifique de la performance du modèle,
mais le reflet de l'expérience réelle sur le site.

Usage :
    python -m examples.bilan_site
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "results.csv"


def run():
    if not CSV_PATH.exists():
        raise SystemExit("results.csv manquant.")

    from mpp_predictor.config import load_config
    from mpp_predictor.ingestion.results_loader import load_results
    from mpp_predictor.features.elo import compute_elo_history
    from mpp_predictor.model.predict import predict_match
    from mpp_predictor.decision.mpp_optimizer import recommend_prediction

    cfg = load_config()
    df = load_results(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

    wc = (df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-06-01")
             & (df.home_score.notna())].sort_values("date"))
    # poules = 3 premiers matchs par équipe
    count = {}; keep = []
    for _, m in wc.iterrows():
        count[m.home_team] = count.get(m.home_team, 0) + 1
        count[m.away_team] = count.get(m.away_team, 0) + 1
        keep.append(max(count[m.home_team], count[m.away_team]) <= 3)
    wc = wc[pd.Series(keep, index=wc.index)]

    # Elo calculé sur TOUTES les données (façon site).
    elo = compute_elo_history(df)
    today = pd.Timestamp(pd.Timestamp.now().date())

    print("#" * 64)
    print("#" + "  BILAN POULES — VERSION SITE (toutes données)".center(62) + "#")
    print("#" * 64)
    print(f"Données jusqu'au : {df['date'].max().date()}")
    print("(pronos calculés comme sur le site : avec tout l'historique actuel)\n")

    rows = []
    for _, m in wc.iterrows():
        real = (int(m.home_score), int(m.away_score))
        try:
            mx, lh, la, eh, ea = predict_match(cfg, df, elo, m.home_team, m.away_team, today)
        except Exception:
            continue
        reco = recommend_prediction(mx, cfg)
        prono = (reco.home_goals, reco.away_goals)
        same_result = ((prono[0] > prono[1]) == (real[0] > real[1])
                       and (prono[0] < prono[1]) == (real[0] < real[1])
                       and (prono[0] == prono[1]) == (real[0] == real[1]))
        same_diff = (prono[0] - prono[1]) == (real[0] - real[1])
        rows.append({
            "home": m.home_team, "away": m.away_team,
            "rh": real[0], "ra": real[1], "ph": prono[0], "pa": prono[1],
            "exact": prono == real, "good_result": same_result, "good_diff": same_diff,
            "real_draw": real[0] == real[1],
        })

    r = pd.DataFrame(rows); n = len(r)
    pts = r.exact.sum()*3 + ((~r.exact)&r.good_diff).sum()*2 + ((~r.exact)&(~r.good_diff)&r.good_result).sum()*1

    print("=" * 64)
    print("  SCORE GLOBAL (façon site)")
    print("=" * 64)
    print(f"Matchs analysés          : {n}")
    print(f"Points MPP (barème 3/2/1): {pts}  (moyenne {pts/n:.2f}/match)")
    print(f"  Scores exacts (3 pts)  : {r.exact.sum():>3}  ({r.exact.mean():.1%})")
    print(f"  Bonne diff. (2 pts)    : {((~r.exact)&r.good_diff).sum():>3}")
    print(f"  Bon résultat (1 pt)    : {((~r.exact)&(~r.good_diff)&r.good_result).sum():>3}")
    print(f"  Ratés (0 pt)           : {(~r.good_result).sum():>3}")
    print(f"  Bons résultats TOTAL   : {r.good_result.sum():>3}  ({r.good_result.mean():.1%})")

    print("\n" + "=" * 64)
    print("  SCORES EXACTS TROUVÉS (façon site)")
    print("=" * 64)
    for _, x in r[r.exact].iterrows():
        print(f"  [3pts] {x.home} {x.rh}-{x.ra} {x.away}")

    print("\n" + "=" * 64)
    print("  COMPARAISON AVEC LE BILAN HONNÊTE")
    print("=" * 64)
    print("  Le bilan 'façon site' trouve souvent quelques scores exacts de plus")
    print("  que le bilan honnête (bilan_poules.py), car le modèle profite ici de")
    print("  données postérieures. La vérité du modèle est entre les deux ;")
    print("  ta vraie expérience sur le site correspond à CE bilan-ci.")


if __name__ == "__main__":
    run()
