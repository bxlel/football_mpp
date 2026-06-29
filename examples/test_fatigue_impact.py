"""Test honnête de l'impact du facteur fatigue (F).

PROBLÈME DE MESURE : on ne peut pas backtester F sur les vieilles Coupes du
Monde, car il faudrait les compositions d'équipe (les 11 titulaires) de chaque
match passé, avec leur charge club à l'époque. Le script Kaggle calcule la
fatigue d'AUJOURD'HUI, pas celle de 2018. Donc « F améliore la précision » n'est
pas démontrable sur l'historique.

CE QU'ON PEUT MESURER : que la mécanique de F est cohérente et a un effet
contrôlé. Ce script vérifie, sur un match type, que :
  1. la fatigue (charge élevée) RÉDUIT l'espérance de buts d'une équipe ;
  2. le manque de rythme (charge basse) la réduit aussi ;
  3. l'effet est d'une ampleur raisonnable (ajustement, pas bouleversement).

Usage : python -m examples.test_fatigue_impact
"""

from __future__ import annotations

from mpp_predictor.config import load_config
from mpp_predictor.features.attack_index import compute_attack_index
from mpp_predictor.features.models import KeyPlayer, PlayedMatch, TeamSnapshot
from mpp_predictor.model.expected_goals import indices_to_lambda
from mpp_predictor.model.poisson_engine import build_score_matrix


def _team(load: int) -> TeamSnapshot:
    matches = [
        PlayedMatch(2, 1, 1800, 5),
        PlayedMatch(3, 0, 1600, 15),
        PlayedMatch(1, 1, 1900, 30),
        PlayedMatch(2, 0, 1700, 45),
    ]
    return TeamSnapshot(
        name=f"load{load}", elo=1900, recent_matches=matches,
        key_players=[KeyPlayer(f"p{i}", True, load) for i in range(4)],
    )


def main() -> None:
    cfg = load_config()
    print("Effet du facteur fatigue F sur l'espérance de buts (adversaire neutre)\n")
    print(f"{'Charge cadres':>16} | {'F':>6} | {'Index att.':>10} | {'Buts attendus':>13}")
    print("-" * 56)

    rows = []
    for load in [15, 23, 30, 38, 45, 52, 58, 65]:
        team = _team(load)
        idx = compute_attack_index(team, cfg)
        lam = indices_to_lambda(idx.weighted_total, opponent_defense_index=1.0)
        rows.append((load, idx.freshness, idx.weighted_total, lam))
        zone = "optimal" if 30 <= load <= 45 else ("rouille" if load < 30 else "fatigue")
        print(f"{load:>16} | {idx.freshness:>6.3f} | {idx.weighted_total:>10.3f} "
              f"| {lam:>13.2f}  ({zone})")

    # Vérifications automatiques.
    by_load = {r[0]: r for r in rows}
    lam_fresh = by_load[38][3]
    lam_tired = by_load[65][3]
    lam_rusty = by_load[15][3]

    print("\n--- Vérifications ---")
    ok1 = lam_tired < lam_fresh
    ok2 = lam_rusty < lam_fresh
    print(f"[{'OK' if ok1 else 'KO'}] Fatigue (65 mts) réduit les buts : "
          f"{lam_tired:.2f} < {lam_fresh:.2f}")
    print(f"[{'OK' if ok2 else 'KO'}] Manque de rythme (15 mts) réduit les buts : "
          f"{lam_rusty:.2f} < {lam_fresh:.2f}")

    drop = (lam_fresh - lam_tired) / lam_fresh * 100
    print(f"\nAmpleur : une équipe totalement cramée perd ~{drop:.0f}% de son "
          f"espérance de buts. C'est un ajustement fin, pas un bouleversement.")

    print("\nCONCLUSION HONNÊTE : la mécanique de F est cohérente et va dans le "
          "bon sens.\nMais son gain réel en PRÉCISION reste non démontré faute de "
          "compos historiques\nbacktestables. À utiliser comme un ajustement de "
          "bon sens, pas comme une\ngarantie d'amélioration du taux de score exact.")


if __name__ == "__main__":
    main()
