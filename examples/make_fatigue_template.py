"""Générer un modèle de fichier fatigue à remplir pour un match.

Le critère fatigue (F) a besoin de la charge en matchs/an des cadres de chaque
équipe. Ces données (minutes en club) ne sont pas dans le dataset de résultats
international et ne sont pas téléchargeables automatiquement de façon fiable
(FBref limite le scraping, Kaggle demande un compte). La voie honnête et
précise est donc la saisie manuelle des 4-5 stars de chaque équipe.

Ce script génère un fichier pré-rempli avec des lignes vides à compléter, pour
que ce soit rapide. Tu remplaces juste les noms et le nombre de matchs joués
en club sur l'année (cherche "<joueur> stats" sur FBref ou Transfermarkt).

Usage :
    python -m examples.make_fatigue_template "Portugal" "DR Congo"
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fatigue_overrides.csv"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage : python -m examples.make_fatigue_template "Équipe A" ["Équipe B" ...]')
        raise SystemExit(1)

    teams = sys.argv[1:]
    lines = ["team,player,is_offensive,club_matches_last_year"]
    for team in teams:
        # 5 lignes par équipe : 3 offensifs, 2 défensifs (gardien inclus).
        for i in range(3):
            lines.append(f"{team},NOM_ATTAQUANT_{i+1},true,40")
        for i in range(2):
            lines.append(f"{team},NOM_DEFENSEUR_{i+1},false,40")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Modèle écrit dans : {OUT_PATH}")
    print("\nProchaine étape :")
    print("  1. Ouvre ce fichier et remplace les NOM_... par les vraies stars.")
    print("  2. Mets le vrai nombre de matchs joués en club cette saison")
    print("     (rappel : ~38 = idéal, >50 = fatigue, <25 = manque de rythme).")
    print("  3. Relance : python -m examples.predict_match \"Équipe A\" \"Équipe B\"")


if __name__ == "__main__":
    main()
