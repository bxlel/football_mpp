import sys
import subprocess
import pandas as pd
from datetime import timedelta
import kagglehub

def get_fatigue(team_name, apps, players):
    team_players = players[players['country_of_citizenship'] == team_name]
    cutoff_date = pd.to_datetime('2022-11-20') - timedelta(days=365)
    recent_apps = apps[(apps['date'] >= cutoff_date) & (apps['date'] < '2022-11-20')]
    merged = recent_apps.merge(team_players, on='player_id')
    matches = merged.groupby('player_id').size()
    
    if matches.empty:
        return 38.0  # Si pas de data, on met l'optimum par défaut (pas de pénalité)
    return matches.nlargest(11).mean()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m examples.magic_predict 'Equipe 1' 'Equipe 2'")
        sys.exit(1)

    t1, t2 = sys.argv[1], sys.argv[2]
    print(f"🤖 [1/4] Initialisation du match {t1} vs {t2}...")
    
    # 1. Créer le template via ton script existant
    subprocess.run(["python", "-m", "examples.make_fatigue_template", t1, t2], stdout=subprocess.DEVNULL)

    print("⏳ [2/4] Aspiration des données Kaggle en sous-marin...")
    path = kagglehub.dataset_download('davidcariboo/player-scores')
    apps = pd.read_csv(f"{path}/appearances.csv", usecols=['player_id', 'date'], parse_dates=['date'])
    players = pd.read_csv(f"{path}/players.csv", usecols=['player_id', 'country_of_citizenship'])

    print("🧠 [3/4] Calcul des Super-Index de fatigue...")
    f1 = get_fatigue(t1, apps, players)
    f2 = get_fatigue(t2, apps, players)
    print(f"   -> {t1} : {f1:.1f} matchs/cadre")
    print(f"   -> {t2} : {f2:.1f} matchs/cadre")

    # 2. Remplir le CSV automatiquement
    csv_path = "data/fatigue_overrides.csv"
    try:
        df_csv = pd.read_csv(csv_path)
        # Mettre à jour les valeurs
        df_csv.loc[df_csv['team'] == t1, 'average_matches_played'] = round(f1, 1)
        df_csv.loc[df_csv['team'] == t2, 'average_matches_played'] = round(f2, 1)
        df_csv.to_csv(csv_path, index=False)
    except FileNotFoundError:
        print("Erreur : le template n'a pas pu être généré.")
        sys.exit(1)

    print("🎯 [4/4] Lancement de la prédiction MPP...")
    print("="*50)
    # 3. Lancer la prédiction finale et afficher le résultat
    subprocess.run(["python", "-m", "examples.predict_match", t1, t2])
