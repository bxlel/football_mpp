import pandas as pd
import sys
from datetime import timedelta
import kagglehub

def calculate_fatigue(team_name):
    print("Téléchargement/Mise en cache du dataset Kaggle (ça peut prendre 1 min la première fois)...")
    try:
        path = kagglehub.dataset_download('davidcariboo/player-scores')
        
        print("Chargement des fichiers en mémoire...")
        apps = pd.read_csv(f"{path}/appearances.csv", usecols=['player_id', 'date'])
        players = pd.read_csv(f"{path}/players.csv", usecols=['player_id', 'country_of_citizenship'])
    except Exception as e:
        print(f"Erreur technique : {e}")
        sys.exit(1)

    print(f"Calcul de la fatigue pour : {team_name}...")
    team_players = players[players['country_of_citizenship'] == team_name]

    apps['date'] = pd.to_datetime(apps['date'])
    cutoff_date = pd.to_datetime('2022-11-20') - timedelta(days=365)
    recent_apps = apps[(apps['date'] >= cutoff_date) & (apps['date'] < '2022-11-20')]

    merged = recent_apps.merge(team_players, on='player_id')
    matches_per_player = merged.groupby('player_id').size()

    if matches_per_player.empty:
        print(f"❌ Impossible de trouver l'équipe '{team_name}'. Mets bien le nom en anglais (ex: 'Argentina').")
        return

    avg_matches = matches_per_player.nlargest(11).mean()

    print(f"\n==================================================")
    print(f"✅ MOYENNE DU 11 TYPE (CADRES) : {avg_matches:.1f} matchs joués")
    print(f"==================================================")
    print(f"-> Valeur à copier dans data/fatigue_overrides.csv")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m examples.auto_fatigue 'NomDuPays'")
    else:
        calculate_fatigue(sys.argv[1])
