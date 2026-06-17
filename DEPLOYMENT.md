# Mettre le site en ligne

Ce projet contient une interface web (`web/`) qui peut tourner **en local** ou
être **déployée en ligne** sur un hébergeur qui accepte Python.

> ⚠️ **Netlify et GitHub Pages ne conviennent PAS** : ils n'hébergent que des
> sites statiques (HTML/JS), pas un backend Python. Utilise Render, Railway ou
> Hugging Face Spaces.

## Tester en local d'abord

```bash
pip install -e ".[web]"
# Télécharger les données si ce n'est pas déjà fait :
curl -sL https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o data/raw/results.csv
python -m web.app
```
Ouvre ensuite http://localhost:5000 dans ton navigateur.

Sous Windows PowerShell, remplace la commande `curl` par :
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" -OutFile "data\raw\results.csv"
```

## Déployer sur Render (gratuit, recommandé)

Render lit automatiquement le fichier `render.yaml` fourni.

1. Pousse ce projet sur un dépôt GitHub.
2. Crée un compte sur https://render.com (connexion via GitHub).
3. Clique **New > Blueprint**, sélectionne ton dépôt. Render détecte
   `render.yaml` et configure tout seul le build et le démarrage.
4. Valide. Au bout de quelques minutes, tu obtiens une URL publique
   (`https://mpp-predictor-xxxx.onrender.com`) que tu peux partager.

Le build télécharge le dataset automatiquement (voir `buildCommand` dans
`render.yaml`).

> Note : sur le plan gratuit de Render, le service "s'endort" après inactivité
> et met ~30 s à se réveiller à la première visite. C'est normal.

## Déployer sur Railway (alternative)

Railway utilise le `Procfile` fourni.

1. Pousse sur GitHub, puis sur https://railway.app : **New Project > Deploy
   from GitHub repo**.
2. Ajoute une étape de build pour télécharger les données, ou commits le CSV
   dans le dépôt (il fait 3,5 Mo).
3. Railway démarre via le `Procfile` et te donne une URL.

## Comment ça marche (architecture)

```
Navigateur (page web statique : web/static/index.html)
        │  appels fetch() vers l'API
        ▼
Serveur Python (Flask : web/app.py)
        │  appelle le modèle
        ▼
Modèle MPP (src/mpp_predictor/...)  +  data/raw/results.csv
```

La page n'est que l'habillage ; tout le calcul reste dans le modèle Python
existant. L'API expose deux routes : `/api/teams` (liste des équipes) et
`/api/predict` (le pronostic).
