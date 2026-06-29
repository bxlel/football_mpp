# 🚀 Comment mettre le site à jour (Git + Render)

Suis ces étapes dans l'ordre. Tout se fait depuis le dossier du projet.

---

## ÉTAPE 0 — Se placer dans le bon dossier

Ouvre PowerShell dans VS Code et tape :

```
cd C:\Users\BIL45\mpp-predictor
```

Vérifie que tu vois bien les dossiers `web`, `examples`, `src` :

```
dir
```

---

## ÉTAPE 1 — Remplacer les fichiers

Décompresse le zip et remplace TON dossier `mpp-predictor` par celui du zip
(ou copie-colle par-dessus en acceptant de tout remplacer).

---

## ÉTAPE 2 — Pousser sur GitHub

Tape ces commandes une par une :

```
git add .
```

```
git commit -m "Mise a jour : phases finales + bilan"
```

```
git push origin main
```

### ⚠️ Si le push est REFUSÉ (erreur "non-fast-forward" / "rejected")

C'est le blocage habituel. La solution (ta version locale fait autorité) :

```
git merge --abort
```

(si ça dit "no merge to abort", ignore et continue)

```
git push origin main --force
```

Le `--force` écrase GitHub avec ta version. C'est normal et voulu.

---

## ÉTAPE 3 — Render redéploie tout seul

Dès que le push passe, **Render détecte le changement et redéploie
automatiquement**. Tu n'as rien à faire.

Pour vérifier / forcer :
1. Va sur **dashboard.render.com**
2. Clique sur ton service **football-mpp**
3. Tu vois le déploiement en cours (barre de progression)
4. Si rien ne se lance : bouton **"Manual Deploy"** → **"Deploy latest commit"**

Attends 3-5 minutes. Le site récupère aussi les nouveaux matchs au passage
(il retélécharge results.csv automatiquement au démarrage).

---

## ÉTAPE 4 — Vérifier en ligne

Ouvre **https://football-mpp.onrender.com**

- Page d'accueil = analyse d'un match (style broadcast)
- Onglet "Tableau phases finales" = le bracket prédit

⏳ Note : sur le plan gratuit, le site "s'endort" après 15 min d'inactivité.
Le premier chargement après une sieste prend ~30 secondes (c'est normal).

---

## Les pages du site

| URL | Contenu |
|-----|---------|
| `/` | Analyse d'un match (sélection par maillots, prono, stats) |
| `/bracket` | Tableau de prédiction des phases finales |
| `/api/bilan` | Bilan chiffré (JSON) |
| `/api/bracket` | Prédictions du bracket (JSON) |

---

## En cas de souci

- **Push toujours bloqué** : `git pull --no-rebase` puis résous les conflits,
  ou en dernier recours `git push --force`.
- **Render ne démarre pas** : vérifie les logs dans le dashboard Render.
- **Les CSV reviennent en conflit** : le `.gitignore` les ignore maintenant.
  Si certains traînent encore, fais :
  `git rm --cached *.csv` puis commit.

---

## 🟢 LE PLUS SIMPLE : lancer le site en local (sans rien retenir)

Tu n'as PAS besoin de te souvenir des commandes. Il suffit de :

**Double-cliquer sur `LANCER_LE_SITE.bat`**

Ça fait tout automatiquement :
1. Télécharge les matchs à jour
2. Vérifie l'installation
3. Ouvre le site dans ton navigateur (http://localhost:5000)

Pour arrêter : ferme la fenêtre noire.

Les deux onglets sont en haut du site :
- **Analyse d'un match** : choisis 2 équipes (par drapeau), obtiens le prono
- **Tableau phases finales** : la simulation complète jusqu'au champion
