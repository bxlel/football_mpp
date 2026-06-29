@echo off
REM ============================================================
REM   LANCER LE SITE MPP - double-clique sur ce fichier
REM ============================================================
cd /d "%~dp0"

echo.
echo  =========================================
echo    MPP PRONOS - demarrage du site
echo  =========================================
echo.

REM 1. Telecharger les donnees fraiches (matchs a jour)
echo  [1/3] Mise a jour des donnees...
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv' -OutFile 'data\raw\results.csv'" 2>nul
if errorlevel 1 (
  echo  [!] Pas de connexion - on utilise les donnees existantes.
) else (
  echo  [OK] Donnees a jour.
)

REM 2. S'assurer que les dependances sont installees
echo  [2/3] Verification de l'installation...
pip install -e ".[web]" --quiet 2>nul

REM 3. Ouvrir le navigateur puis lancer le serveur
echo  [3/3] Ouverture du site...
start "" http://localhost:5000
echo.
echo  =========================================
echo    Le site est ouvert dans ton navigateur.
echo    Pour l'ARRETER : ferme cette fenetre.
echo  =========================================
echo.
python -m web.app

pause
