# TP — Régression linéaire multiple : loyers à Antananarivo

## 📁 Contenu

- `TP_Regression_lineaire.ipynb` — notebook complet (Parties 1 à 4), **déjà exécuté** avec
  tous les graphiques et résultats visibles.
- `app.py` — application Streamlit de prédiction (formulaire + carte interactive).
- `data/logements_antananarivo.csv` — jeu de données **simulé** (voir note dans le notebook).
- `data/generate_data.py` — script ayant généré ce jeu de données (à adapter si vous
  collectez vos propres données).
- `model/` — modèle entraîné, scaler et métadonnées, générés par le notebook (Partie 4).
- `requirements.txt` — dépendances Python.

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Utilisation

1. **Ouvrir et exécuter le notebook** (Jupyter ou VS Code) : `TP_Regression_lineaire.ipynb`.
   Il est déjà exécuté, mais vous pouvez le relancer entièrement (`Run All`) — il régénère les
   fichiers du dossier `model/`.
2. **Lancer l'application** :
   ```bash
   streamlit run app.py
   ```
   puis ouvrir l'URL affichée (par défaut http://localhost:8501).

## 🔁 Utiliser vos vraies données

Remplacez `data/logements_antananarivo.csv` par votre propre fichier, avec les mêmes colonnes :

```
quartier, superficie, nombre_chambres, douche_wc, type_d_acces, meuble,
etat_general, latitude, longitude, loyer_mensuel
```

Puis relancez le notebook en entier (`Run All`) : le modèle, le scaler et l'application seront
automatiquement recalculés sur vos données.

## 📝 Ce qui reste à faire / pistes pour aller plus loin

- Remplacer le jeu de données simulé par de vraies données collectées.
- Améliorer la détection de quartier sur la carte avec un vrai fichier GeoJSON (`geopandas`)
  au lieu de la distance au centroïde.
- Ajouter d'autres variables (sécurité, distance au centre-ville, pollution) comme suggéré
  en bonus dans l'énoncé.
- Comparer la régression linéaire à d'autres modèles (Random Forest, Gradient Boosting).
