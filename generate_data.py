"""
Génération d'un jeu de données SIMULÉ mais réaliste de logements à louer
à Antananarivo, pour le TP de régression linéaire multiple.

Les quartiers, leurs coordonnées GPS (approximatives) et leur "standing"
sont inspirés de la géographie réelle d'Antananarivo. Les loyers sont
générés selon une formule paramétrique + bruit, afin de garantir des
relations exploitables (mais pas parfaites) pour la régression.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 650  # nombre d'annonces simulées

# ---------------------------------------------------------------
# 1. Quartiers : (nom, standing, latitude, longitude)
#    standing : "haut", "moyen", "populaire" -> influence le prix au m2
# ---------------------------------------------------------------
QUARTIERS = [
    ("Ivandry",         "haut",      -18.8657, 47.5259),
    ("Ankorondrano",    "haut",      -18.8837, 47.5219),
    ("Andraharo",       "haut",      -18.8804, 47.5259),
    ("Ambatobe",        "haut",      -18.8586, 47.5316),
    ("Analakely",       "moyen",     -18.9088, 47.5255),
    ("Antaninarenina",  "moyen",     -18.9126, 47.5259),
    ("Isoraka",         "moyen",     -18.9126, 47.5211),
    ("Ambohijatovo",    "moyen",     -18.9070, 47.5223),
    ("Ankadifotsy",     "moyen",     -18.8975, 47.5165),
    ("Andravoahangy",   "moyen",     -18.9013, 47.5320),
    ("67 Hectares",     "moyen",     -18.9174, 47.5137),
    ("Analamahitsy",    "populaire", -18.8698, 47.5225),
    ("Ankatso",         "populaire", -18.8636, 47.5443),
    ("Itaosy",          "populaire", -18.9083, 47.4611),
    ("Tanjombato",      "populaire", -18.9536, 47.5183),
    ("Ambohimanarina",  "populaire", -18.8756, 47.4975),
]

QUARTIER_INFO = {q[0]: {"standing": q[1], "lat": q[2], "lon": q[3]} for q in QUARTIERS}

STANDING_PRIX_M2 = {  # loyer de base en Ariary / m2 / mois
    "haut": (26000, 36000),
    "moyen": (15000, 22000),
    "populaire": (8000, 13000),
}

DOUCHE_WC = ["interieur", "exterieur"]
TYPE_ACCES = ["sans", "moto", "voiture", "voiture_avec_parking"]
ETAT_GENERAL = ["bon", "moyen", "mauvais"]

rows = []
for _ in range(N):
    nom_quartier = RNG.choice([q[0] for q in QUARTIERS])
    info = QUARTIER_INFO[nom_quartier]
    standing = info["standing"]

    superficie = float(np.clip(RNG.normal(55, 25), 12, 220)).__round__(1)
    nombre_chambres = int(np.clip(round(superficie / 22 + RNG.normal(0, 0.7)), 1, 6))

    # probabilité d'un douche/wc intérieur plus élevée dans les quartiers hauts
    p_int = {"haut": 0.9, "moyen": 0.6, "populaire": 0.25}[standing]
    douche_wc = RNG.choice(DOUCHE_WC, p=[p_int, 1 - p_int])

    # accès véhicule plus fréquent dans les quartiers hauts
    p_acces = {
        "haut": [0.05, 0.15, 0.30, 0.50],
        "moyen": [0.20, 0.30, 0.30, 0.20],
        "populaire": [0.45, 0.35, 0.15, 0.05],
    }[standing]
    type_d_acces = RNG.choice(TYPE_ACCES, p=p_acces)

    p_meuble = {"haut": 0.5, "moyen": 0.25, "populaire": 0.08}[standing]
    meuble = RNG.choice(["oui", "non"], p=[p_meuble, 1 - p_meuble])

    p_etat = {
        "haut": [0.75, 0.22, 0.03],
        "moyen": [0.45, 0.45, 0.10],
        "populaire": [0.20, 0.55, 0.25],
    }[standing]
    etat_general = RNG.choice(ETAT_GENERAL, p=p_etat)

    # ---- formule de génération du loyer ----
    prix_m2_min, prix_m2_max = STANDING_PRIX_M2[standing]
    prix_m2 = RNG.uniform(prix_m2_min, prix_m2_max)

    loyer = prix_m2 * (superficie ** 0.92)
    loyer *= 1 + 0.06 * (nombre_chambres - superficie / 22)  # effet chambre marginal
    loyer *= 1.18 if douche_wc == "interieur" else 1.0
    loyer *= {"sans": 0.95, "moto": 1.0, "voiture": 1.10, "voiture_avec_parking": 1.20}[type_d_acces]
    loyer *= 1.15 if meuble == "oui" else 1.0
    loyer *= {"bon": 1.10, "moyen": 1.0, "mauvais": 0.85}[etat_general]

    # bruit multiplicatif (variabilité du marché / négociation)
    loyer *= RNG.normal(1.0, 0.10)
    loyer = max(loyer, 40000)  # plancher réaliste

    # légère variation GPS autour du centroïde du quartier (annonces dispersées)
    lat = info["lat"] + RNG.normal(0, 0.004)
    lon = info["lon"] + RNG.normal(0, 0.004)

    rows.append({
        "quartier": nom_quartier,
        "superficie": superficie,
        "nombre_chambres": nombre_chambres,
        "douche_wc": douche_wc,
        "type_d_acces": type_d_acces,
        "meuble": meuble,
        "etat_general": etat_general,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "loyer_mensuel": round(loyer, -2),  # arrondi à la centaine d'Ariary
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------
# 2. Injection de valeurs manquantes réalistes (pour la partie 1 du TP)
# ---------------------------------------------------------------
for col, taux in [("superficie", 0.03), ("nombre_chambres", 0.02),
                   ("douche_wc", 0.04), ("etat_general", 0.05),
                   ("type_d_acces", 0.02)]:
    idx = RNG.choice(df.index, size=int(len(df) * taux), replace=False)
    df.loc[idx, col] = np.nan

# quelques doublons (annonces republiées), à détecter/nettoyer dans le TP
doublons = df.sample(8, random_state=1)
df = pd.concat([df, doublons], ignore_index=True)

# quelques outliers extrêmes de superficie (fausses annonces / erreurs de saisie)
out_idx = RNG.choice(df.index, size=4, replace=False)
df.loc[out_idx, "superficie"] = RNG.uniform(300, 450, size=4).round(1)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # mélange

df.to_csv("/home/claude/tp/data/logements_antananarivo.csv", index=False, encoding="utf-8")
print(df.shape)
print(df.head())
print(df.isna().sum())
