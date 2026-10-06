"""
Application Streamlit — Prédiction du loyer mensuel de logements à Antananarivo
Lancer avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction loyer - Antananarivo", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------
# Chargement du modèle, du scaler et des métadonnées (mis en cache)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/modele_final.pkl")
    scaler = joblib.load("model/scaler.pkl")
    meta = joblib.load("model/meta.pkl")
    return model, scaler, meta


try:
    model, scaler, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Fichiers du modèle introuvables. Exécutez d'abord tout le notebook "
        "`TP_Regression_lineaire.ipynb` (Partie 4) pour générer le dossier `model/`."
    )
    st.stop()

raw_feature_cols = meta["raw_feature_cols"]
selected_features = meta["selected_features"]
quartier_list = meta["quartier_list"]
acces_list = meta["acces_list"]
etat_order = meta["etat_order"]
quartier_centroids = meta["quartier_centroids"]


# ---------------------------------------------------------------
# Reconstruction du vecteur de features (même logique que le notebook)
# ---------------------------------------------------------------
def build_input_row(quartier, superficie, nombre_chambres, douche_wc, type_d_acces, meuble, etat_general):
    row = {c: 0.0 for c in raw_feature_cols}
    row["superficie"] = float(superficie)
    row["nombre_chambres"] = float(nombre_chambres)
    row["meuble_bin"] = 1.0 if meuble == "oui" else 0.0
    row["douche_wc_bin"] = 1.0 if douche_wc == "interieur" else 0.0
    row["etat_general_ord"] = float(etat_order[etat_general])

    qcol = f"quartier_{quartier}"
    if qcol in row:
        row[qcol] = 1.0  # sinon, quartier = catégorie de référence -> tout à 0

    acol = f"type_d_acces_{type_d_acces}"
    if acol in row:
        row[acol] = 1.0

    return pd.DataFrame([row])[raw_feature_cols]


def predict_loyer(quartier, superficie, nombre_chambres, douche_wc, type_d_acces, meuble, etat_general):
    row = build_input_row(quartier, superficie, nombre_chambres, douche_wc, type_d_acces, meuble, etat_general)
    row_scaled = pd.DataFrame(scaler.transform(row), columns=raw_feature_cols)
    pred = model.predict(row_scaled[selected_features])[0]
    return max(pred, 0.0)


# ---------------------------------------------------------------
# État partagé entre les onglets (quartier choisi via la carte)
# ---------------------------------------------------------------
if "quartier_select" not in st.session_state:
    st.session_state.quartier_select = quartier_list[0]

st.title("🏠 Prédiction du loyer mensuel — Antananarivo")
st.caption("Application développée dans le cadre du TP de régression linéaire multiple (modèle : élimination arrière).")

tab_predict, tab_map, tab_model = st.tabs(["📋 Estimer un loyer", "🗺️ Carte interactive", "📊 À propos du modèle"])

# ---------------------------------------------------------------
# Onglet Carte
# ---------------------------------------------------------------
with tab_map:
    st.subheader("Cliquez sur la carte pour choisir un quartier")
    st.write(
        "Le quartier le plus proche du point cliqué est automatiquement sélectionné "
        "dans l'onglet **Estimer un loyer**."
    )

    center = [-18.8792, 47.5079]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    for q, coords in quartier_centroids.items():
        folium.Marker(
            location=[coords["latitude"], coords["longitude"]],
            tooltip=q,
            popup=q,
            icon=folium.Icon(color="blue", icon="home"),
        ).add_to(m)

    map_data = st_folium(m, height=500, use_container_width=True, key="carte")

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        best_q, best_dist = None, float("inf")
        for q, coords in quartier_centroids.items():
            d = (coords["latitude"] - clicked_lat) ** 2 + (coords["longitude"] - clicked_lon) ** 2
            if d < best_dist:
                best_dist, best_q = d, q

        st.session_state.quartier_select = best_q
        st.success(f"Quartier détecté : **{best_q}**")
        st.caption(
            "ℹ️ Détection simplifiée par distance au centroïde du quartier le plus proche. "
            "Une version plus précise utiliserait les limites administratives réelles "
            "(fichier GeoJSON + `geopandas.contains`)."
        )

# ---------------------------------------------------------------
# Onglet Prédiction
# ---------------------------------------------------------------
with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        quartier = st.selectbox(
            "Quartier",
            quartier_list,
            index=quartier_list.index(st.session_state.quartier_select),
        )
        superficie = st.number_input("Superficie (m²)", min_value=10.0, max_value=300.0, value=50.0, step=1.0)
        nombre_chambres = st.number_input("Nombre de chambres", min_value=1, max_value=8, value=2, step=1)
        etat_general = st.selectbox("État général du logement", ["bon", "moyen", "mauvais"])
    with col2:
        douche_wc = st.radio("Douche / WC", ["interieur", "exterieur"], horizontal=True)
        type_d_acces = st.selectbox("Type d'accès", acces_list)
        meuble = st.radio("Meublé ?", ["oui", "non"], horizontal=True)

    st.divider()

    if st.button("💰 Estimer le loyer mensuel", type="primary"):
        pred = predict_loyer(quartier, superficie, nombre_chambres, douche_wc, type_d_acces, meuble, etat_general)
        st.metric("Loyer mensuel estimé", f"{pred:,.0f} Ar".replace(",", " "))
        st.caption(
            f"Marge d'erreur indicative : ± {meta['rmse_test']:,.0f} Ar (RMSE sur le jeu de test)".replace(",", " ")
        )

        st.subheader("Poids des variables du modèle")
        coefs = pd.Series(model.coef_, index=selected_features).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#c0392b" if v < 0 else "#27ae60" for v in coefs]
        coefs.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Poids (coefficient standardisé)")
        ax.set_title("Contribution de chaque variable à la prédiction")
        st.pyplot(fig)

# ---------------------------------------------------------------
# Onglet À propos du modèle
# ---------------------------------------------------------------
with tab_model:
    st.subheader("Performance du modèle final (élimination arrière)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (test)", f"{meta['r2_test']:.3f}")
    c2.metric("RMSE (test)", f"{meta['rmse_test']:,.0f} Ar".replace(",", " "))
    c3.metric("Variables retenues", len(selected_features))

    with st.expander("Voir la liste des variables utilisées par le modèle"):
        st.write(selected_features)

    with st.expander("Voir un échantillon du jeu de données d'entraînement"):
        try:
            df_apercu = pd.read_csv("data/logements_antananarivo.csv")
            st.dataframe(df_apercu.head(20))
        except FileNotFoundError:
            st.info("Fichier data/logements_antananarivo.csv introuvable.")

    st.caption(
        "⚠️ Modèle entraîné sur un jeu de données simulé à but pédagogique — "
        "les montants prédits ne reflètent pas nécessairement le marché réel."
    )
