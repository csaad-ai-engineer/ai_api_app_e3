"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: streamlit_app/app.py
COMPÉTENCES: C10
═══════════════════════════════════════════════════════════════════════════════

Interface utilisateur GUI - Consommation API

═══════════════════════════════════════════════════════════════════════════════
"""


import os
import streamlit as st
import requests
import jwt
from datetime import datetime, timedelta, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 AI API App E3",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


API_URL    = os.getenv("API_URL", "http://api:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")



# ── Styles personnalisés ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0; }
    .sub-header  { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 0.3rem; }
    .info-box {
        background: #F0F7FF; border-left: 4px solid #2E86AB;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_token():
    return jwt.encode(
        {"sub": "streamlit-app",
         "exp": datetime.now(timezone.utc) + timedelta(hours=1)}, 
        JWT_SECRET, algorithm="HS256"
    )


def call_api(endpoint: str, payload: dict) -> dict:
    token = get_token()
    resp  = requests.post(
        f"{API_URL}/{endpoint}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def check_health() -> dict:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 AI API App E3")
    st.markdown("*Estimation basée sur les données DVF (Demandes de Valeurs Foncières)*")
    st.divider()

    health = check_health()
    status_color = "🟢" if health.get("status") == "ok" else "🔴"
    st.markdown(f"**API** : {status_color} {health.get('status', 'N/A')}")
    st.markdown(f"**Modèle** : {'✅ Chargé' if health.get('model_loaded') else '❌ Non chargé'}")
    st.divider()

    page = st.radio("Navigation",
                    ["🔍 Estimation", "📊 Analyse de marché", "ℹ️ À propos"])


# ── Page : Estimation ──────────────────────────────────────────────────────────
if page == "🔍 Estimation":
    st.markdown('<p class="main-header">🏠 Estimation de prix immobilier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Obtenez une estimation basée sur les données réelles de transactions (DVF)</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Caractéristiques du bien")
        type_local = st.selectbox("Type de bien", ["Appartement", "Maison", "Dépendance"])
        surface    = st.slider("Surface habitable (m²)", 10, 500, 70)
        pieces     = st.slider("Nombre de pièces", 1, 10, 3)
        terrain    = st.number_input("Surface terrain (m²)", 0, 10000, 0,
                                      help="0 pour un appartement")
        departement = st.selectbox("Département",
                                   ["75", "69", "13", "33", "44", "06", "59", "67",
                                    "31", "76", "92", "93", "94"],
                                   index=0)

    with col2:
        st.subheader("📍 Localisation")
        st.markdown('<div class="info-box">Indiquez les coordonnées GPS du bien ou choisissez une ville type.</div>',
                    unsafe_allow_html=True)

        ville_preset = st.selectbox("Ville (pré-rempli)", [
            "Personnalisée", "Paris 75", "Lyon 69", "Marseille 13",
            "Bordeaux 33", "Nantes 44", "Nice 06"
        ])
        coords = {
            "Paris 75":     (2.3522, 48.8566),
            "Lyon 69":      (4.8357, 45.7640),
            "Marseille 13": (5.3698, 43.2965),
            "Bordeaux 33":  (-0.5792, 44.8378),
            "Nantes 44":    (-1.5534, 47.2184),
            "Nice 06":      (7.2620, 43.7102),
        }
        default_lon, default_lat = coords.get(ville_preset, (2.35, 48.85))
        longitude = st.number_input("Longitude", -5.0, 10.0, default_lon, 0.001, format="%.4f")
        latitude  = st.number_input("Latitude",  41.0, 52.0, default_lat, 0.001, format="%.4f")

    st.divider()
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        estimate_btn = st.button("🔮 Estimer le prix", use_container_width=True, type="primary")

    if estimate_btn:
        payload = {
            "surface_reelle_bati":       surface,
            "nombre_pieces_principales": pieces,
            "surface_terrain":           terrain,
            "longitude":                 longitude,
            "latitude":                  latitude,
            "type_local":                type_local,
            "code_departement":          departement,
        }

        with st.spinner("Calcul en cours..."):
            try:
                result = call_api("predict", payload)
                st.success("✅ Estimation calculée !")
                st.divider()

                # Métriques principales
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['prix_estime']:,.0f} €</div>
                        <div class="metric-label">Prix estimé</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['prix_m2']:,.0f} €/m²</div>
                        <div class="metric-label">Prix au m²</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    latence = result.get("latence_ms", 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{latence:.1f} ms</div>
                        <div class="metric-label">Latence API</div>
                    </div>""", unsafe_allow_html=True)

                st.divider()
                # Visualisation intervalle de confiance
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=result["prix_estime"],
                    delta={"reference": result["intervalle_bas"]},
                    number={"suffix": " €", "valueformat": ",.0f"},
                    gauge={
                        "axis":  {"range": [result["intervalle_bas"] * 0.8,
                                            result["intervalle_haut"] * 1.2]},
                        "bar":   {"color": "#1E3A5F"},
                        "steps": [
                            {"range": [result["intervalle_bas"], result["prix_estime"]],
                             "color": "#C8E6FF"},
                            {"range": [result["prix_estime"], result["intervalle_haut"]],
                             "color": "#90C8F0"},
                        ],
                        "threshold": {
                            "line":      {"color": "red", "width": 2},
                            "thickness": 0.75,
                            "value":     result["prix_estime"],
                        },
                    },
                    title={"text": "Fourchette estimée (± 15%)"},
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class="info-box">
                    📊 <b>Intervalle de confiance :</b>
                    {result['intervalle_bas']:,.0f} € – {result['intervalle_haut']:,.0f} €<br>
                    🏷️ <b>Modèle :</b> {result['modele_version']}
                </div>""", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de joindre l'API. Vérifiez que le service est démarré.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Erreur API : {e.response.text}")
            except Exception as e:
                st.error(f"❌ Erreur inattendue : {e}")


# ── Page : Analyse de marché ───────────────────────────────────────────────────
elif page == "📊 Analyse de marché":
    st.subheader("📊 Analyse comparative de marché")
    st.info("Simulez plusieurs scénarios pour comparer les estimations.")

    with st.expander("⚙️ Paramètres de comparaison"):
        dept_compare = st.selectbox("Département", ["75", "69", "13", "33", "44"])
        type_compare = st.selectbox("Type de bien", ["Appartement", "Maison"])

    surfaces = list(range(20, 201, 10))
    token    = get_token()
    prices   = []

    if st.button("Lancer la simulation", type="primary"):
        progress = st.progress(0)
        with st.spinner("Simulation en cours..."):
            batch = [
                {
                    "surface_reelle_bati":       float(s),
                    "nombre_pieces_principales": max(1, s // 25),
                    "surface_terrain":           0.0,
                    "longitude": 2.35, "latitude": 48.85,
                    "type_local": type_compare,
                    "code_departement": dept_compare,
                }
                for s in surfaces
            ]
            try:
                result = requests.post(
                    f"{API_URL}/predict/batch", json=batch,
                    headers={"Authorization": f"Bearer {token}"}, timeout=15
                ).json()
                prices = result["predictions"]
                progress.progress(100)

                df_sim = pd.DataFrame({
                    "Surface (m²)": surfaces,
                    "Prix estimé (€)": prices,
                    "Prix/m² (€)": [p / s for p, s in zip(prices, surfaces)]
                })

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.line(df_sim, x="Surface (m²)", y="Prix estimé (€)",
                                   title=f"Prix selon la surface – {type_compare} Dép. {dept_compare}",
                                   color_discrete_sequence=["#1E3A5F"])
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.line(df_sim, x="Surface (m²)", y="Prix/m² (€)",
                                   title="Prix au m² selon la surface",
                                   color_discrete_sequence=["#2E86AB"])
                    st.plotly_chart(fig2, use_container_width=True)

                st.dataframe(df_sim.style.format({
                    "Prix estimé (€)": "{:,.0f}",
                    "Prix/m² (€)":     "{:,.0f}",
                }), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur : {e}")


# ── Page : À propos ────────────────────────────────────────────────────────────
else:
    st.subheader("ℹ️ À propos du projet")
    st.markdown("""
    ### AI API App E3 – Certification RNCP37827

    Ce projet valide les compétences de l'**épreuve E3** du titre
    **Développeur en Intelligence Artificielle** (Bloc 2, C9–C13).

    #### Stack technique
    | Composant       | Technologie |
    |-----------------|-------------|
    | Modèle ML       | Gradient Boosting Regressor (scikit-learn) |
    | Model Registry  | MLflow |
    | API REST        | FastAPI + JWT |
    | Interface       | Streamlit |
    | Monitoring      | Evidently AI + Prometheus + Grafana |
    | CI/CD           | GitHub Actions |
    | Déploiement     | Azure Container Apps |

    #### Source de données
    **DVF – Demandes de Valeurs Foncières** (data.gouv.fr)
    Données open data des transactions immobilières en France.

    #### Compétences couvertes
    - **C9** : API REST exposant un modèle IA (FastAPI)
    - **C10** : Intégration de l'API dans une application (Streamlit)
    - **C11** : Monitoring du modèle (Evidently AI)
    - **C12** : Tests automatisés (pytest)
    - **C13** : CI/CD et MLOps (GitHub Actions + Azure)
    """)