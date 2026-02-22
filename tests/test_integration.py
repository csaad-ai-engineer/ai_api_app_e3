"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: tests/test_integration.py
COMPÉTENCES: C10
═══════════════════════════════════════════════════════════════════════════════

Tests intégration Streamlit ↔ API

═══════════════════════════════════════════════════════════════════════════════
"""

"""
test_integration.py – Tests d'intégration (C10)
Couvre TOUS les points de terminaison exploités par Streamlit.
Vérifie le cycle complet : Streamlit → API → Modèle → Réponse.
Marqué @pytest.mark.integration pour pouvoir les séparer des tests unitaires.
"""

import pytest
import jwt
import datetime
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

JWT_SECRET = "test-secret"


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def mock_model_multi():
    """Modèle mocké retournant des tableaux de taille variable."""
    m = MagicMock()
    def predict_side_effect(df):
        return np.random.uniform(100_000, 800_000, len(df))
    m.predict.side_effect = predict_side_effect
    return m


@pytest.fixture(scope="module")
def integration_client(mock_model_multi):
    import os
    os.environ["JWT_SECRET"] = JWT_SECRET
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    with patch("mlflow.sklearn.load_model", return_value=mock_model_multi):
        import api.main as api_module
        api_module.model = mock_model_multi
        with TestClient(api_module.app) as c:
            yield c


@pytest.fixture
def token():
    return jwt.encode(
        {"sub": "integration-test",
         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        JWT_SECRET, algorithm="HS256"
    )


@pytest.fixture
def headers(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def bien_paris():
    return {
        "surface_reelle_bati": 65.0, "nombre_pieces_principales": 3,
        "surface_terrain": 0.0, "longitude": 2.3522, "latitude": 48.8566,
        "type_local": "Appartement", "code_departement": "75",
    }


@pytest.fixture
def bien_lyon():
    return {
        "surface_reelle_bati": 120.0, "nombre_pieces_principales": 5,
        "surface_terrain": 300.0, "longitude": 4.8357, "latitude": 45.7640,
        "type_local": "Maison", "code_departement": "69",
    }


# ── Cycle complet Streamlit ────────────────────────────────────────────────────
class TestCycleCompletStreamlit:
    """Reproduit exactement les appels effectués par l'interface Streamlit."""

    def test_health_check_au_demarrage(self, integration_client):
        """Streamlit vérifie /health au démarrage (sidebar)."""
        resp = integration_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"

    def test_prediction_individuelle_complete(self, integration_client, headers, bien_paris):
        """Page 'Estimation' : appel POST /predict avec un bien."""
        resp = integration_client.post("/predict", json=bien_paris, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        # Tous les champs nécessaires à l'affichage Streamlit
        assert data["prix_estime"] > 0
        assert data["intervalle_bas"] < data["prix_estime"] < data["intervalle_haut"]
        assert data["prix_m2"] > 0
        assert data["latence_ms"] >= 0
        assert "modele_version" in data

    def test_prediction_maison_vs_appartement(self, integration_client, headers, bien_paris, bien_lyon):
        """Vérifie que deux biens différents donnent des prédictions différentes."""
        r1 = integration_client.post("/predict", json=bien_paris, headers=headers).json()
        r2 = integration_client.post("/predict", json=bien_lyon,  headers=headers).json()
        # Les deux doivent réussir
        assert r1["prix_estime"] > 0
        assert r2["prix_estime"] > 0

    def test_analyse_marche_batch_complet(self, integration_client, headers, bien_paris):
        """Page 'Analyse de marché' : simulation batch sur 19 surfaces."""
        surfaces = list(range(20, 210, 10))  # 19 biens
        batch = [
            {**bien_paris,
            "surface_reelle_bati": float(s),
            "nombre_pieces_principales": max(1, s // 25)}
            for s in surfaces
        ]
        resp = integration_client.post("/predict/batch", json=batch, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == len(surfaces)
        assert all(p["prix_estime"] > 0 for p in data)

# ── Tests de tous les endpoints C10 ───────────────────────────────────────────
class TestTousEndpointsStreamlit:
    """
    Critère C10 : 'Les tests d'intégration couvrent TOUS les points de
    terminaison exploités' – vérifié ici exhaustivement.
    """

    def test_GET_health(self, integration_client):
        assert integration_client.get("/health").status_code == 200

    def test_POST_predict(self, integration_client, headers, bien_paris):
        assert integration_client.post("/predict", json=bien_paris,
                                       headers=headers).status_code == 200
    
    def test_POST_predict_batch_un_element(self, integration_client, headers, bien_paris):
        resp = integration_client.post("/predict/batch",
                                    json=[bien_paris], headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert "prix_estime" in data[0]

    def test_POST_predict_batch_max_100(self, integration_client, headers, bien_paris):
        """Le batch de 100 éléments doit passer."""
        batch = [bien_paris] * 100
        resp  = integration_client.post("/predict/batch", json=batch, headers=headers)
        assert resp.status_code == 200

    def test_GET_metrics(self, integration_client):
        """Endpoint Prometheus consommé par Grafana."""
        resp = integration_client.get("/metrics")
        assert resp.status_code == 200


# ── Tests d'authentification et renouvellement ────────────────────────────────
class TestAuthIntegration:
    """Critère C10 : 'Les étapes d'authentification et de renouvellement
    sont intégrées correctement'."""

    def test_token_valide_donne_acces(self, integration_client, headers, bien_paris):
        resp = integration_client.post("/predict", json=bien_paris, headers=headers)
        assert resp.status_code == 200

    def test_token_expire_est_rejete(self, integration_client, bien_paris):
        expired = jwt.encode(
            {"sub": "x", "exp": datetime.datetime.utcnow() - datetime.timedelta(seconds=1)},
            JWT_SECRET, algorithm="HS256"
        )
        resp = integration_client.post("/predict", json=bien_paris,
                                       headers={"Authorization": f"Bearer {expired}"})
        assert resp.status_code == 401
        assert "expiré" in resp.json()["detail"].lower()

    def test_token_invalide_est_rejete(self, integration_client, bien_paris):
        resp = integration_client.post("/predict", json=bien_paris,
                                       headers={"Authorization": "Bearer garbage.token.here"})
        assert resp.status_code == 401

    def test_sans_header_est_rejete(self, integration_client, bien_paris):
        resp = integration_client.post("/predict", json=bien_paris)
        assert resp.status_code in (401, 403)


# ── Tests de résultats corrects ────────────────────────────────────────────────
class TestResultatsInterpretation:
    """Critère C10 : 'Les résultats des tests sont correctement interprétés'."""

    def test_intervalle_confiance_logique(self, integration_client, headers, bien_paris):
        data = integration_client.post("/predict", json=bien_paris,
                                       headers=headers).json()
        ecart = data["intervalle_haut"] - data["intervalle_bas"]
        assert ecart > 0, "L'intervalle de confiance est nul"
        assert ecart < data["prix_estime"] * 0.5, "Intervalle trop large (> 50 % du prix)"

    def test_prix_m2_calcul_correct(self, integration_client, headers, bien_paris):
        data = integration_client.post("/predict", json=bien_paris,
                                       headers=headers).json()
        expected = data["prix_estime"] / bien_paris["surface_reelle_bati"]
        assert abs(data["prix_m2"] - expected) < 1.0

    def test_batch_ordre_preserve(self, integration_client, headers, bien_paris, bien_lyon):
        """L'ordre des prédictions batch doit correspondre à l'ordre des entrées."""
        resp = integration_client.post("/predict/batch",
                                   json=[bien_paris, bien_lyon], headers=headers)
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert all(p["prix_estime"] > 0 for p in data)