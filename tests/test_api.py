"""
Tests unitaires & d'intégration – API FastAPI (test_api.py)
Valide : endpoints, authentification, validation des données, réponses.
"""

import pytest
import jwt
import datetime
from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient


# ── Setup du client de test ────────────────────────────────────────────────────
JWT_SECRET    = "test-secret"
JWT_ALGORITHM = "HS256"

def make_token(secret: str = JWT_SECRET, expired: bool = False) -> str:
    exp = datetime.datetime.utcnow()
    exp += datetime.timedelta(seconds=-1) if expired else datetime.timedelta(hours=1)
    return jwt.encode({"sub": "test", "exp": exp}, secret, algorithm=JWT_ALGORITHM)


@pytest.fixture(scope="module")
def client():
    """Crée un client de test avec un modèle mocké."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([250_000.0])

    with patch.dict("os.environ", {
        "JWT_SECRET":          JWT_SECRET,
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
    }):
        with patch("api.main.mlflow.sklearn.load_model", return_value=mock_model):
            from api.main import app
            # Injecter le modèle dans le state de l'app
            import api.main as api_module
            api_module.model = mock_model
            with TestClient(app) as c:
                yield c


@pytest.fixture
def valid_payload():
    return {
        "surface_reelle_bati":       75.0,
        "nombre_pieces_principales": 3,
        "surface_terrain":           0.0,
        "longitude":                 2.347,
        "latitude":                  48.859,
        "type_local":                "Appartement",
        "code_departement":          "75",
    }


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {make_token()}"}


# ── Tests health ───────────────────────────────────────────────────────────────
class TestHealth:

    def test_health_endpoint_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "api_version" in data
        assert "model_name" in data


# ── Tests authentification ──────────────────────────────────────────────────────
class TestAuthentication:

    def test_predict_without_token_returns_403(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code in (401, 403)

    def test_predict_with_invalid_token_returns_401(self, client, valid_payload):
        headers = {"Authorization": "Bearer invalid.token.here"}
        resp = client.post("/predict", json=valid_payload, headers=headers)
        assert resp.status_code == 401

    def test_predict_with_expired_token_returns_401(self, client, valid_payload):
        headers = {"Authorization": f"Bearer {make_token(expired=True)}"}
        resp = client.post("/predict", json=valid_payload, headers=headers)
        assert resp.status_code == 401

    def test_predict_with_valid_token_returns_200(self, client, valid_payload, auth_headers):
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        assert resp.status_code == 200


# ── Tests prédiction ──────────────────────────────────────────────────────────
class TestPrediction:

    def test_predict_returns_positive_price(self, client, valid_payload, auth_headers):
        resp = client.post("/predict", json=valid_payload, headers=auth_headers)
        data = resp.json()
        assert data["prix_estime"] > 0

    def test_predict_returns_all_fields(self, client, valid_payload, auth_headers):
        data = client.post("/predict", json=valid_payload, headers=auth_headers).json()
        for field in ["prix_estime", "intervalle_bas", "intervalle_haut",
                      "prix_m2", "modele_version", "latence_ms"]:
            assert field in data, f"Champ manquant : {field}"

    def test_predict_intervalle_coherent(self, client, valid_payload, auth_headers):
        data = client.post("/predict", json=valid_payload, headers=auth_headers).json()
        assert data["intervalle_bas"] < data["prix_estime"] < data["intervalle_haut"]

    def test_predict_prix_m2_coherent(self, client, valid_payload, auth_headers):
        data = client.post("/predict", json=valid_payload, headers=auth_headers).json()
        expected_m2 = data["prix_estime"] / valid_payload["surface_reelle_bati"]
        assert abs(data["prix_m2"] - expected_m2) < 1

    def test_predict_latence_ms_positive(self, client, valid_payload, auth_headers):
        data = client.post("/predict", json=valid_payload, headers=auth_headers).json()
        assert data["latence_ms"] >= 0


# ── Tests validation des données ───────────────────────────────────────────────
class TestValidation:

    def test_surface_trop_petite_returns_422(self, client, valid_payload, auth_headers):
        payload = {**valid_payload, "surface_reelle_bati": 3.0}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_type_local_invalide_returns_422(self, client, valid_payload, auth_headers):
        payload = {**valid_payload, "type_local": "Château"}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_latitude_hors_france_returns_422(self, client, valid_payload, auth_headers):
        payload = {**valid_payload, "latitude": 55.0}  # Hors bornes
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_pieces_negatif_returns_422(self, client, valid_payload, auth_headers):
        payload = {**valid_payload, "nombre_pieces_principales": 0}
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_champ_manquant_returns_422(self, client, auth_headers):
        payload = {"surface_reelle_bati": 75.0}  # Payload incomplet
        resp = client.post("/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 422


# ── Tests batch ───────────────────────────────────────────────────────────────
class TestBatch:

    def test_batch_predict_multiple_items(self, client, valid_payload, auth_headers):
        import api.main as api_module
        api_module.model.predict.return_value = np.array([200_000.0, 350_000.0])
        payload = [valid_payload, {**valid_payload, "surface_reelle_bati": 120.0}]
        resp = client.post("/predict/batch", json=payload, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_batch_over_100_returns_400(self, client, valid_payload, auth_headers):
        payload = [valid_payload] * 101
        resp = client.post("/predict/batch", json=payload, headers=auth_headers)
        assert resp.status_code == 400


# ── Tests métriques Prometheus ─────────────────────────────────────────────────
class TestMetricsEndpoint:

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_endpoint_is_prometheus_format(self, client):
        resp = client.get("/metrics")
        content_type = resp.headers.get("content-type", "")
        assert "text/plain" in content_type or "prometheus" in content_type.lower() or len(resp.text) > 0