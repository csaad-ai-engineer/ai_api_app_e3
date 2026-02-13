"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: tests/conftest.py
COMPÉTENCES: C9, C10, C12
═══════════════════════════════════════════════════════════════════════════════

Fixtures partagées - Données + Modèle + JWT

═══════════════════════════════════════════════════════════════════════════════
"""

"""
conftest.py – Fixtures partagées pour tous les tests
Fournit : données synthétiques DVF, modèle entraîné, client API, token JWT.
"""

import pytest
import datetime
import numpy as np
import pandas as pd
import jwt
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

JWT_SECRET    = "test-secret"
JWT_ALGORITHM = "HS256"

NUMERIC_FEATURES     = ["surface_reelle_bati", "nombre_pieces_principales",
                         "surface_terrain", "longitude", "latitude"]
CATEGORICAL_FEATURES = ["type_local", "code_departement"]
TARGET               = "valeur_fonciere"
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ── Données ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def dvf_dataframe():
    """Jeu de données DVF synthétique représentatif."""
    np.random.seed(42)
    n = 500
    surfaces = np.random.uniform(20, 250, n)
    pieces   = np.random.randint(1, 8, n)
    return pd.DataFrame({
        "surface_reelle_bati":       surfaces,
        "nombre_pieces_principales": pieces,
        "surface_terrain":           np.random.uniform(0, 600, n),
        "longitude":                 np.random.uniform(-1.5, 7.5, n),
        "latitude":                  np.random.uniform(43.0, 51.0, n),
        "type_local":                np.random.choice(["Appartement", "Maison", "Dépendance"], n,
                                                       p=[0.65, 0.30, 0.05]),
        "code_departement":          np.random.choice(["75", "69", "13", "33", "44", "06",
                                                       "59", "67", "31", "92"], n),
        # Prix simulé avec logique réaliste (prix/m² variable)
        "valeur_fonciere": surfaces * np.random.uniform(2500, 9000, n)
                           + pieces * np.random.uniform(3000, 8000, n)
                           + np.random.normal(0, 15000, n),
    })


@pytest.fixture(scope="session")
def dvf_reference(dvf_dataframe):
    """70 % du jeu comme données de référence (entraînement)."""
    return dvf_dataframe.iloc[:350].copy()


@pytest.fixture(scope="session")
def dvf_current(dvf_dataframe):
    """30 % du jeu comme données courantes (production simulée)."""
    return dvf_dataframe.iloc[350:].copy()


@pytest.fixture(scope="session")
def dvf_drifted(dvf_dataframe):
    """Jeu de données avec drift artificiel pour tester la détection."""
    df = dvf_dataframe.iloc[350:].copy()
    # Drift : surfaces augmentées de 30 % (marché qui évolue)
    df["surface_reelle_bati"] = df["surface_reelle_bati"] * 1.3
    # Drift : déséquilibre géographique (plus de Paris)
    df.loc[df.index[:50], "code_departement"] = "75"
    return df


# ── Modèle ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def trained_pipeline(dvf_reference):
    """Pipeline scikit-learn entraîné sur les données de référence."""
    import sys; sys.path.insert(0, ".")
    from ml.train import build_pipeline
    X = dvf_reference[ALL_FEATURES]
    y = dvf_reference[TARGET].clip(lower=10_000)  # Seuil min cohérent
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def mock_model():
    """Modèle mocké pour les tests d'API (sans MLflow)."""
    m = MagicMock()
    m.predict.return_value = np.array([250_000.0])
    return m


# ── Tokens JWT ─────────────────────────────────────────────────────────────────
@pytest.fixture
def valid_token():
    return jwt.encode(
        {"sub": "test-user",
         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        JWT_SECRET, algorithm=JWT_ALGORITHM
    )


@pytest.fixture
def expired_token():
    return jwt.encode(
        {"sub": "test-user",
         "exp": datetime.datetime.utcnow() - datetime.timedelta(seconds=1)},
        JWT_SECRET, algorithm=JWT_ALGORITHM
    )


@pytest.fixture
def auth_headers(valid_token):
    return {"Authorization": f"Bearer {valid_token}"}


# ── Client API ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def api_client(mock_model):
    """Client FastAPI de test avec modèle mocké."""
    import os
    os.environ["JWT_SECRET"] = JWT_SECRET
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("mlflow.sklearn.load_model", lambda uri: mock_model)
        import importlib, api.main as api_module
        api_module.model = mock_model
        with TestClient(api_module.app) as c:
            yield c


# ── Payload de prédiction ──────────────────────────────────────────────────────
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