"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: api/main.py
COMPÉTENCES: C9
═══════════════════════════════════════════════════════════════════════════════

API REST FastAPI + JWT + Prometheus

═══════════════════════════════════════════════════════════════════════════════
"""

"""
API REST FastAPI - Prédiction de prix immobilier
Expose le modèle MLflow en Production via des endpoints REST sécurisés.
Intègre : authentification JWT, logging structuré, métriques Prometheus.
"""

from datetime import datetime, timedelta, timezone
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import jwt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger("ai-api-app-api")

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow-production-e3fe.up.railway.app")
MODEL_NAME          = os.getenv("MODEL_NAME", "ai-api-app-model")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")
JWT_SECRET          = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM       = "HS256"
API_VERSION         = "1.0.0"

# ── Métriques Prometheus ───────────────────────────────────────────────────────
PREDICTIONS_TOTAL   = Counter("predictions_total", "Nombre total de prédictions")
PREDICTION_ERRORS   = Counter("prediction_errors_total", "Prédictions en erreur")
PREDICTION_LATENCY  = Histogram("prediction_latency_seconds",
                                "Latence des prédictions",
                                buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5])
PREDICTION_PRICE    = Histogram("prediction_price_euros",
                                "Distribution des prix prédits",
                                buckets=[50_000, 100_000, 150_000, 200_000,
                                         300_000, 500_000, 1_000_000])

# ── Modèle global ──────────────────────────────────────────────────────────────
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle MLflow au démarrage."""
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # ANCIEN (avec stages)
    # model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    
    # NOUVEAU (avec alias)
    model_uri = f"models:/{MODEL_NAME}@champion"
    
    logger.info(f"Chargement du modèle : {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur chargement modèle : {e}")
        # Fallback : charge la dernière version
        try:
            model_uri_fallback = f"models:/{MODEL_NAME}/latest"
            model = mlflow.sklearn.load_model(model_uri_fallback)
            logger.info(f"Modèle chargé (fallback) : {model_uri_fallback}")
        except Exception as e2:
            logger.error(f"Fallback échoué : {e2}")
    yield
    logger.info("Arrêt de l'API")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI API App E3 API",
    description="API de prédiction de prix immobilier basée sur les données DVF",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ── Schémas Pydantic ───────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    surface_reelle_bati:       float = Field(..., gt=5,   le=2000, description="Surface habitable en m²")
    nombre_pieces_principales: int   = Field(..., ge=1,   le=20,   description="Nombre de pièces principales")
    surface_terrain:           float = Field(..., ge=0,   le=50000,description="Surface du terrain en m²")
    longitude:                 float = Field(..., ge=-5,  le=10,   description="Longitude (France métropolitaine)")
    latitude:                  float = Field(..., ge=41,  le=52,   description="Latitude (France métropolitaine)")
    type_local:                str   = Field(..., description="Type : Maison | Appartement | Dépendance")
    code_departement:          str   = Field(..., min_length=2, max_length=3, description="Code département (ex: 75)")

    @validator("type_local")
    def validate_type(cls, v):
        allowed = {"Maison", "Appartement", "Dépendance"}
        if v not in allowed:
            raise ValueError(f"type_local doit être parmi : {allowed}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "surface_reelle_bati": 75.0,
                "nombre_pieces_principales": 3,
                "surface_terrain": 100.0,
                "longitude": 2.347,
                "latitude": 48.859,
                "type_local": "Appartement",
                "code_departement": "75"
            }
        }


class PredictionResponse(BaseModel):
    prix_estime:      float = Field(..., description="Prix estimé en euros")
    intervalle_bas:   float = Field(..., description="Borne basse de l'intervalle à 80%")
    intervalle_haut:  float = Field(..., description="Borne haute de l'intervalle à 80%")
    prix_m2:          float = Field(..., description="Prix au m² estimé")
    modele_version:   str
    latence_ms:       float


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_name:     str
    model_stage:    str
    api_version:    str

# ─────────── Modèle pour la requête token ───────────
class LoginRequest(BaseModel):
    username: str
    password: str

# Base simple d'utilisateurs pour tests 
USERS_DB = { "admin": "admin", "user": "user" }

# ── Authentification JWT ───────────────────────────────────────────────────────
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET,
                             algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

# ── Endpoint token ─────────────────────────────────────────────────────────────
# ─────────── Endpoint pour générer un token ─────────── 
@app.post("/token") 
def generate_token(data: LoginRequest): 
    # Vérification de l'utilisateur et mot de passe 
    if data.username in USERS_DB and USERS_DB[data.username] == data.password: 
        payload = { "username": data.username, 
                   "exp": datetime.now(timezone.utc) + timedelta(hours=1)  # token valide 1h # token valide 1h 
                   } 
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM) 
        return {"access_token": token} 
    else: 
        raise HTTPException(status_code=401, detail="Utilisateur ou mot de passe invalide")

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        api_version=API_VERSION,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Endpoint Prometheus pour le scraping des métriques."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(
    request: PredictionRequest,
    _token: dict = Depends(verify_token),
):
    if model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    start = time.perf_counter()
    try:
        features = pd.DataFrame([request.dict()])
        prix = float(model.predict(features)[0])

        # Intervalle de confiance empirique ±15 %
        intervalle_bas  = prix * 0.85
        intervalle_haut = prix * 1.15
        prix_m2         = prix / request.surface_reelle_bati
        latence_ms      = (time.perf_counter() - start) * 1000

        PREDICTIONS_TOTAL.inc()
        PREDICTION_LATENCY.observe(latence_ms / 1000)
        PREDICTION_PRICE.observe(prix)

        logger.info({
            "event": "prediction",
            "prix_estime": prix,
            "type_local": request.type_local,
            "departement": request.code_departement,
            "surface": request.surface_reelle_bati,
            "latence_ms": latence_ms,
        })

        return PredictionResponse(
            prix_estime=round(prix, 2),
            intervalle_bas=round(intervalle_bas, 2),
            intervalle_haut=round(intervalle_haut, 2),
            prix_m2=round(prix_m2, 2),
            modele_version=f"{MODEL_NAME}/{MODEL_STAGE}",
            latence_ms=round(latence_ms, 2),
        )

    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(
    requests: list[PredictionRequest],
    _token: dict = Depends(verify_token),
):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 biens par batch")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    df = pd.DataFrame([r.dict() for r in requests])
    prix = model.predict(df).tolist()
    return {"predictions": [round(p, 2) for p in prix], "count": len(prix)}


@app.get("/model/info", tags=["Modèle"])
def model_info(_token: dict = Depends(verify_token)):
    """Retourne les métadonnées du modèle en production."""
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise HTTPException(status_code=404, detail="Aucun modèle en production")
    v = versions[0]
    run = client.get_run(v.run_id)
    return {
        "name":    MODEL_NAME,
        "version": v.version,
        "stage":   v.current_stage,
        "run_id":  v.run_id,
        "metrics": run.data.metrics,
        "params":  run.data.params,
    }