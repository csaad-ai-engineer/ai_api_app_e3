<!--
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: README.md
COMPÉTENCES: C9, C10, C11, C12, C13
═══════════════════════════════════════════════════════════════════════════════
-->

# 🏠 AI API App E3 - Prédiction Prix Immobilier

> **Projet certification RNCP37827** - Développeur en Intelligence Artificielle  
> **Évaluation E3** - Validation compétences C9 à C13 (Bloc 2)

## 🎯 Objectif

Système MLOps complet de prédiction de prix immobiliers basé sur les données DVF (Open Data France). 
Expose un modèle GradientBoosting via API REST FastAPI sécurisée, avec interface Streamlit, 
monitoring continu et pipeline CI/CD automatisé.

**Performance:** R² = 0.83 | MAE = 18 200 € | Latence API = 42 ms

## 🚀 Installation rapide
```bash
# Lancer tous les services
docker-compose up -d

# Accéder aux interfaces
# - API:        http://localhost:8000/docs
# - Streamlit:  http://localhost:8501
# - MLflow:     http://localhost:5000
# - Grafana:    http://localhost:3000
```

## 📚 Compétences validées

| Compétence | Fichiers principaux |
|-----------|-------------------|
| **C9** - API REST | `api/main.py`, `tests/test_api.py` |
| **C10** - Intégration | `streamlit_app/app.py`, `tests/test_integration.py` |
| **C11** - Monitoring | `monitoring/monitor.py` |
| **C12** - Tests | `tests/test_*.py` |
| **C13** - CI/CD | `.github/workflows/ci-cd.yml` |

## 🧪 Tests
```bash
pytest tests/ -v --cov=api --cov=ml --cov-report=html
```

**Couverture: 87%** (API: 91%, ML: 83%)

## 📁 Structure
```
ai_api_app_e3/
├── api/              # C9 - API REST FastAPI
├── ml/               # Pipeline ML
├── streamlit_app/    # C10 - Interface GUI
├── monitoring/       # C11 - Surveillance
├── tests/            # C12 - Tests automatisés
├── .github/          # C13 - CI/CD
└── infrastructure/   # IaC
```
# API AI API App E3 – Documentation technique

API REST FastAPI exposant le modèle de prédiction de prix immobilier (C9).

---

## Installation et lancement

### Prérequis
- Python 3.11+
- Docker (optionnel)
- MLflow Tracking Server accessible

### Installation locale

```bash
cd api/
pip install -r requirements.txt
```

### Variables d'environnement

| Variable | Description | Défaut |
|---|---|---|
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow | `http://mlflow:5000` |
| `MODEL_NAME` | Nom du modèle dans le registry | `ai-api-app-model` |
| `MODEL_STAGE` | Stage MLflow à charger | `Production` |
| `JWT_SECRET` | Clé secrète pour signer les JWT | *(obligatoire en prod)* |

### Lancement

```bash
# Développement (avec rechargement automatique)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

# Docker
docker build -t ai-api-app-api .
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e JWT_SECRET=your-secret \
  ai-api-app-api
```

---

## Authentification (JWT Bearer)

L'API utilise des tokens JWT signés (HS256). Tous les endpoints sauf `/health` et `/metrics` requièrent un token valide.

**Génération d'un token (Python) :**
```python
import jwt, datetime

token = jwt.encode(
    {"sub": "mon-app", "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
    "your-jwt-secret",
    algorithm="HS256"
)
```

**Utilisation dans les requêtes :**
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/predict
```

---

## Endpoints

### GET /health
Vérifie l'état de l'API et du modèle chargé.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "ai-api-app-model",
  "model_stage": "Production",
  "api_version": "1.0.0"
}
```

---

### POST /predict
Prédit le prix d'un bien immobilier.

**Corps de la requête :**
```json
{
  "surface_reelle_bati": 75.0,
  "nombre_pieces_principales": 3,
  "surface_terrain": 0.0,
  "longitude": 2.347,
  "latitude": 48.859,
  "type_local": "Appartement",
  "code_departement": "75"
}
```

**Réponse :**
```json
{
  "prix_estime": 425000.0,
  "intervalle_bas": 361250.0,
  "intervalle_haut": 488750.0,
  "prix_m2": 5666.67,
  "modele_version": "ai-api-app-model/Production",
  "latence_ms": 42.3
}
```

**Contraintes de validation :**
- `surface_reelle_bati` : entre 5 et 2 000 m²
- `nombre_pieces_principales` : entre 1 et 20
- `surface_terrain` : entre 0 et 50 000 m²
- `longitude` : entre -5 et 10 (France métropolitaine)
- `latitude` : entre 41 et 52 (France métropolitaine)
- `type_local` : `Maison`, `Appartement` ou `Dépendance`
- `code_departement` : 2 ou 3 caractères (ex: `75`, `2A`)

---

### POST /predict/batch
Prédit les prix d'une liste de biens (maximum 100).

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '[{"surface_reelle_bati": 65, ...}, {"surface_reelle_bati": 120, ...}]'
```

---

### GET /model/info
Retourne les métadonnées du modèle en production (version, métriques, paramètres).

---

### GET /metrics
Endpoint Prometheus pour le scraping des métriques système.

**Métriques exposées :**
- `predictions_total` : compteur de prédictions effectuées
- `prediction_errors_total` : compteur d'erreurs
- `prediction_latency_seconds` : histogramme des latences
- `prediction_price_euros` : histogramme des prix prédits

---

## Documentation OpenAPI interactive

Disponible à l'adresse `http://localhost:8000/docs` (Swagger UI) ou `http://localhost:8000/redoc`.

---

## Sécurité (OWASP Top 10)

| Risque OWASP | Mesure implémentée |
|---|---|
| A01 – Broken Access Control | JWT obligatoire sur tous les endpoints sensibles |
| A03 – Injection | Validation stricte via Pydantic (types + bornes) |
| A04 – Insecure Design | Intervalles de valeurs définis pour chaque paramètre |
| A05 – Security Misconfiguration | CORS explicite, pas de valeurs par défaut en prod |
| A09 – Logging | Logging structuré de toutes les prédictions |

---

## Accessibilité de la documentation (WCAG/Valentin Haüy)

La documentation de l'API est publiée en format HTML (Swagger/ReDoc) avec :
- Textes alternatifs sur les schémas
- Structure sémantique (h1, h2, h3)
- Contraste suffisant (fond blanc, texte sombre)
- Navigation possible au clavier

---

## Exécution des tests

```bash
# Depuis la racine du projet
pip install pytest pytest-cov

# Tests unitaires API uniquement
pytest tests/test_api.py -v --cov=api --cov-report=term-missing

# Tests d'intégration
pytest tests/test_integration.py -v -m integration

# Tous les tests avec rapport de couverture HTML
pytest tests/ -v --cov=api --cov=ml --cov-report=html:reports/coverage
```

