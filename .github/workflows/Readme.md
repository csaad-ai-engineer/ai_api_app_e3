# CI/CD – Documentation technique (C13)

Chaîne de livraison continue du projet AI API App E3 via **GitHub Actions** → **Azure Container Apps**.

---

## Vue d'ensemble de la chaîne

```
┌─────────────────────────────────────────────────────────────────┐
│                     GitHub Actions                              │
│                                                                 │
│  Push / PR          Build           Train          Deploy       │
│  ──────────         ───────         ──────         ───────      │
│  ruff lint    ──►   Docker API  ──► DVF DL    ──► Staging      │
│  pytest       │     Docker GUI  │   train.py  │   (develop)    │
│  test_data    │     Docker mon  │   gate R²   │               │
│  test_ml      │     push GHCR   │   MLflow    │   Production   │
│  test_api     │                 │   Registry  ──► (main)       │
│  test_integ   │                 │                               │
│               │                 │   Cron lundi 8h               │
│               │                 └── Evidently monitoring        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Jobs du pipeline

| Job | Déclencheur | Description |
|---|---|---|
| `test` | Tout push, toute PR | Linting + tous les tests pytest |
| `build` | Merge main ou develop | Build Docker (API, Streamlit, monitoring) + push GHCR |
| `train` | `[retrain]` dans le message de commit OU déclenchement manuel | Télécharge DVF, entraîne, enregistre dans MLflow, gate qualité |
| `deploy-staging` | Merge develop | Déploiement Azure Container Apps staging |
| `deploy-production` | Merge main (après staging) | Déploiement Azure Container Apps production |
| `monitoring` | Cron lundi 8h | Rapport Evidently, upload artefacts |

---

## Installation et configuration

### Secrets GitHub à configurer

Dans `Settings > Secrets and variables > Actions` :

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | JSON du service principal Azure (`az ad sp create-for-rbac`) |
| `AZURE_RG` | Nom du resource group Azure |
| `MLFLOW_TRACKING_URI` | URL du serveur MLflow (ex: `https://mlflow.mondomaine.com`) |
| `MLFLOW_USERNAME` | Identifiant MLflow (si authentification activée) |
| `MLFLOW_PASSWORD` | Mot de passe MLflow |
| `JWT_SECRET_STAGING` | Clé JWT pour l'environnement staging |
| `JWT_SECRET_PROD` | Clé JWT pour l'environnement production |
| `API_URL_STAGING` | URL de l'API staging pour les smoke tests |
| `API_URL_PROD` | URL de l'API production |
| `CODECOV_TOKEN` | Token Codecov pour le rapport de couverture |

### Déclencheurs configurés

```yaml
on:
  push:
    branches: [main, develop]        # CI sur chaque push
  pull_request:
    branches: [main]                  # CI sur les PR vers main
  workflow_dispatch:                  # Déclenchement manuel
    inputs:
      retrain:
        description: "Forcer le ré-entraînement"
        type: boolean
  # schedule:
  #   - cron: '0 8 * * 1'            # Monitoring hebdomadaire (décommenter)
```

---

## Étapes détaillées

### 1. Test & Qualité

```bash
# Reproduire localement le job CI
pip install -r requirements.txt ruff pytest pytest-cov
ruff check . --select E,W,F --ignore E501        # Linting
pytest tests/test_ml.py  -v --cov=ml             # Tests ML
pytest tests/test_api.py -v --cov=api            # Tests API
pytest tests/test_integration.py -v -m integration
```

### 2. Build Docker

Les 3 images sont construites et poussées sur GitHub Container Registry (GHCR) :

```
ghcr.io/<org>/ai-api-app-e3/ai-api-app-api:sha-<hash>
ghcr.io/<org>/ai-api-app-e3/ai-api-app-api:latest         (main seulement)
ghcr.io/<org>/ai-api-app-e3/ai-api-app-streamlit:latest
```

Le cache Docker BuildKit (type=gha) réduit les temps de build de ~60 %.

### 3. Entraînement du modèle (MLOps)

Déclenché par le message de commit contenant `[retrain]` :

```bash
git commit -m "feat: ajout feature DPE [retrain]"
git push
```

Étapes dans le CI :
1. `python ml/download_dvf.py --year 2023 --output data/dvf_clean.csv`
2. `pytest tests/test_data.py` — gate qualité des données
3. `python ml/train.py` — entraînement + enregistrement MLflow
4. `python ml/evaluate_gate.py --min-r2 0.80 --max-mape 15` — gate qualité modèle
   - Retourne `exit 1` si les seuils ne sont pas atteints → le déploiement est bloqué

### 4. Packaging du modèle

Le modèle est packagé sous forme d'image Docker intégrant le modèle MLflow :

```dockerfile
# Dans api/Dockerfile
# Le modèle est chargé dynamiquement depuis MLflow au démarrage
# → Pas de modèle embarqué dans l'image, découplage API / modèle
```

Pour embarquer le modèle dans l'image (alternative) :
```bash
mlflow models build-docker -m "models:/ai-api-app-model/Production" -n immo-model-standalone
```

### 5. Déploiement Azure

```bash
# Reproduire localement le déploiement
az login
az containerapp update \
  --name ai-api-app-api-prod \
  --resource-group $AZURE_RG \
  --image ghcr.io/<org>/ai-api-app-e3/ai-api-app-api:latest

# Vérifier le déploiement
curl https://ai-api-app-api-prod.<region>.azurecontainerapps.io/health
```

---

## Infrastructure Azure (Bicep)

```bash
# Créer le service principal
az ad sp create-for-rbac \
  --name "ai-api-app-e3-cicd" \
  --role contributor \
  --scopes /subscriptions/<sub-id>/resourceGroups/<rg>
  # Copier le JSON dans le secret AZURE_CREDENTIALS

# Déployer l'infrastructure (première fois)
az deployment group create \
  --resource-group ai-api-app-e3-rg \
  --template-file infrastructure/azure-main.bicep \
  --parameters acrLoginServer=ghcr.io/<org> jwtSecret=<secret> mlflowTrackingUri=<uri>
```

---

## Smoke tests post-déploiement

```bash
# Vérification automatique après chaque déploiement
sleep 30  # Attendre le démarrage du container
curl -f https://<api-url>/health || exit 1
```

---

## Rollback

En cas d'erreur en production :

```bash
# Revenir à la version précédente de l'image
az containerapp update \
  --name ai-api-app-api-prod \
  --resource-group $AZURE_RG \
  --image ghcr.io/<org>/ai-api-app-e3/ai-api-app-api:sha-<previous-hash>
```

---

## Accessibilité de la documentation (Valentin Haüy / AcceDe)

Ce README est rédigé en Markdown avec :
- Structure hiérarchique (h1, h2, h3)
- Tableaux avec en-têtes explicites
- Blocs de code avec langage précisé (syntaxe)
- Pas d'information transmise uniquement par la couleur