# Tests automatisés – Documentation technique (C12)

Suite de tests du projet AI API App E3 couvrant les 5 étapes du cycle ML.

---

## Organisation des tests

```
tests/
├── conftest.py              # Fixtures partagées (données, modèle, token JWT, client API)
├── test_data.py             # C12 – Validation du jeu de données DVF
├── test_ml.py               # C12 – Pipeline ML (build, fit, predict, métriques)
├── test_model_pipeline.py   # C12 – Pipeline complet : 5 étapes de bout en bout
├── test_api.py              # C9  – Endpoints API, authentification, validation
└── test_integration.py      # C10 – Intégration Streamlit ↔ API (tous endpoints)
```

---

## Installation de l'environnement de test

### Prérequis
- Python 3.11+
- Depuis la racine du projet

```bash
pip install -r requirements.txt
pip install pytest pytest-cov pytest-html pytest-asyncio ruff
```

### Dépendances de test

| Package | Version | Rôle |
|---|---|---|
| pytest | 8.1.1 | Framework de test |
| pytest-cov | 5.0.0 | Calcul de la couverture |
| pytest-html | 4.1.1 | Rapport HTML de test |
| pytest-asyncio | 0.23.6 | Support des coroutines async |
| httpx | 0.27.0 | Client HTTP pour les tests d'intégration |
| ruff | latest | Linting du code |

---

## Exécution des tests

### Tous les tests

```bash
pytest tests/ -v
```

### Par catégorie

```bash
# Validation du jeu de données uniquement (avant entraînement)
pytest tests/test_data.py -v

# Pipeline ML complet (5 étapes C12)
pytest tests/test_model_pipeline.py -v

# Tests unitaires ML
pytest tests/test_ml.py -v

# Tests unitaires API
pytest tests/test_api.py -v

# Tests d'intégration Streamlit ↔ API
pytest tests/test_integration.py -v -m integration

# Exclure les tests d'intégration (CI rapide)
pytest tests/ -v -m "not integration"
```

### Avec rapport de couverture

```bash
# Couverture console
pytest tests/ --cov=api --cov=ml --cov-report=term-missing

# Couverture HTML (ouvrir reports/coverage/index.html)
pytest tests/ --cov=api --cov=ml --cov-report=html:reports/coverage

# Couverture XML (pour CI/CD Codecov)
pytest tests/ --cov=api --cov=ml --cov-report=xml:reports/coverage.xml
```

### Avec rapport HTML des résultats

```bash
pytest tests/ --html=reports/test_report.html --self-contained-html
```

---

## Couverture cible

| Module | Couverture cible |
|---|---|
| `api/main.py` | ≥ 85 % |
| `ml/train.py` | ≥ 80 % |
| `ml/evaluate_gate.py` | ≥ 75 % |
| **Total** | **≥ 80 %** |

---

## Étapes du pipeline ML testées (C12)

Le critère C12 exige que les tests couvrent **toutes** les étapes :

| Étape | Fichier de test | Classe |
|---|---|---|
| 1. Validation du jeu de données | `test_data.py` + `test_model_pipeline.py` | `TestDVFSchema`, `TestDVFDistributions`, `TestValidationJeuDeDonnees` |
| 2. Préparation des données | `test_model_pipeline.py` | `TestPreparationDonnees` |
| 3. Entraînement | `test_model_pipeline.py` | `TestEntrainement` |
| 4. Évaluation | `test_model_pipeline.py` | `TestEvaluation` |
| 5. Validation / Gate qualité | `test_model_pipeline.py` | `TestValidationModele` |

---

## Cas testés par fichier

### test_data.py
- Toutes les colonnes requises présentes
- Aucun doublon
- Taux de valeurs nulles < 5 %
- Prix entre 10 000 et 5 000 000 €
- Surface entre 5 et 2 500 m²
- Coordonnées dans la France métropolitaine
- Types de bien valides
- Prix/m² cohérent (200 – 30 000 €)

### test_model_pipeline.py
- Chargement sans erreur
- Filtres de nettoyage appliqués (outliers, prix négatifs)
- Normalisation numérique (StandardScaler ajusté)
- Encodage catégoriel (OrdinalEncoder ajusté)
- Gestion des catégories inconnues
- Absence de NaN après preprocessing
- Pipeline fit sans exception
- Reproductibilité (random_state=42)
- Absence d'overfitting sévère (écart train/test < 0.40)
- Toutes les métriques retournées (mae, rmse, r2, mape)
- Prédictions dans une plage réaliste
- Gate R² > seuil minimal

### test_api.py
- Endpoint /health retourne 200 avec tous les champs
- JWT valide donne accès, expiré/invalide retourne 401
- /predict retourne prix positif, intervalle cohérent, prix/m² correct
- /predict/batch max 100 éléments
- Validation Pydantic (surface trop petite, type invalide, latitude hors bornes)
- /metrics retourne format Prometheus

### test_integration.py
- Cycle complet Streamlit (health → predict → batch)
- Tous les endpoints exploités couverts
- Renouvellement de token (expiré → 401)
- Résultats interprétés correctement (intervalle logique, ordre batch préservé)

---

## Interprétation des résultats

Un run pytest réussi ressemble à :

```
tests/test_data.py::TestDVFSchema::test_all_required_columns_present PASSED
tests/test_data.py::TestDVFDistributions::test_prix_minimum_realiste PASSED
...
tests/test_model_pipeline.py::TestEntrainement::test_pipeline_fit_sans_erreur PASSED
...
---------- coverage: api=91%, ml=83% ----------
87 passed in 12.4s
```

**Si un test échoue :**
- `FAILED test_data.py::TestDVFDistributions::test_prix_m2_coherent` → vérifier le nettoyage DVF
- `FAILED test_model_pipeline.py::TestValidationModele::test_gate_r2_minimum` → modèle insuffisant, revoir les hyperparamètres
- `FAILED test_api.py::TestAuthentication::test_predict_with_valid_token_returns_200` → vérifier JWT_SECRET

---

## Linting

```bash
# Vérification du style (erreurs critiques seulement)
ruff check . --select E,W,F --ignore E501

# Correction automatique
ruff check . --fix
```