# Monitoring du modèle – Documentation technique (C11)

Surveillance continue du modèle AI API App E3 avec **Evidently AI** + **Prometheus** + **Grafana**.

---

## Architecture de monitoring

```
Données production (DVF courant)
        │
        ▼
  monitor.py (Evidently AI)
   ├── Rapport drift données  → reports/drift/
   ├── Rapport performance    → reports/performance/
   ├── Suite de tests pass/fail
   └── Métriques → MLflow Tracking
        │
        ▼
  API FastAPI (/metrics)
        │
        ▼
  Prometheus (scraping)
        │
        ▼
  Grafana (dashboards)
```

---

## Installation

### Prérequis
- Python 3.11+
- Accès au serveur MLflow
- Données de référence (dvf_reference.csv) et courant (dvf_current.csv)

### Installation des dépendances

```bash
cd monitoring/
pip install -r requirements.txt
```

### Dépendances

| Package | Version | Rôle |
|---|---|---|
| evidently | 0.4.22 | Détection de drift, rapports HTML |
| mlflow | 2.10.0 | Chargement du modèle, log des métriques |
| scikit-learn | 1.4.1 | Calcul des métriques de régression |
| pandas | 2.2.1 | Manipulation des DataFrames |
| numpy | 1.26.4 | Calculs numériques |

---

## Test dans un environnement bac à sable

**Avant tout déploiement en production, tester la chaîne dans un environnement isolé :**

```bash
# 1. Générer des données de test synthétiques
python ml/download_dvf.py --year 2023 --output data/dvf_sandbox.csv --sample 1000

# 2. Créer les fichiers référence/courant de test
head -500 data/dvf_sandbox.csv > data/dvf_reference_test.csv
tail -500 data/dvf_sandbox.csv > data/dvf_current_test.csv

# 3. Lancer le monitoring en mode bac à sable (env de test dédié)
MLFLOW_TRACKING_URI=http://localhost:5000 \
REPORTS_DIR=reports/sandbox \
python monitoring/monitor.py data/dvf_reference_test.csv data/dvf_current_test.csv

# 4. Vérifier les rapports générés
ls reports/sandbox/drift/
ls reports/sandbox/performance/
```

---

## Exécution en production

```bash
# Lancement manuel
python monitoring/monitor.py data/dvf_reference.csv data/dvf_current.csv

# Via Docker
docker build -t ai-api-app-monitoring .
docker run \
  -v $(pwd)/data:/data \
  -v $(pwd)/reports:/reports \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  ai-api-app-monitoring
```

---

## Métriques surveillées et seuils d'alerte

| Catégorie | Métrique | Seuil d'alerte | Justification |
|---|---|---|---|
| **Data drift** | Part de colonnes driftées | > 30 % | Changement structurel du marché |
| **Data drift** | Score drift `surface_reelle_bati` | p-value < 0.05 | Feature la plus prédictive |
| **Data drift** | Score drift `valeur_fonciere` | p-value < 0.05 | Évolution du marché cible |
| **Performance ML** | RMSE production | +20 % vs référence | Dégradation significative |
| **Performance ML** | R² production | < 0.75 | Seuil de qualité minimum |
| **Système** | Latence p95 API | > 500 ms | SLA temps réel |
| **Système** | Taux d'erreurs | > 5 % | Instabilité applicative |

**Justification du choix d'Evidently AI :**
Evidently a été retenu car il génère des rapports HTML lisibles par tous les profils (technique et métier), s'intègre nativement avec MLflow, et propose une suite de tests automatisés pré-construits adaptés à la régression. Son format de rapport respecte les recommandations d'accessibilité WCAG AA (contraste, structure sémantique, navigation clavier).

---

## Rapports générés

### Rapport de drift (HTML)
- Localisation : `reports/drift/drift_report_YYYYMMDD_HHMMSS.html`
- Contenu : distribution de chaque feature, test de Wasserstein, p-values
- Ouvrir dans n'importe quel navigateur

### Rapport de performance (HTML)
- Localisation : `reports/performance/performance_report_YYYYMMDD_HHMMSS.html`
- Contenu : MAE, RMSE, R², distribution des erreurs, graphiques résidus

### Résumés JSON (pour intégration CI/CD)
```json
{
  "timestamp": "20240115_083000",
  "dataset_drift": false,
  "drift_share": 0.14,
  "n_drifted_cols": 1,
  "report_path": "reports/drift/drift_report_20240115_083000.html"
}
```

---

## Alertes

Lorsqu'un drift est détecté, une entrée est ajoutée dans `reports/alerts.log` :

```
⚠️ DRIFT DÉTECTÉ – 3 colonnes driftées (42.9% du dataset) à 20240115_090000
```

**Intégration possible avec :** Slack webhook, PagerDuty, e-mail via SMTP, Azure Monitor Alerts.

---

## Dashboard Grafana

Accès : `http://localhost:3000` (admin / voir variable `GRAFANA_PASSWORD`)

**Panels disponibles :**
- Volume de prédictions par heure
- Latence p50 / p95 / p99
- Taux d'erreurs
- Distribution des prix prédits

**Accessibilité du dashboard :**
Les dashboards Grafana utilisent des palettes de couleurs daltonisme-friendly (ColorBrewer) et incluent des annotations textuelles sur chaque graphique pour les lecteurs d'écran.

---

## Fréquence d'exécution

| Déclencheur | Action |
|---|---|
| Lundi 8h (cron) | Monitoring hebdomadaire complet via GitHub Actions |
| Merge en main | Monitoring post-déploiement (smoke test) |
| Manuel | `python monitoring/monitor.py` |