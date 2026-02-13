"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: monitoring/monitor.py
COMPÉTENCES: C11
═══════════════════════════════════════════════════════════════════════════════

Evidently AI - Détection drift + rapports

═══════════════════════════════════════════════════════════════════════════════
"""

"""
Monitoring du modèle avec Evidently AI.
Détecte le data drift, le target drift et la dégradation de performance.
Génère des rapports HTML et des métriques Prometheus.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric,
    DatasetSummaryMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnValueMean,
    TestValueRange,
    # TestNumberOfOutliers removed - doesn't exist in this version
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitoring")

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "ai-api-app-model")
REPORTS_DIR         = Path(os.getenv("REPORTS_DIR", "reports"))
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/dvf_clean_reference.csv")

NUMERIC_FEATURES  = ["surface_reelle_bati", "nombre_pieces_principales",
                     "surface_terrain", "longitude", "latitude"]
CATEGORICAL_FEATURES = ["type_local", "code_departement"]
TARGET            = "valeur_fonciere"
ALL_FEATURES      = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_reference_data(path: str) -> pd.DataFrame:
    """Charge les données de référence (jeu d'entraînement ou période de référence)."""
    df = pd.read_csv(path, low_memory=False)
    return df[ALL_FEATURES + [TARGET]].dropna()


def load_current_data(path: str) -> pd.DataFrame:
    """Charge les données de production récentes (dernières 24h/7j)."""
    df = pd.read_csv(path, low_memory=False)
    return df[ALL_FEATURES + [TARGET]].dropna()


def compute_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ajoute les prédictions au DataFrame pour l'analyse de performance."""
    df = df.copy()
    df["prediction"] = model.predict(df[ALL_FEATURES])
    return df


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                           output_dir: Path) -> dict:
    """Génère un rapport de data drift Evidently HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = NUMERIC_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    # ── Rapport de drift des données ──────────────────────────────────────────
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DatasetSummaryMetric(),
        *[ColumnDriftMetric(column_name=col) for col in NUMERIC_FEATURES],
    ])
    drift_report.run(
        reference_data=reference[ALL_FEATURES],
        current_data=current[ALL_FEATURES],
        column_mapping=column_mapping
    )

    report_path = output_dir / f"drift_report_{timestamp}.html"
    drift_report.save_html(str(report_path))
    logger.info(f"Rapport drift : {report_path}")

    # Extraction des résultats JSON
    results = drift_report.as_dict()
    
    # Safely extract metrics with error handling
    try:
        # Find the DatasetDriftMetric index
        drift_metric_index = None
        for i, metric in enumerate(results["metrics"]):
            if metric.get("metric", "") == "DatasetDriftMetric":
                drift_metric_index = i
                break
        
        if drift_metric_index is not None:
            drift_share = results["metrics"][drift_metric_index]["result"]["share_of_drifted_columns"]
            dataset_drift = results["metrics"][drift_metric_index]["result"]["dataset_drift"]
            n_drifted_cols = results["metrics"][drift_metric_index]["result"]["number_of_drifted_columns"]
        else:
            drift_share = 0.0
            dataset_drift = False
            n_drifted_cols = 0
    except (KeyError, IndexError, TypeError):
        # Fallback values if structure is different
        drift_share = 0.0
        dataset_drift = False
        n_drifted_cols = 0
        logger.warning("Could not extract drift metrics from report")

    summary = {
        "timestamp":       timestamp,
        "dataset_drift":   dataset_drift,
        "drift_share":     drift_share,
        "n_drifted_cols":  n_drifted_cols,
        "report_path":     str(report_path),
    }

    # Sauvegarde du résumé JSON
    with open(output_dir / f"drift_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_performance_report(reference: pd.DataFrame, current: pd.DataFrame,
                                 output_dir: Path) -> dict:
    """Génère un rapport de performance du modèle (régression)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create ColumnMapping object
    column_mapping = ColumnMapping()
    column_mapping.target = TARGET
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = NUMERIC_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES

    perf_report = Report(metrics=[RegressionPreset()])
    perf_report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    report_path = output_dir / f"performance_report_{timestamp}.html"
    perf_report.save_html(str(report_path))

    # Safely extract metrics
    results = perf_report.as_dict()
    
    try:
        # Find regression metrics
        metrics_data = {}
        for metric in results["metrics"]:
            if "result" in metric and isinstance(metric["result"], dict):
                if "current" in metric["result"] and "reference" in metric["result"]:
                    metrics_data = metric["result"]
                    break
        
        summary = {
            "timestamp":    timestamp,
            "rmse_current": metrics_data.get("current", {}).get("rmse", None),
            "mae_current":  metrics_data.get("current", {}).get("mean_abs_error", None),
            "r2_current":   metrics_data.get("current", {}).get("r2_score", None),
            "rmse_ref":     metrics_data.get("reference", {}).get("rmse", None),
            "report_path":  str(report_path),
        }
    except (KeyError, IndexError, AttributeError):
        summary = {
            "timestamp":    timestamp,
            "rmse_current": None,
            "mae_current":  None,
            "r2_current":   None,
            "rmse_ref":     None,
            "report_path":  str(report_path),
        }
        logger.warning("Could not extract performance metrics from report")

    with open(output_dir / f"perf_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_tests(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Lance la suite de tests Evidently (pass/fail)."""
    
    tests = [
        TestNumberOfDriftedColumns(lt=4),
        TestShareOfDriftedColumns(lt=0.3),
    ]
    
    # Add mean tests for numeric columns with reasonable ranges
    for col in NUMERIC_FEATURES[:3]:  # Test key numeric columns
        if col in reference.columns:
            # Calculate mean and std from reference
            ref_mean = reference[col].mean()
            ref_std = reference[col].std()
            
            # Use TestColumnValueMean
            tests.append(
                TestColumnValueMean(
                    column_name=col,
                    gt=float(ref_mean - 3*ref_std),  # greater than mean - 3 sigma
                    lt=float(ref_mean + 3*ref_std),   # less than mean + 3 sigma
                )
            )
    
    # Add range test for target
    if TARGET in reference.columns:
        ref_min = float(reference[TARGET].min())
        ref_max = float(reference[TARGET].max())
        tests.append(
            TestValueRange(
                column_name=TARGET,
                left=ref_min,
                right=ref_max
            )
        )
    
    # Create column mapping for tests
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = NUMERIC_FEATURES
    column_mapping.categorical_features = CATEGORICAL_FEATURES
    column_mapping.target = TARGET
    
    suite = TestSuite(tests=tests)
    suite.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )
    results = suite.as_dict()

    passed = sum(1 for t in results["tests"] if t["status"] == "SUCCESS")
    failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
    logger.info(f"Tests Evidently : {passed} réussis, {failed} échoués")

    return {
        "total":  len(results["tests"]),
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "details": results["tests"],
    }


def alert_if_drift(summary: dict):
    """Déclenche une alerte si le drift dépasse le seuil critique."""
    if summary["dataset_drift"]:
        msg = (
            f"⚠️ DRIFT DÉTECTÉ – {summary['n_drifted_cols']} colonnes driftées "
            f"({summary['drift_share']:.1%} du dataset) à {summary['timestamp']}"
        )
        logger.warning(msg)
        # Dans un vrai système : envoyer un webhook Slack/Teams, créer une alerte PagerDuty, etc.
        with open(REPORTS_DIR / "alerts.log", "a") as f:
            f.write(msg + "\n")
        return True
    return False


def run_monitoring_pipeline(reference_path: str, current_path: str):
    """Pipeline complet de monitoring."""
    logger.info("=== Démarrage du pipeline de monitoring ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement du modèle
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Modèle chargé : {model_uri}")

    # Chargement des données
    reference = load_reference_data(reference_path)
    current   = load_current_data(current_path)
    logger.info(f"Référence : {len(reference)} lignes | Courant : {len(current)} lignes")

    # Ajout des prédictions
    reference_with_preds = compute_predictions(reference, model)
    current_with_preds   = compute_predictions(current, model)

    # Génération des rapports
    drift_summary = generate_drift_report(reference, current, REPORTS_DIR / "drift")
    perf_summary  = generate_performance_report(
        reference_with_preds, current_with_preds, REPORTS_DIR / "performance"
    )
    test_results  = run_tests(reference, current)

    # Alertes
    has_drift = alert_if_drift(drift_summary)

    # Log dans MLflow
    with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d')}"):
        # Filter numeric metrics for MLflow
        mlflow_metrics = {
            "drift_share":     float(drift_summary["drift_share"]),
            "n_drifted_cols":  int(drift_summary["n_drifted_cols"]),
            "tests_passed":    int(test_results["passed"]),
            "tests_failed":    int(test_results["failed"]),
        }
        
        # Add performance metrics if they exist and are numeric
        for k, v in perf_summary.items():
            if isinstance(v, (int, float)) and v is not None and k != "timestamp":
                mlflow_metrics[k] = float(v)
        
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(drift_summary["report_path"])
        mlflow.log_artifact(perf_summary["report_path"])

    logger.info("=== Monitoring terminé ===")
    return {
        "drift":       drift_summary,
        "performance": perf_summary,
        "tests":       test_results,
        "has_drift":   has_drift,
    }


if __name__ == "__main__":
    import sys
    ref_path = sys.argv[1] if len(sys.argv) > 1 else REFERENCE_DATA_PATH
    cur_path = sys.argv[2] if len(sys.argv) > 2 else "data/dvf_clean.csv"
    result   = run_monitoring_pipeline(ref_path, cur_path)
    print(json.dumps(result, indent=2, default=str))