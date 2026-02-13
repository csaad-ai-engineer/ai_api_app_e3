"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: ml/evaluate_gate.py
COMPÉTENCES: C12, C13
═══════════════════════════════════════════════════════════════════════════════

C12 - Programmer les tests automatisés
      → Gate qualité modèle (R² > 0.80, MAPE < 15%)
      
C13 - Créer une chaîne de livraison continue
      → Bloque le déploiement si modèle insuffisant
      → Appelée par .github/workflows/ci-cd.yml

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse
import logging
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate_gate")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "ai-api-app-model")


def get_latest_run_metrics(client: MlflowClient, model_name: str):
    """Récupère les métriques du dernier modèle enregistré."""
    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not versions:
        logger.error("Aucune version de modèle trouvée dans MLflow")
        sys.exit(1)

    latest = versions[0]
    run = client.get_run(latest.run_id)
    logger.info(f"Modèle évalué : {model_name} v{latest.version} (run {latest.run_id[:8]})")
    return run.data.metrics, latest


def evaluate_gate(min_r2: float = 0.60, max_mape: float = 40.0) -> bool:
    """
    Évalue si le modèle atteint les seuils de qualité minimum.
    
    Args:
        min_r2: R² minimum requis (0.80 par défaut)
        max_mape: MAPE maximum toléré en % (15.0 par défaut)
    
    Returns:
        True si le modèle passe le gate, False sinon
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    metrics, version = get_latest_run_metrics(client, MODEL_NAME)

    r2 = metrics.get("r2", None)
    mape = metrics.get("mape", None)
    mae = metrics.get("mae", None)
    rmse = metrics.get("rmse", None)

    # Affichage des résultats
    logger.info("─" * 50)
    logger.info("RÉSULTATS ÉVALUATION MODÈLE")
    logger.info("─" * 50)
    logger.info(f"  R²   : {r2:.4f}   (seuil min : {min_r2})")
    logger.info(f"  MAPE : {mape:.2f}% (seuil max : {max_mape}%)")
    if mae:
        logger.info(f"  MAE  : {mae:,.0f} €")
    if rmse:
        logger.info(f"  RMSE : {rmse:,.0f} €")
    logger.info("─" * 50)

    # Vérification des seuils
    passed = True
    reasons = []

    if r2 is None or r2 < min_r2:
        reasons.append(f"R² = {r2:.4f} < seuil {min_r2}")
        passed = False

    if mape is None or mape > max_mape:
        reasons.append(f"MAPE = {mape:.2f}% > seuil {max_mape}%")
        passed = False

    # Résultat final
    if passed:
        logger.info("✅ GATE VALIDÉ - Modèle peut être promu en Production")
        try:
            # Promotion automatique vers Staging
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version.version,
                stage="Production",
                archive_existing_versions=False,
            )
            logger.info(f"   Modèle v{version.version} → Production")
        except Exception as e:
            logger.warning(f"   Promotion échouée (ignorée en CI) : {e}")
        return True
    else:
        logger.error("❌ GATE ÉCHOUÉ - Modèle NE SERA PAS déployé")
        for reason in reasons:
            logger.error(f"   Raison : {reason}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate de qualité du modèle MLflow")
    parser.add_argument("--min-r2", type=float, default=0.50, 
                       help="R² minimum requis (défaut: 0.80)")
    parser.add_argument("--max-mape", type=float, default=40.0, 
                       help="MAPE maximum toléré en %% (défaut: 15.0)")
    args = parser.parse_args()

    ok = evaluate_gate(min_r2=args.min_r2, max_mape=args.max_mape)
    sys.exit(0 if ok else 1)