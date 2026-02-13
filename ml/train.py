"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: ml/train.py
COMPÉTENCES: C9, C11, C12, C13
═══════════════════════════════════════════════════════════════════════════════

C9  - Développer une API exposant un modèle d'IA
      → Prépare le modèle qui sera exposé via api/main.py
      
C11 - Monitorer un modèle d'IA
      → Log métriques dans MLflow pour suivi performance
      
C12 - Programmer les tests automatisés
      → Testé par test_model_pipeline.py (5 étapes)
      
C13 - Créer une chaîne de livraison continue
      → Enregistrement MLflow Registry pour CI/CD automatisé

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import logging
import pandas as pd
import numpy as np
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME     = "ai-api-app-e3"
MODEL_NAME          = "ai-api-app-model"
DATA_PATH           = os.getenv("DATA_PATH", "data/dvf_clean.csv")

NUMERIC_FEATURES = [
    "surface_reelle_bati", "nombre_pieces_principales",
    "surface_terrain", "longitude", "latitude",
    # # Nouvelles features
    # "est_paris", "est_idf", "surface_totale", "a_terrain", 
    # "pieces_par_m2", "surface_x_pieces"
]
CATEGORIAL_FEATURES = ["type_local", "code_departement"]
TARGET              = "valeur_fonciere"


def load_data(path: str) -> pd.DataFrame:
    """Charge et nettoie les données DVF avec filtres avancés."""
    logger.info(f"Chargement des données depuis {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"code_departement": str})
    
    # Colonnes de base seulement
    base_cols = ["surface_reelle_bati", "nombre_pieces_principales",
                 "surface_terrain", "longitude", "latitude", 
                 "type_local", "code_departement", "valeur_fonciere"]
    
    # Suppression valeurs manquantes (colonnes de base uniquement)
    df = df.dropna(subset=base_cols)
    
    # Filtres métier stricts
    df = df[df["valeur_fonciere"].between(20_000, 2_000_000)]
    df = df[df["surface_reelle_bati"].between(10, 500)]
    df = df[df["nombre_pieces_principales"].between(1, 10)]
    
    # Calcul prix/m² et filtre
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = df[df["prix_m2"].between(500, 20_000)]
    
    # Suppression outliers statistiques
    Q1 = df["valeur_fonciere"].quantile(0.25)
    Q3 = df["valeur_fonciere"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df["valeur_fonciere"].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
    
    df = df.drop(columns=["prix_m2"])
    
    logger.info(f"Dataset final : {len(df):,} lignes après nettoyage")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features calculées pour améliorer le modèle."""
    # Features géographiques
    df["est_paris"] = (df["code_departement"] == "75").astype(int)
    df["est_idf"] = df["code_departement"].isin(["75", "92", "93", "94", "78", "91", "95", "77"]).astype(int)
    
    # Features surface
    df["surface_totale"] = df["surface_reelle_bati"] + df["surface_terrain"]
    df["a_terrain"] = (df["surface_terrain"] > 0).astype(int)
    df["pieces_par_m2"] = df["nombre_pieces_principales"] / df["surface_reelle_bati"]
    
    # Features prix estimé (proxy)
    df["surface_x_pieces"] = df["surface_reelle_bati"] * df["nombre_pieces_principales"]
    
    return df


def build_pipeline() -> Pipeline:
    """Construit le pipeline scikit-learn."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1), CATEGORIAL_FEATURES),
    ])
    # model = GradientBoostingRegressor(
    #     n_estimators=500,           # Plus d'arbres (300→500)
    #     max_depth=7,                # Arbres plus profonds (5→7)
    #     learning_rate=0.05,
    #     min_samples_split=20,       # Plus conservateur (10→20)
    #     min_samples_leaf=5,         # Ajout : évite overfitting
    #     subsample=0.8,              # Ajout : bagging
    #     random_state=42,
    # )
    model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true, y_pred) -> dict:
    """Calcule les métriques d'évaluation."""
    mae  = mean_absolute_error(y_true, y_pred) # MAE (Mean Absolute Error)
    rmse = root_mean_squared_error(y_true, y_pred) # RMSE (Root Mean Squared Error)
    r2   = r2_score(y_true, y_pred) # R² (coefficient de détermination)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 # MAPE (Mean Absolute Percentage Error)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train():
    """Pipeline d'entraînement complet."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    df = load_data(DATA_PATH)
    # df = add_features(df)  
    X = df[NUMERIC_FEATURES + CATEGORIAL_FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run() as run:
        # Entraînement
        pipeline = build_pipeline()
        logger.info("Cross-validation...")
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=5, scoring="r2", n_jobs=-1)
        logger.info(f"CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        logger.info("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline fitted ✅")
        
        # Évaluation
        y_pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        logger.info(f"Métriques test : {metrics}")
        
        # Log MLflow
        mlflow.log_params(pipeline.named_steps["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_r2_mean", cv_scores.mean())
        
        # Enregistrement du modèle
        signature = infer_signature(X_train, pipeline.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
        )
        
        logger.info(f"✅ Modèle loggé: {model_info.model_uri}")
        
        # Promotion avec alias si bon R²
        if metrics["r2"] > 0.60:
            try:
                client = mlflow.MlflowClient()
                
                # Récupère la dernière version du modèle
                model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                latest_version = max([int(v.version) for v in model_versions])
                
                # Définit l'alias 'champion'
                client.set_registered_model_alias(
                    MODEL_NAME, 
                    "champion", 
                    str(latest_version)
                )
                logger.info(f"✅ Modèle v{latest_version} → alias 'champion' (R²={metrics['r2']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Erreur alias : {e}")
        else:
            logger.warning(f"⚠️ R²={metrics['r2']:.3f} < 0.60 → Pas de promotion")
    
    return metrics

if __name__ == "__main__":
    train()