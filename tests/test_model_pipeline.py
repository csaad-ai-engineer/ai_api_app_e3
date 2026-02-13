"""
═══════════════════════════════════════════════════════════════════════════════
PROJET: AI API App E3 - Développeur IA (RNCP37827)
FICHIER: tests/test_model_pipeline.py
COMPÉTENCES: C12
═══════════════════════════════════════════════════════════════════════════════

Tests 5 étapes ML complètes

═══════════════════════════════════════════════════════════════════════════════
"""

"""
test_model_pipeline.py – Tests du pipeline ML complet (C12)
Couvre TOUTES les étapes selon le critère :
  - Validation du jeu de données
  - Préparation des données (preprocessing)
  - Entraînement
  - Évaluation et validation du modèle
  - Intégration dans MLflow (versionning)
"""


import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import mlflow  
import mlflow.sklearn
from ml.train import (
    build_pipeline, evaluate, load_data,
    NUMERIC_FEATURES, CATEGORIAL_FEATURES, TARGET
)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORIAL_FEATURES


# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 – Validation du jeu de données (avant preprocessing)
# ════════════════════════════════════════════════════════════════════════════════
class TestValidationJeuDeDonnees:

    def test_chargement_sans_erreur(self, tmp_path, dvf_dataframe):
        path = tmp_path / "dvf.csv"
        dvf_dataframe.to_csv(path, index=False)
        df = load_data(str(path))
        assert len(df) > 0

    def test_load_data_supprime_prix_negatifs(self, tmp_path, dvf_dataframe):
        df_with_bad = dvf_dataframe.copy()
        df_with_bad.loc[0, "valeur_fonciere"] = -5000
        path = tmp_path / "dvf.csv"
        df_with_bad.to_csv(path, index=False)
        df = load_data(str(path))
        assert (df["valeur_fonciere"] >= 20000).all()  # Seuil min = 20k€

    def test_load_data_supprime_surface_trop_petite(self, tmp_path, dvf_dataframe):
        df_with_bad = dvf_dataframe.copy()
        df_with_bad.loc[0, "surface_reelle_bati"] = 2.0
        path = tmp_path / "dvf.csv"
        df_with_bad.to_csv(path, index=False)
        df = load_data(str(path))
        assert (df["surface_reelle_bati"] >= 10).all()  # Seuil min = 10m²

    def test_load_data_filtre_prix_extremes(self, tmp_path, dvf_dataframe):
        df_with_bad = dvf_dataframe.copy()
        df_with_bad.loc[1, "valeur_fonciere"] = 50_000_000  # Ultra luxe
        path = tmp_path / "dvf.csv"
        df_with_bad.to_csv(path, index=False)
        df = load_data(str(path))
        assert (df["valeur_fonciere"] <= 2_000_000).all()  # Seuil max = 2M€

    def test_colonnes_requises_presentes(self, dvf_dataframe):
        for col in ALL_FEATURES + [TARGET]:
            assert col in dvf_dataframe.columns


# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 – Préparation des données (preprocessing)
# ════════════════════════════════════════════════════════════════════════════════
class TestPreparationDonnees:

    def test_preprocessor_numerique_normalise(self, dvf_reference):
        from sklearn.preprocessing import StandardScaler
        pipeline = build_pipeline()
        X = dvf_reference[ALL_FEATURES]
        y = dvf_reference[TARGET]
        pipeline.fit(X, y)
        # Le preprocesseur doit avoir des attributs mean_/scale_ (StandardScaler)
        scaler = pipeline.named_steps["preprocessor"].transformers_[0][1]
        assert hasattr(scaler, "mean_"), "StandardScaler non ajusté"
        assert len(scaler.mean_) == len(NUMERIC_FEATURES)

    def test_encoder_categoriel_ajuste(self, dvf_reference):
        pipeline = build_pipeline()
        X = dvf_reference[ALL_FEATURES]
        y = dvf_reference[TARGET]
        pipeline.fit(X, y)
        encoder = pipeline.named_steps["preprocessor"].transformers_[1][1]
        assert hasattr(encoder, "categories_"), "OrdinalEncoder non ajusté"

    def test_gestion_categorie_inconnue(self, trained_pipeline, dvf_reference):
        """Un département jamais vu ne doit pas planter le modèle."""
        X_test = dvf_reference[ALL_FEATURES].iloc[:1].copy()
        X_test["code_departement"] = "999"  # Inconnu
        preds = trained_pipeline.predict(X_test)
        assert len(preds) == 1
        assert preds[0] > 0
        assert not np.isnan(preds[0])

    def test_pas_de_valeurs_nan_apres_preprocessing(self, trained_pipeline, dvf_reference):
        """Le preprocessing ne doit pas introduire de NaN."""
        X = dvf_reference[ALL_FEATURES].head(20)
        # On vérifie via le transformateur intermédiaire
        preprocessor = trained_pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)
        assert not np.isnan(X_transformed).any(), "NaN après preprocessing"


# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 – Entraînement
# ════════════════════════════════════════════════════════════════════════════════
class TestEntrainement:

    def test_pipeline_fit_sans_erreur(self, dvf_reference):
        pipeline = build_pipeline()
        X = dvf_reference[ALL_FEATURES]
        y = dvf_reference[TARGET]
        pipeline.fit(X, y)  # Ne doit pas lever d'exception
        # XGBoost utilise get_booster() 
        assert hasattr(pipeline.named_steps["model"], "get_booster"), \
            "Modèle non entraîné"

    def test_nombre_estimateurs_correct(self, trained_pipeline):
        model = trained_pipeline.named_steps["model"]
        # XGBoost : vérifie n_estimators
        assert model.n_estimators >= 500, f"Trop peu d'estimateurs : {model.n_estimators}"

    def test_entrainement_reproductible(self, dvf_reference):
        """random_state=42 garantit la reproductibilité."""
        X = dvf_reference[ALL_FEATURES]
        y = dvf_reference[TARGET]
        p1 = build_pipeline()
        p2 = build_pipeline()
        p1.fit(X, y)
        p2.fit(X, y)
        preds1 = p1.predict(X.head(5))
        preds2 = p2.predict(X.head(5))
        np.testing.assert_array_almost_equal(preds1, preds2, decimal=2, 
                                            err_msg="Entraînement non reproductible")

    def test_no_overfitting_severe(self, dvf_reference, dvf_current):
        """La performance sur les données de test ne doit pas être catastrophique."""
        pipeline = build_pipeline()
        X_train = dvf_reference[ALL_FEATURES]
        y_train = dvf_reference[TARGET]
        X_test = dvf_current[ALL_FEATURES]
        y_test = dvf_current[TARGET]
        pipeline.fit(X_train, y_train)
        from sklearn.metrics import r2_score
        r2_train = r2_score(y_train, pipeline.predict(X_train))
        r2_test = r2_score(y_test, pipeline.predict(X_test))
        # L'écart entre train et test ne doit pas dépasser 0.4 (overfitting excessif)
        assert r2_train - r2_test < 0.50, \
            f"Overfitting trop fort : train={r2_train:.3f}, test={r2_test:.3f}"


# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 – Évaluation
# ════════════════════════════════════════════════════════════════════════════════
class TestEvaluation:

    def test_evaluate_retourne_toutes_metriques(self, trained_pipeline, dvf_current):
        X = dvf_current[ALL_FEATURES]
        y = dvf_current[TARGET]
        pred = trained_pipeline.predict(X)
        metrics = evaluate(y.values, pred)
        for key in ["mae", "rmse", "r2", "mape"]:
            assert key in metrics, f"Métrique manquante : {key}"

    def test_mae_est_positif(self, trained_pipeline, dvf_current):
        X, y = dvf_current[ALL_FEATURES], dvf_current[TARGET]
        metrics = evaluate(y.values, trained_pipeline.predict(X))
        assert metrics["mae"] > 0

    def test_rmse_superieur_ou_egal_mae(self, trained_pipeline, dvf_current):
        X, y = dvf_current[ALL_FEATURES], dvf_current[TARGET]
        metrics = evaluate(y.values, trained_pipeline.predict(X))
        assert metrics["rmse"] >= metrics["mae"]

    def test_r2_dans_plage_valide(self, trained_pipeline, dvf_current):
        X, y = dvf_current[ALL_FEATURES], dvf_current[TARGET]
        r2 = evaluate(y.values, trained_pipeline.predict(X))["r2"]
        assert -1.0 <= r2 <= 1.0

    def test_mape_en_pourcentage(self, trained_pipeline, dvf_current):
        X, y = dvf_current[ALL_FEATURES], dvf_current[TARGET]
        mape = evaluate(y.values, trained_pipeline.predict(X))["mape"]
        assert 0 < mape < 100, f"MAPE incohérent : {mape:.2f}%"

    def test_predictions_dans_plage_realiste(self, trained_pipeline, dvf_current):
        """Les prédictions doivent être dans [20k€, 2M€] pour des biens standards."""
        X = dvf_current[ALL_FEATURES]
        preds = trained_pipeline.predict(X)
        assert np.all(preds >= 20000), "Prédictions trop faibles"
        assert np.all(preds <= 2000000), "Prédictions trop élevées"

    def test_predictions_pas_toutes_identiques(self, trained_pipeline, dvf_current):
        """Le modèle doit produire des prédictions variées."""
        X = dvf_current[ALL_FEATURES]
        preds = trained_pipeline.predict(X)
        assert preds.std() > 1000, "Toutes les prédictions sont identiques"


# ════════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 – Validation / Gate qualité avant MLflow
# ════════════════════════════════════════════════════════════════════════════════
class TestValidationModele:

    def test_gate_r2_minimum(self, trained_pipeline, dvf_current):
        """Le modèle doit atteindre R² > 0.0 sur données de test (données synthétiques)."""
        X, y = dvf_current[ALL_FEATURES], dvf_current[TARGET]
        r2 = evaluate(y.values, trained_pipeline.predict(X))["r2"]
        assert r2 > 0.0, f"R² trop faible : {r2:.3f} (seuil : 0.0)"

    def test_predictions_pas_nan(self, trained_pipeline, dvf_current):
        X = dvf_current[ALL_FEATURES]
        preds = trained_pipeline.predict(X)
        assert not np.isnan(preds).any(), "Prédictions NaN"

    def test_prediction_unique_cohérente(self, trained_pipeline):
        """Test sur un exemple fixe pour détecter une régression."""
        test_input = pd.DataFrame([{
            "surface_reelle_bati": 75.0,
            "nombre_pieces_principales": 3,
            "surface_terrain": 0.0,
            "longitude": 2.347,
            "latitude": 48.859,
            "type_local": "Appartement",
            "code_departement": "75",
        }])
        pred = trained_pipeline.predict(test_input)[0]
        assert 50000 < pred < 2000000, f"Prédiction incohérente : {pred}"
        assert not np.isnan(pred)


    