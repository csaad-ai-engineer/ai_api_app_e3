"""
Tests unitaires – Pipeline ML (test_ml.py)
Valide : chargement des données, pipeline d'entraînement, métriques minimales.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ml.train import build_pipeline, evaluate, NUMERIC_FEATURES, CATEGORIAL_FEATURES


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "surface_reelle_bati":       np.random.uniform(20, 200, n),
        "nombre_pieces_principales": np.random.randint(1, 8, n),
        "surface_terrain":           np.random.uniform(0, 500, n),
        "longitude":                 np.random.uniform(-1, 5, n),
        "latitude":                  np.random.uniform(43, 50, n),
        "type_local":                np.random.choice(["Appartement", "Maison"], n),
        "code_departement":          np.random.choice(["75", "69", "13", "33"], n),
        "valeur_fonciere":           np.random.uniform(80_000, 600_000, n),
    })


@pytest.fixture
def trained_pipeline(sample_df):
    X = sample_df[NUMERIC_FEATURES + CATEGORIAL_FEATURES]
    y = sample_df["valeur_fonciere"]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline, X, y


# ── Tests pipeline ─────────────────────────────────────────────────────────────
class TestPipeline:

    def test_build_pipeline_not_none(self):
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_pipeline_has_preprocessor_and_model(self):
        pipeline = build_pipeline()
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_pipeline_fit_predict(self, sample_df):
        X = sample_df[NUMERIC_FEATURES + CATEGORIAL_FEATURES]
        y = sample_df["valeur_fonciere"]
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert all(p > 0 for p in preds)

    def test_predictions_are_positive(self, trained_pipeline):
        _, X, _ = trained_pipeline
        pipeline, _, _ = trained_pipeline
        preds = pipeline.predict(X)
        assert np.all(preds > 0)

    def test_single_prediction(self, trained_pipeline):
        pipeline, X, _ = trained_pipeline
        single = X.iloc[:1]
        pred = pipeline.predict(single)
        assert pred.shape == (1,)
        assert pred[0] > 0

    def test_unknown_category_handled(self, trained_pipeline):
        """Le pipeline ne doit pas planter sur un département inconnu."""
        pipeline, X, _ = trained_pipeline
        test_row = X.iloc[:1].copy()
        test_row["code_departement"] = "999"
        pred = pipeline.predict(test_row)
        assert len(pred) == 1


# ── Tests métriques ────────────────────────────────────────────────────────────
class TestMetrics:

    def test_evaluate_returns_all_keys(self, trained_pipeline):
        pipeline, X, y = trained_pipeline
        preds   = pipeline.predict(X)
        metrics = evaluate(y, preds)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

    def test_evaluate_r2_reasonable(self, trained_pipeline):
        """Sur données synthétiques, R² doit être > -1 (non catastrophique)."""
        pipeline, X, y = trained_pipeline
        preds   = pipeline.predict(X)
        metrics = evaluate(y, preds)
        assert metrics["r2"] > -1.0

    def test_evaluate_mae_positive(self, trained_pipeline):
        pipeline, X, y = trained_pipeline
        preds   = pipeline.predict(X)
        metrics = evaluate(y, preds)
        assert metrics["mae"] >= 0

    def test_evaluate_rmse_gte_mae(self, trained_pipeline):
        pipeline, X, y = trained_pipeline
        preds   = pipeline.predict(X)
        metrics = evaluate(y, preds)
        assert metrics["rmse"] >= metrics["mae"]