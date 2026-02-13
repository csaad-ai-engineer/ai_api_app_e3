import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

print("Test MLFlow minimal")

mlflow.set_tracking_uri("http://localhost:5001")
print(f"URI: {mlflow.get_tracking_uri()}")

# Créer des données bidon
X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
model = RandomForestRegressor(n_estimators=10)
model.fit(X, y)

try:
    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        
        # Log simple
        mlflow.log_param("test", "hello")
        mlflow.log_metric("score", 0.95)
        
        # Log modèle simple
        mlflow.sklearn.log_model(model, "test_model")
        
        print("✅ Succès!")
        print(f"Voir: http://localhost:5001/#/experiments/0/runs/{run.info.run_id}")
except Exception as e:
    print(f"❌ Échec: {e}")
    import traceback
    traceback.print_exc()