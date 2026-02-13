import mlflow
import requests

# 1. Vérifier que le serveur est accessible
try:
    response = requests.get("http://localhost:5001")
    print(f"✅ Serveur MLflow accessible: {response.status_code}")
except Exception as e:
    print(f"❌ Serveur MLflow inaccessible: {e}")

# 2. Vérifier l'URI configuré
mlflow.set_tracking_uri("http://localhost:5001")
print(f"✅ URI configuré: {mlflow.get_tracking_uri()}")

# 3. Tester un log simple
try:
    with mlflow.start_run():
        mlflow.log_param("test", "ok")
        print("✅ Log réussi!")
except Exception as e:
    print(f"❌ Échec du log: {e}")