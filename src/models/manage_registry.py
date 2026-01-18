import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
import os
import sys

# Configuração
MLFLOW_DB_URI = "sqlite:///mlflow.db"
MODEL_NAME = "FraudDetectionRandomForest"
BASELINE_MODEL_NAME = "FraudDetectionBaseline"
PRODUCTION_MODEL_PATH = "artifacts/models/model.pkl"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    return MlflowClient()


def register_baseline_if_needed(client):
    print(f"\n--- Verificando Baseline ({BASELINE_MODEL_NAME}) ---")

    # Verificar se o modelo baseline já existe
    try:
        client.get_registered_model(BASELINE_MODEL_NAME)
        print(f"✅ Modelo '{BASELINE_MODEL_NAME}' já existe no registro.")
    except Exception:
        print(f"⚠️ Modelo '{BASELINE_MODEL_NAME}' não encontrado. Criando...")
        client.create_registered_model(BASELINE_MODEL_NAME)

    # Verificar versões
    versions = client.search_model_versions(f"name='{BASELINE_MODEL_NAME}'")
    if versions:
        print(f"✅ Já existem {len(versions)} versões do Baseline.")
    else:
        print(
            "⚠️ Nenhuma versão encontrada. Registrando 'model.pkl' atual como Baseline v1..."
        )

        if not os.path.exists(PRODUCTION_MODEL_PATH):
            print(f"❌ Erro: Arquivo {PRODUCTION_MODEL_PATH} não encontrado.")
            return

        # Para registrar um arquivo arbitrário como modelo, precisamos de um Run.
        # Vamos criar um Run específico para o Baseline.
        mlflow.set_experiment("fraud_detection_baseline_registration")
        with mlflow.start_run(run_name="Baseline Registration") as run:
            model = joblib.load(PRODUCTION_MODEL_PATH)
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=BASELINE_MODEL_NAME
            )
            print(f"✅ Baseline registrado com sucesso! Run ID: {run.info.run_id}")


def promote_latest_to_production(client):
    print(f"\n--- Promovendo {MODEL_NAME} para Produção ---")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(
            f"❌ Nenhum modelo encontrado com nome '{MODEL_NAME}'. Execute train.py primeiro."
        )
        return None

    # Pegar a última versão
    latest_version = versions[
        0
    ]  # search_model_versions retorna desc order por padrão? Vamos garantir.
    latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]

    print(
        f"Última versão encontrada: v{latest_version.version} (Stage: {latest_version.current_stage})"
    )

    if latest_version.current_stage != "Production":
        print(f"🔄 Promovendo v{latest_version.version} para 'Production'...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print("✅ Promoção concluída.")
    else:
        print("✅ Esta versão já está em Produção.")

    return latest_version.version


def test_load_production_model():
    print("\n--- Testando Carregamento de Produção ---")

    model_uri = f"models:/{MODEL_NAME}/Production"
    print(f"Tentando carregar de: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Modelo carregado com sucesso via MLflow!")

        # Teste rápido de predição
        X_test_path = "src/data/X_test.csv"
        if os.path.exists(X_test_path):
            print("Executando predição de teste...")
            X_test = pd.read_csv(X_test_path).head(5)
            preds = model.predict(X_test)
            print(f"Predições (primeiras 5): {preds}")

        return True
    except Exception as e:
        print(f"❌ Falha ao carregar modelo: {e}")
        return False


if __name__ == "__main__":
    client = setup_mlflow()
    register_baseline_if_needed(client)
    version = promote_latest_to_production(client)
    if version:
        success = test_load_production_model()
        if success:
            print("\n✅ TUDO PRONTO PARA SUBSTITUIÇÃO EM PRODUÇÃO!")
        else:
            print("\n❌ FALHA NO TESTE DE CARREGAMENTO.")
            sys.exit(1)
