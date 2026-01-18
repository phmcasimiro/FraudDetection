import pandas as pd
import joblib
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score
import os


def compare_models():
    print("--- Comparação de Modelos: Produção (Legacy) vs MLflow (Novo) ---")

    # 1. Carregar Dados de Teste
    X_test_path = "src/data/X_test.csv"
    y_test_path = "src/data/y_test.csv"

    if not os.path.exists(X_test_path):
        print("Erro: Dados de teste não encontrados.")
        return

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # 2. Carregar Modelo de Produção (Legacy .pkl)
    prod_model_path = "artifacts/models/model.pkl"
    print(f"\nCarregando Modelo de Produção: {prod_model_path}")
    try:
        prod_model = joblib.load(prod_model_path)
        prod_preds = prod_model.predict(X_test)
        prod_probs = prod_model.predict_proba(X_test)[:, 1]

        prod_auc = roc_auc_score(y_test, prod_probs)
        prod_report = classification_report(y_test, prod_preds, output_dict=True)
        print("✅ Modelo de Produção carregado.")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo de produção: {e}")
        return

    # 3. Carregar Modelo do MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "FraudDetectionRandomForest"
    model_version = 1  # Assumindo versão 1, ou poderia buscar a latest
    model_uri = f"models:/{model_name}/{model_version}"

    print(f"\nCarregando Modelo MLflow: {model_uri}")
    try:
        mlflow_model = mlflow.sklearn.load_model(model_uri)
        mlflow_preds = mlflow_model.predict(X_test)
        mlflow_probs = mlflow_model.predict_proba(X_test)[:, 1]

        mlflow_auc = roc_auc_score(y_test, mlflow_probs)
        mlflow_report = classification_report(y_test, mlflow_preds, output_dict=True)
        print("✅ Modelo MLflow carregado.")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo MLflow: {e}")
        return

    # 4. Comparar Métricas
    print("\n--- Resultados da Comparação ---")
    print(
        f"{'Métrica':<20} | {'Produção (.pkl)':<20} | {'MLflow (Registry)':<20} | {'Diferença':<10}"
    )
    print("-" * 80)

    metrics = [
        ("AUC-ROC", prod_auc, mlflow_auc),
        ("Acurácia", prod_report["accuracy"], mlflow_report["accuracy"]),
        ("Recall (Fraude)", prod_report["1"]["recall"], mlflow_report["1"]["recall"]),
        (
            "Precision (Fraude)",
            prod_report["1"]["precision"],
            mlflow_report["1"]["precision"],
        ),
        (
            "F1-Score (Fraude)",
            prod_report["1"]["f1-score"],
            mlflow_report["1"]["f1-score"],
        ),
    ]

    for name, val_prod, val_mlflow in metrics:
        diff = val_mlflow - val_prod
        print(
            f"{name:<20} | {val_prod:.4f}{' '*14} | {val_mlflow:.4f}{' '*14} | {diff:+.4f}"
        )

    # 5. Conclusão
    print("\n--- Conclusão ---")
    if abs(prod_auc - mlflow_auc) < 0.0001:
        print(
            "✅ Os modelos são idênticos (como esperado, pois o MLflow registrou o mesmo treino)."
        )
    else:
        print(
            "⚠️ Os modelos apresentam diferenças. Verifique se foram treinados com os mesmos dados/seeds."
        )


if __name__ == "__main__":
    compare_models()
