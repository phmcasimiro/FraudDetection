# predictor.py
# Arquivo com a lógica de predição
# Elaborado por phmcasimiro
# Data: 10/01/2026


import joblib
import pandas as pd
import logging
import mlflow.sklearn
import os
from src.api.schemas import TransactionInput

# Configurar logs para rastrear o que acontece na predição
logger = logging.getLogger(__name__)

# Configurar URI do MLflow (caso não esteja configurada no ambiente)
# Em produção real, isso viria de variável de ambiente
if not mlflow.get_tracking_uri():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


class FraudPredictor:
    def __init__(
        self, model_name: str = "FraudDetectionRandomForest", stage: str = "Production"
    ):
        """
        Ao iniciar, o serviço carrega o modelo do MLflow Registry.
        Isso garante que estamos usando a versão marcada como Produção.
        """
        model_uri = f"models:/{model_name}/{stage}"
        try:
            logger.info(f"Tentando carregar modelo do MLflow: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Modelo carregado com sucesso de: {model_uri}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar o modelo do MLflow: {e}")
            # Fallback para arquivo local em caso de falha do MLflow (Opcional, mas recomendado)
            local_path = "artifacts/models/model.pkl"
            if os.path.exists(local_path):
                logger.warning(f"⚠️ Usando fallback local: {local_path}")
                self.model = joblib.load(local_path)
            else:
                raise e

    def predict(self, data: TransactionInput):
        """
        Recebe os dados validados pelo Pydantic, transforma em DataFrame
        e solicita a predição ao Random Forest.
        """
        # 1. Converter o objeto Pydantic em um dicionário e depois em DataFrame
        df_input = pd.DataFrame([data.model_dump()])

        # 2. Realizar a predição de classe (0 ou 1)
        prediction = self.model.predict(df_input)[0]  # 0 (Legítima) ou 1 (Fraude)

        # 3. Calcular a probabilidade de ser fraude
        # [0, 1] -> [Legítima, Fraude] -> Utilizou-se o índice 1 que é a probabilidade de ser fraude
        probability = self.model.predict_proba(df_input)[0][1]

        # 4. Definir uma mensagem de status baseada na probabilidade
        if probability > 0.8:
            status = "Risco Crítico: Bloqueio Imediato"
        elif probability > 0.5:
            status = "Risco Moderado: Requer Análise Humana"
        else:
            status = "Transação Aprovada"

        return {
            "is_fraud": int(prediction),
            "probability": float(probability),
            "status": status,
        }


# Instância única (Singleton) para ser usada pela API
predictor = FraudPredictor()
