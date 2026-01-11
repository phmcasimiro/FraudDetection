# predictor.py
# Arquivo com a lógica de predição
# Elaborado por phmcasimiro
# Data: 10/01/2026


import joblib
import pandas as pd
import logging
from src.schemas.schemas import TransactionInput

# Configurar logs para rastrear o que acontece na predição
logger = logging.getLogger(__name__)

class FraudPredictor:
    def __init__(self, model_path: str = "artifacts/models/model.pkl"):
        """
        Ao iniciar, o serviço carrega o modelo para a memória.
        Isso evita ler o arquivo do disco a cada nova transação.
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f" Modelo carregado com sucesso de: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {e}")
            raise e

    def predict(self, data: TransactionInput):
        """
        Recebe os dados validados pelo Pydantic, transforma em DataFrame
        e solicita a predição ao Random Forest.
        """
        # 1. Converter o objeto Pydantic em um dicionário e depois em DataFrame
        df_input = pd.DataFrame([data.model_dump()])

        # 2. Realizar a predição de classe (0 ou 1)
        prediction = self.model.predict(df_input)[0] # 0 (Legítima) ou 1 (Fraude)

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
            "status": status
        }

# Instância única (Singleton) para ser usada pela API
predictor = FraudPredictor()