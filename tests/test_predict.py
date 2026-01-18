# tests/test_predict.py
# Testes Unitários para o Serviço de Predição (FraudPredictor)
# Elaborado por phmcasimiro
# Data: 18/01/2026

import pytest
from unittest.mock import patch, MagicMock
from src.models.predictor import FraudPredictor
from src.api.schemas import TransactionInput

# Dados de exemplo para os testes
SAMPLE_TRANSACTION_DATA = {
    "Time": 0.0,
    "V1": 0.0,
    "V2": 0.0,
    "V3": 0.0,
    "V4": 0.0,
    "V5": 0.0,
    "V6": 0.0,
    "V7": 0.0,
    "V8": 0.0,
    "V9": 0.0,
    "V10": 0.0,
    "V11": 0.0,
    "V12": 0.0,
    "V13": 0.0,
    "V14": 0.0,
    "V15": 0.0,
    "V16": 0.0,
    "V17": 0.0,
    "V18": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V21": 0.0,
    "V22": 0.0,
    "V23": 0.0,
    "V24": 0.0,
    "V25": 0.0,
    "V26": 0.0,
    "V27": 0.0,
    "V28": 0.0,
    "Amount": 100.0,
}


@pytest.fixture
def mock_mlflow():
    with patch("src.models.predictor.mlflow") as mock:
        yield mock


@pytest.fixture
def mock_joblib():
    with patch("src.models.predictor.joblib") as mock:
        yield mock


@pytest.fixture
def mock_os_path():
    with patch("src.models.predictor.os.path.exists") as mock:
        yield mock


class TestFraudPredictorInitialization:

    def test_init_success_mlflow(self, mock_mlflow):
        """Testa se o modelo é carregado corretamente do MLflow."""
        # Configurar o mock para retornar um modelo fake
        fake_model = MagicMock()
        mock_mlflow.sklearn.load_model.return_value = fake_model

        predictor = FraudPredictor()

        # Verificar se tentou carregar do MLflow
        mock_mlflow.sklearn.load_model.assert_called_once()
        assert predictor.model == fake_model

    def test_init_fallback_local(self, mock_mlflow, mock_joblib, mock_os_path):
        """Testa o fallback para arquivo local quando o MLflow falha."""
        # Simular erro no MLflow
        mock_mlflow.sklearn.load_model.side_effect = Exception("MLflow fora do ar")

        # Simular que o arquivo local existe
        mock_os_path.return_value = True

        # Configurar mock do joblib
        fake_local_model = MagicMock()
        mock_joblib.load.return_value = fake_local_model

        predictor = FraudPredictor()

        # Verificar chamadas
        mock_mlflow.sklearn.load_model.assert_called_once()  # Tentou MLflow
        mock_joblib.load.assert_called_once_with(
            "artifacts/models/model.pkl"
        )  # Usou fallback
        assert predictor.model == fake_local_model

    def test_init_failure(self, mock_mlflow, mock_joblib, mock_os_path):
        """Testa se levanta exceção quando ambos (MLflow e Local) falham."""
        mock_mlflow.sklearn.load_model.side_effect = Exception("MLflow Error")
        mock_os_path.return_value = False  # Arquivo local não existe

        with pytest.raises(Exception) as excinfo:
            FraudPredictor()

        assert "MLflow Error" in str(excinfo.value)


class TestFraudPredictorLogic:

    def test_predict_fraud_critical(self, mock_mlflow):
        """Testa predição de fraude com alta probabilidade."""
        fake_model = MagicMock()
        fake_model.predict.return_value = [1]  # Classe 1 (Fraude)
        fake_model.predict_proba.return_value = [
            [0.1, 0.95]
        ]  # 95% probabilidade de fraude
        mock_mlflow.sklearn.load_model.return_value = fake_model

        predictor = FraudPredictor()
        input_data = TransactionInput(**SAMPLE_TRANSACTION_DATA)

        result = predictor.predict(input_data)

        assert result["is_fraud"] == 1
        assert result["probability"] == 0.95
        assert result["status"] == "Risco Crítico: Bloqueio Imediato"

    def test_predict_legitimate(self, mock_mlflow):
        """Testa predição de transação legítima."""
        fake_model = MagicMock()
        fake_model.predict.return_value = [0]  # Classe 0 (Legítima)
        fake_model.predict_proba.return_value = [
            [0.9, 0.1]
        ]  # 10% probabilidade de fraude
        mock_mlflow.sklearn.load_model.return_value = fake_model

        predictor = FraudPredictor()
        input_data = TransactionInput(**SAMPLE_TRANSACTION_DATA)

        result = predictor.predict(input_data)

        assert result["is_fraud"] == 0
        assert result["probability"] == 0.1
        assert result["status"] == "Transação Aprovada"
