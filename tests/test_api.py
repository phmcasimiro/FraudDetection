# tests/test_api.py
# Testes Automatizados para a API de Detecção de Fraudes
# Elaborado por phmcasimiro
# Data: 11/01/2026

# Importação da classe TestClient que serve para simular requisições HTTP
from fastapi.testclient import TestClient
# Importação da classe app que serve para inicializar a API
from src.api.main import app
# Importação da biblioteca os que serve para interagir com o sistema operacional
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente para o teste
load_dotenv()

# Cliente de teste do FastAPI (simula requisições HTTP)
client = TestClient(app)

# Recuperar a chave de API do ambiente
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Erro no Teste: API_KEY não encontrada no ambiente ou no .env")

HEADERS = {"access_token": API_KEY}


def test_health_check():
    """
    Testa se o endpoint raiz (/) retorna 200 OK e status online.
    """
    response = client.get("/")  # Simula alguém acessando http://localhost:8000/
    assert response.status_code == 200  # Expectativa de sucesso (200 OK)
    # Verificar se o JSON retornado é EXATAMENTE o que esperamos
    assert response.json() == {
        "message": "API Detecção de Fraudes está online!",
        "status": "online",
    }


def test_predict_legitimate_transaction():
    """
    Testa uma transação válida que deve ser processada com sucesso (200 OK).
    Envia um payload com todas as 30 features E o header de autenticação.
    """
    # Payload de teste (transação normal)
    payload = {
        "Time": 406.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62,
    }

    # Simular um POST para /predict enviando o JSON e o HEADER de autenticação
    response = client.post("/predict", json=payload, headers=HEADERS)

    # Verificar se a resposta é 200 OK
    assert response.status_code == 200
    data = response.json()

    # A resposta deve conter os campos is_fraud, probability e status
    assert "is_fraud" in data
    assert "probability" in data
    assert "status" in data

    # Garantir que a probabilidade seja um número decimal (float)
    assert isinstance(data["probability"], float)


def test_predict_invalid_amount():
    """
    Testa se a API rejeita um valor de transação negativo (422 Unprocessable Entity).
    """
    payload = {
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
        "Amount": -50.00,  # Valor inválido!
    }

    # Mesmo com erro de validação, precisamos enviar o token para passar da porta de segurança
    response = client.post("/predict", json=payload, headers=HEADERS)

    assert (
        response.status_code == 422
    )  # Expectativa de falha (422 Unprocessable Entity)

    assert (
        "Amount" in response.text
    )  # Verificar se a mensagem de erro menciona o campo Amount


def test_predict_unauthorized():
    """
    Testa se a API bloqueia acesso sem a chave de API (403 Forbidden).
    """
    payload = {
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
        "Amount": 100.00,
    }

    # Tenta acessar SEM o header de autenticação
    response = client.post("/predict", json=payload)

    assert response.status_code == 403  # Deve ser proibido!
    assert response.json() == {"detail": "Não foi possível validar as credenciais"}
