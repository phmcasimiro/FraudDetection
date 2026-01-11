# src/api/main.py
# API Principal usando FastAPI
# Elaborado por phmcasimiro
# Data: 10/01/2026

# Importação das classes FastAPI, HTTPException, Security e status que servem para criar a API, gerenciar as exceções, fornecer segurança e status, respectivamente
from fastapi import FastAPI, HTTPException, Security, status
# Importação da classe APIKeyHeader que serve para autenticação via API Key
from fastapi.security import APIKeyHeader
# Importação das classes TransactionInput e PredictionOutput que servem para validar os dados de entrada e saída
from src.schemas.schemas import TransactionInput, PredictionOutput
# Importação da classe predictor que serve para fazer a predição de fraude
from src.services.predictor import predictor
# Importação da biblioteca logging que serve para registrar logs da API
import logging
# Importação da biblioteca os que serve para interagir com o sistema operacional
import os
# Importação da biblioteca dotenv que serve para carregar as variáveis de ambiente
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar Logs da API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Configuração da API Key
API_KEY_NAME = "access_token"  # Nome do header de autenticação
API_KEY = os.getenv("API_KEY")  # Chave de API

if not API_KEY:
    raise ValueError("ERROR: A variável de ambiente API_KEY não está configurada.")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Função para validar a API Key
async def get_api_key(api_key_header: str = Security(api_key_header)):
    # Verifica se a API Key é válida
    if api_key_header == API_KEY:
        # Retorna a API Key
        return api_key_header
    # Lança uma exceção se a API Key for inválida
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Não foi possível validar as credenciais",
    )


# Inicializar App
app = FastAPI(
    title="API Detecção de Fraudes",
    description="API para detecção de fraudes em transações de cartão de crédito.",
    version="1.0.0",
)


# Endpoint de Health Check
@app.get("/", tags=["Health Check"])
def read_root():
    """
    Endpoint raiz para verificar se a API está online.
    """
    return {"message": "API Detecção de Fraudes está online!", "status": "online"}


# Endpoint de Predição
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_fraud(transaction: TransactionInput, api_key: str = Security(get_api_key)):
    """
    Endpoint para predição de fraude.
    Recebe dados da transação e retorna probabilidade de ser fraude.
    Requer autenticação via API Key no header 'access_token'.
    """
    try:
        logger.info("Recebendo nova transação para análise.")
        result = predictor.predict(transaction)
        logger.info(f"Predição realizada com sucesso: {result['status']}")
        return result
    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Erro interno no servidor: {str(e)}"
        )
