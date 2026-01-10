from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    """
    Schema de entrada para uma única transação de cartão de crédito.
    Inclui as 30 features que o modelo Random Forest espera.
    """
    Time: float = Field(..., description="Segundos decorridos desde a primeira transação", example=406.0)
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., ge=0, description="Valor da transação (deve ser >= 0)", example=149.62)

class PredictionOutput(BaseModel):
    """
    Schema de saída da API com o resultado da classificação.
    """
    is_fraud: int = Field(..., description="0 para Legítima, 1 para Fraude")
    probability: float = Field(..., description="Probabilidade da transação ser fraude")
    status: str = Field(..., description="Mensagem descritiva do resultado")