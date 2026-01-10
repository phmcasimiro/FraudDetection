# src/data/pandera_schemas.py
# Project: Fraud Detection
# author: phmcasimiro
# date: 2026-01-07
# Schema for the data

import pandera.pandas as pa
from pandera import Column, Check

# Definir schema do dataset
transaction_schema = pa.DataFrameSchema(
    columns={
        "Time": Column(float, nullable=False),
        "Amount": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
        **{f"V{i}": Column(float, nullable=False) for i in range(1, 29)} # Usar dicionário para validar V1 até V28
        },
    strict=True, # Garante que não existam colunas extras não mapeadas
    coerce=True  # Tenta converter os dados para o tipo correto se possível
)

def validar_dados(df):
    """
    Função auxiliar para validar o DataFrame
    """
    return transaction_schema.validate(df)