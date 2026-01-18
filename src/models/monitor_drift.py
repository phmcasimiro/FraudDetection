# src/models/monitor_drift.py
# Script de Monitoramento de Data Drift com Evidently AI
# Elaborado por phmcasimiro
# Data: 18/01/2026

import pandas as pd
import mlflow
import os
import logging
import numpy as np

# Evidently imports (Legacy API for 0.7.x)
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset

# Project imports
from src.data.db import load_data

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do MLflow
if not mlflow.get_tracking_uri():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "fraud_detection_monitoring"
mlflow.set_experiment(EXPERIMENT_NAME)


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simula um Data Drift artificial para fins de demonstração.
    Altera a distribuição da coluna 'Amount' e 'V1'.
    """
    logger.info("🧪 Simulando Data Drift artificial...")
    df_drifted = df.copy()

    # Simular aumento drástico no valor das transações (Amount * 5)
    df_drifted["Amount"] = df_drifted["Amount"] * 5 + 100

    # Simular mudança na distribuição de V1 (Shift na média)
    df_drifted["V1"] = df_drifted["V1"] + 5.0

    return df_drifted


def run_monitoring():
    logger.info("🚀 Iniciando monitoramento de Data Drift...")

    # 1. Carregar Dados de Referência (Treino) e Atual (Teste)
    try:
        reference_data = load_data("X_train")
        current_data = load_data("X_test")

        # Carregar targets para Target Drift (Opcional, mas bom ter)
        reference_target = load_data("y_train")
        current_target = load_data("y_test")

        # Juntar features e target para o Evidently
        reference_data["Class"] = reference_target["Class"]
        current_data["Class"] = current_target["Class"]

        logger.info(
            f"Dados carregados. Ref: {reference_data.shape}, Curr: {current_data.shape}"
        )

    except Exception as e:
        logger.error(f"Erro ao carregar dados do banco: {e}")
        return

    # 2. Simular Drift (Para demonstração)
    # Vamos usar metade do X_test normal e metade "driftado" para ver o alerta
    drifted_part = simulate_drift(current_data.sample(frac=0.5, random_state=42))
    normal_part = current_data.drop(drifted_part.index)

    # O "Novo Lote" de dados que chegou na API (Mistura de normal com drift)
    current_batch = pd.concat([normal_part, drifted_part])

    logger.info("📊 Gerando relatório de Drift com Evidently...")

    # 3. Configurar e Gerar Relatório
    report = Report(
        metrics=[
            DataDriftPreset(),  # Verifica mudança nas colunas (X)
            TargetDriftPreset(),  # Verifica mudança na classe alvo (y)
        ]
    )

    report.run(reference_data=reference_data, current_data=current_batch)

    # 4. Salvar Relatório HTML
    report_path = "artifacts/drift_report.html"
    os.makedirs("artifacts", exist_ok=True)
    report.save_html(report_path)
    logger.info(f"✅ Relatório salvo em: {report_path}")

    # 5. Logar no MLflow
    with mlflow.start_run(run_name="Drift_Analysis_Simulation"):
        # Logar o HTML como artefato
        mlflow.log_artifact(report_path)

        # Extrair métricas principais (Ex: Quantas colunas sofreram drift)
        # O Evidently retorna um dicionário JSON complexo, vamos pegar um resumo simples
        results = report.as_dict()
        drift_share = results["metrics"][0]["result"]["drift_share"]
        number_of_drifted_columns = results["metrics"][0]["result"][
            "number_of_drifted_columns"
        ]

        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("drifted_columns", number_of_drifted_columns)

        if drift_share > 0.5:
            logger.warning(
                f"⚠️ ALERTA DE DRIFT: {drift_share*100:.1f}% das colunas mudaram de distribuição!"
            )
        else:
            logger.info(f"Status do Drift: {drift_share*100:.1f}% (Dentro do esperado)")

    logger.info("🏁 Monitoramento concluído com sucesso!")


if __name__ == "__main__":
    run_monitoring()
