# evaluate.py
# Avaliação e Validação do modelo treinado
# Elaborado por: phmcasimiro
# Data: 2026-01-05

import pandas as pd
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------------------------------
# ---------------------- FUNÇÃO AUXILIAR - CONFIGURAÇÃO DE LOGS ----------------------
# ------------------------------------------------------------------------------------


def configurar_logger():  # Configuração de logs
    os.makedirs("logs", exist_ok=True)  # Criação do diretório de logs
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )  # Timestamp para o nome do arquivo de log
    log_filename = f"logs/evaluate_{timestamp}.log"  # Nome do arquivo de log

    logger = logging.getLogger("FraudDetectionEval")  # Logger
    logger.setLevel(logging.INFO)  # Nível de logging

    file_handler = logging.FileHandler(log_filename)  # Handler de arquivo
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )  # Formatação do log

    console_handler = logging.StreamHandler()  # Handler de console
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )  # Formatação do log

    logger.addHandler(file_handler)  # Adiciona o handler de arquivo ao logger
    logger.addHandler(console_handler)  # Adiciona o handler de console ao logger
    return logger


# ------------------------------------------------------------------------------
# ---------------------- FUNÇÃO PRINCIPAL - AVALIAR MODELO ---------------------
# ------------------------------------------------------------------------------


def avaliar_modelo():  # Avaliação e Validação do modelo treinado
    logger = configurar_logger()
    logger.info("Iniciando script de avaliação...")

    # Caminhos dos arquivos
    model_path = "artifacts/models/model.pkl"
    X_test_path = "src/data/X_test.csv"
    y_test_path = "src/data/y_test.csv"
    output_dir = "artifacts/evaluation"
    os.makedirs(output_dir, exist_ok=True)

    # Gerar timestamp para versionamento dos artefatos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not all(os.path.exists(p) for p in [model_path, X_test_path, y_test_path]):
        logger.error(
            "Arquivos necessários não encontrados. Verifique o treino e o pré-processamento."
        )
        return

    try:
        # 1. CARREGAR MODELO E DADOS DE TESTE
        logger.info("Carregando modelo e dados de teste...")
        model = joblib.load(model_path)  # Carrega o modelo treinado
        X_test = pd.read_csv(X_test_path)  # Carrega os dados de teste
        y_test = pd.read_csv(y_test_path).values.ravel()  # Carrega os dados de teste

        # 2. REALIZAR PREDIÇÕES
        logger.info("Realizando predições no conjunto de teste...")
        y_pred = model.predict(X_test)  # Realiza predições no conjunto de teste
        y_probs = model.predict_proba(X_test)[:, 1]  # Probabilidades para a curva ROC

        # 3. GERAR RELATÓRIO DE MÉTRICAS (Precision, Recall, F1)
        report = classification_report(y_test, y_pred)  # Gera relatório de métricas
        logger.info("\n" + report)

        # Salvar relatório em texto com timestamp
        report_filename = f"{output_dir}/metrics_report_{timestamp}.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        logger.info(f"Relatório salvo em: {report_filename}")

        # 4. MATRIZ DE CONFUSÃO VISUAL
        logger.info("Gerando Matriz de Confusão...")
        cm = confusion_matrix(y_test, y_pred)  # Gera matriz de confusão
        plt.figure(figsize=(8, 6))  # Define o tamanho da figura
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)  # Gera heatmap
        plt.title(f"Matriz de Confusão - {timestamp}")  # Define o título
        plt.xlabel("Predição do Modelo")  # Define o eixo x
        plt.ylabel("Valor Real (Gabarito)")  # Define o eixo y
        plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png")  # Salva a imagem
        plt.close()

        # 5. CURVA ROC E AUC
        logger.info("Calculando AUC-ROC...")
        auc = roc_auc_score(y_test, y_probs)  # Calcula o AUC-ROC
        logger.info(f"AUC-ROC Score: {auc:.4f}")  # Imprime o AUC-ROC

        fpr, tpr, _ = roc_curve(y_test, y_probs)  # Calcula a curva ROC
        plt.figure(figsize=(8, 6))  # Define o tamanho da figura
        plt.plot(
            fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})"
        )  # Plota a curva ROC
        plt.plot(
            [0, 1], [0, 1], "k--", label="Aleatório (AUC = 0.5)"
        )  # Plota a curva ROC
        plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)")  # Define o eixo x
        plt.ylabel("Taxa de Verdadeiros Positivos (Recall)")  # Define o eixo y
        plt.title(f"Curva ROC - {timestamp}")  # Define o título
        plt.legend()  # Plota a legenda
        plt.savefig(f"{output_dir}/roc_curve_{timestamp}.png")  # Salva a imagem
        plt.close()

        logger.info(f"Avaliação concluída! Resultados salvos em: {output_dir}")

    except Exception as e:
        logger.exception("Erro durante a avaliação.")
        raise e


if __name__ == "__main__":
    avaliar_modelo()
