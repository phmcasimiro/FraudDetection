# train.py
# Treinamento do modelo com técnica híbrida SMOTE + Tomek Links
# Elaborado por: phmcasimiro
# Data: 2026-01-04
# Atualizado: 2026-01-17 - Adicionado MLflow para tracking

import joblib
import os
import logging
from datetime import datetime
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from src.data.db import load_data

# Configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud_detection_training")

# ------------------------------------------------------------------------------------
# ---------------------- FUNÇÃO AUXILIAR - CONFIGURAÇÃO DE LOGS ----------------------
# ------------------------------------------------------------------------------------


def configurar_logger():  # Configuração de Logs
    os.makedirs("logs", exist_ok=True)  # Cria o diretório logs se não existir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Gera um timestamp
    log_filename = f"logs/train_{timestamp}.log"  # Nome do arquivo de log

    # Configura o logger
    logger = logging.getLogger("FraudDetectionTrain")  # Cria o logger
    logger.setLevel(logging.INFO)  # Define o nível de logging

    # Handler para Arquivo
    file_handler = logging.FileHandler(log_filename)  # Cria o handler para arquivo
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )  # Define o formato do log

    # Handler para Console (Terminal)
    console_handler = logging.StreamHandler()  # Cria o handler para console
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )  # Define o formato do log

    logger.addHandler(file_handler)  # Adiciona o handler para arquivo
    logger.addHandler(console_handler)  # Adiciona o handler para console

    return logger


# ------------------------------------------------------------------------------
# ---------------------- FUNÇÃO PRINCIPAL - TREINAR MODELO----------------------
# ------------------------------------------------------------------------------


def treinar_modelo():  # Função principal para treinar o modelo
    logger = configurar_logger()  # Configuração de Logs
    logger.info("Iniciando script de treinamento...")

    try:
        # Iniciar MLflow Run
        with mlflow.start_run():
            # Adicionar tags ao experimento
            mlflow.set_tag("autor", "phmcasimiro")
            mlflow.set_tag("projeto", "FraudDetection")
            mlflow.set_tag("versao", "1.1")  # Bump version for DB integration

            # 1. CARREGAR DADOS
            logger.info("Carregando dados de treino do banco de dados...")
            try:
                X_train = load_data("X_train")
                y_train = load_data("y_train").values.ravel()
                logger.info(f"Dados carregados. Shape original: {X_train.shape}")
            except Exception as e:
                logger.error(f"Erro ao carregar dados do banco: {e}")
                return

            # Logar informações do dataset
            mlflow.log_param("dataset_size", X_train.shape[0])
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("fraud_percentage", (y_train.sum() / len(y_train)) * 100)

            # 2. APLICAR TÉCNICA HÍBRIDA (SMOTETomek) - OPÇÃO D
            # Inicialmente o SMOTE cria dados sintéticos da classe minoritária (fraude)
            # Posteriormente o Tomek Links remove os pares de pontos de classes diferentes muito próximos
            logger.info(
                "Aplicando SMOTETomek para balancear os dados (pode demorar)..."
            )
            smt = SMOTETomek(random_state=42)  # Inicializa o SMOTETomek
            X_resampled, y_resampled = smt.fit_resample(
                X_train, y_train
            )  # Aplica o SMOTETomek

            logger.info(f"Dados originais: {len(X_train)} amostras")
            logger.info(f"Dados após SMOTETomek: {len(X_resampled)} amostras")

            # Logar informações do balanceamento
            mlflow.log_param("balancing_method", "SMOTETomek")
            mlflow.log_param("resampled_size", len(X_resampled))

            # 3. CONFIGURAR E TREINAR O MODELO (RANDOM FOREST)
            # Como os dados já estão balanceados pelo SMOTETomek,
            # não usaremos o class_weight='balanced' aqui.
            logger.info("Iniciando treinamento do Random Forest...")

            n_estimators = 100  # Número de árvores
            max_depth = 10  # Profundidade máxima das árvores

            modelo = RandomForestClassifier(  # Inicializa o Random Forest
                n_estimators=n_estimators,  # Número de árvores
                max_depth=max_depth,  # Profundidade máxima das árvores
                random_state=42,  # Estado aleatório para reproducibilidade
                n_jobs=-1,  # Utiliza todos os núcleos do processador
            )

            # Logar hiperparâmetros
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("algorithm", "RandomForest")

            logger.info(
                f"Hiperparâmetros: n_estimators={n_estimators}, max_depth={max_depth}"
            )

            modelo.fit(X_resampled, y_resampled)  # Treina o Random Forest
            logger.info("Treinamento concluído com sucesso!")

            # Calcular score de treino
            train_score = modelo.score(X_resampled, y_resampled)
            mlflow.log_metric("train_accuracy", train_score)
            logger.info(f"Acurácia no treino: {train_score:.4f}")

            # 4. EXPORTAR O MODELO
            os.makedirs(
                "artifacts/models", exist_ok=True
            )  # Cria o diretório artifacts/models se não existir
            model_path = "artifacts/models/model.pkl"  # Caminho do arquivo do modelo
            joblib.dump(modelo, model_path)  # Exporta o modelo (compatibilidade)
            logger.info(f"Modelo exportado para: {model_path}")

            # Logar o modelo no MLflow
            mlflow.sklearn.log_model(
                sk_model=modelo,
                artifact_path="model",
                registered_model_name="FraudDetectionRandomForest",
            )
            logger.info("Modelo registrado no MLflow com sucesso!")

    except Exception as e:
        logger.exception("Ocorreu um erro fatal durante o treinamento.")
        raise e


if __name__ == "__main__":
    treinar_modelo()
