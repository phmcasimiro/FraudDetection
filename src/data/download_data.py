# download_data.py
# Script para download do dataset do Kaggle
# Elaborado por: phmcasimiro

import kagglehub
import shutil
import os

def download_dataset():
    print("Iniciando download do Kaggle...")
    # Baixa a versão mais recente do dataset
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    
    # O kagglehub baixa o arquivo .csv para um cache.
    source_file = os.path.join(path, "creditcard.csv") # Localiza o arquivo .csv baixado
    destination_path = "src/data/creditcard.csv" # Define o caminho de destino
    
    # Move o arquivo para sua pasta de dados do projeto
    if os.path.exists(source_file):
        shutil.move(source_file, destination_path)
        print(f"Dataset movido com sucesso para: {destination_path}")
    else:
        print(f"Erro: Arquivo não encontrado em {source_file}")

if __name__ == "__main__":
    download_dataset()