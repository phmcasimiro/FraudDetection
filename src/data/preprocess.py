# preprocess.py
# Pré-processamento dos dados
# Elaborado por: phmcasimiro
# Data: 02/01/2026

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os
from src.data.pandera_schemas import validar_dados

# -------------------------------------------------------------------------
# ITEM 4: FUNÇÃO DE PRÉ-PROCESSAMENTO DOS DADOS
# -------------------------------------------------------------------------
def executar_pre_processamento(): 
    
    # Definir o caminho do dataset
    path_raw_data = "src/data/creditcard.csv"
    # Verificar se o arquivo existe
    if not os.path.exists(path_raw_data):
        print("Erro: Arquivo creditcard.csv não encontrado. Rode o download primeiro.")
        return

    # Carregar o dataset
    df = pd.read_csv(path_raw_data)
    print(f"Dataset carregado com {df.shape[0]} linhas.")

    # -------------------------------------------------------------------------
    # ITEM 4.1: VERIFICAÇÃO DE INTEGRIDADE (DATA CLEANING)
    # -------------------------------------------------------------------------
    
    # Remover valores nulos
    nulos = df.isnull().sum().sum() # Soma todos os valores nulos do dataset
    if nulos > 0:
        print(f"Limpando {nulos} valores nulos...") # Informa quantos valores nulos foram removidos
        df = df.dropna() # Remove as linhas com valores vazios
    else:
        print("Integridade confirmada: Nenhum valor nulo encontrado.")

    # Remover Duplicatas
    duplicados = df.duplicated().sum() # Soma todos os valores duplicados do dataset
    if duplicados > 0:
        print(f"Removendo {duplicados} transações duplicadas para evitar overfitting...")
        df = df.drop_duplicates() # Remove as linhas duplicadas

    # Validar com Pandera
    try:
        print("Validando integridade estatística com Pandera...")
        df = validar_dados(df) # Agora a variável df existe e está limpa
        print("Dados validados pelo Pandera com sucesso!")
    except Exception as e:
        print(f"Erro na validação de dados: {e}")
        return
    
    # -------------------------------------------------------------------------
    # ITEM 4.2: ESCALONAMENTO DE ATRIBUTOS (FEATURE SCALING)
    # -------------------------------------------------------------------------
    
    # Criar objeto RobustScaler (ideal para lidar com Outliers em Time e Amount)
    scaler = RobustScaler()

    # Ajustar e transformar as colunas Time e Amount
    print("Escalonando colunas 'Time' e 'Amount' com RobustScaler...")
    df['Time'] = scaler.fit_transform(df[['Time']]) # Ajustar e transformar a coluna Time
    df['Amount'] = scaler.fit_transform(df[['Amount']]) # Ajustar e transformar a coluna Amount

    # -------------------------------------------------------------------------
    # ITEM 4.3: DIVISÃO DE CONJUNTOS (SPLITTING)
    # -------------------------------------------------------------------------
    
        # Separação em X = Atributos (V1 a V28, Time, Amount) e  y = Alvo (Class)
    X = df.drop('Class', axis=1) # Remove a coluna 'Class' do dataset
    y = df['Class'] # Define a coluna 'Class' como alvo

    # Dividindo em conjuntos 80% treino e 20% teste
    # stratify=y garante que a proporção de fraudes seja a mesma nos dois grupos
    print("Dividindo dados em Treino e Teste (Estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, # 20% do dataset para teste
        random_state=42, # Garante que a divisão seja a mesma em todas as execuções
        stratify=y # Garante que a proporção de fraudes seja a mesma nos dois grupos
    )

    # -------------------------------------------------------------------------
    # ITEM 4.4: PREPARAÇÃO E EXPORTAÇÃO DOS ARQUIVOS DE TREINO E TESTE 
    # -------------------------------------------------------------------------
    
    # OBS: Balanceamento será feito no script de treinamento 
    # Neste trecho serão preparados os arquivos finais para o modelo.
    
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    print(f"Fraudes no treino: {y_train.sum()} | Fraudes no teste: {y_test.sum()}")

    # Salvando os arquivos processados para a Etapa 5 (Treinamento)
    os.makedirs("src/data/", exist_ok=True) # Garante que o diretório exista
    X_train.to_csv("src/data/X_train.csv", index=False) # Salva o arquivo X_train.csv
    X_test.to_csv("src/data/X_test.csv", index=False) # Salva o arquivo X_test.csv
    y_train.to_csv("src/data/y_train.csv", index=False) # Salva o arquivo y_train.csv
    y_test.to_csv("src/data/y_test.csv", index=False) # Salva o arquivo y_test.csv
    
    print("Processamento concluído! Arquivos salvos em src/data/")

if __name__ == "__main__":
    executar_pre_processamento()