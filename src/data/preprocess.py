# preprocess.py
# Pré-processamento dos dados
# Elaborado por: phmcasimiro
# Data: 02/01/2026

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from src.data.pandera_schemas import validar_dados
from src.data.db import load_data, save_data


# -------------------------------------------------------------------------
# ITEM 4: FUNÇÃO DE PRÉ-PROCESSAMENTO DOS DADOS
# -------------------------------------------------------------------------
def executar_pre_processamento():

    print("Carregando dataset do banco de dados...")
    try:
        df = load_data("transactions")
        print(f"Dataset carregado com {df.shape[0]} linhas.")
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # -------------------------------------------------------------------------
    # ITEM 4.1: VERIFICAÇÃO DE INTEGRIDADE (DATA CLEANING)
    # -------------------------------------------------------------------------

    # Remover valores nulos
    nulos = df.isnull().sum().sum()  # Soma todos os valores nulos do dataset
    if nulos > 0:
        print(
            f"Limpando {nulos} valores nulos..."
        )  # Informa quantos valores nulos foram removidos
        df = df.dropna()  # Remove as linhas com valores vazios
    else:
        print("Integridade confirmada: Nenhum valor nulo encontrado.")

    # Remover Duplicatas
    duplicados = df.duplicated().sum()  # Soma todos os valores duplicados do dataset
    if duplicados > 0:
        print(
            f"Removendo {duplicados} transações duplicadas para evitar overfitting..."
        )
        df = df.drop_duplicates()  # Remove as linhas duplicadas

    # Validar com Pandera
    try:
        print("Validando integridade estatística com Pandera...")
        df = validar_dados(df)  # Agora a variável df existe e está limpa
        print("Dados validados pelo Pandera com sucesso!")
    except Exception as e:
        print(f"Erro na validação de dados: {e}")
        return

    # -------------------------------------------------------------------------
    # ITEM 4.2: ESCALONAMENTO DE ATRIBUTOS (FEATURE SCALING)
    # -------------------------------------------------------------------------

    # Criar objeto RobustScaler para tratar Outliers em Time e Amount)
    scaler = RobustScaler()

    # Ajustar e transformar as colunas Time e Amount
    print("Escalonando colunas 'Time' e 'Amount' com RobustScaler...")
    df["Time"] = scaler.fit_transform(
        df[["Time"]]
    )  # Ajustar e transformar a coluna Time
    df["Amount"] = scaler.fit_transform(
        df[["Amount"]]
    )  # Ajustar e transformar a coluna Amount

    # -------------------------------------------------------------------------
    # ITEM 4.3: DIVISÃO DE CONJUNTOS (SPLITTING)
    # -------------------------------------------------------------------------

    # Separação em X = Atributos (V1 a V28, Time, Amount) e  y = Alvo (Class)
    X = df.drop("Class", axis=1)  # Remove a coluna 'Class' do dataset
    y = df["Class"]  # Define a coluna 'Class' como alvo

    # Dividindo em conjuntos 80% treino e 20% teste
    # stratify=y garante que a proporção de fraudes seja a mesma nos dois grupos
    print("Dividindo dados em Treino e Teste (Estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,  # 20% do dataset para teste
        random_state=42,  # Garante que a divisão seja a mesma em todas as execuções
        stratify=y,  # Garante que a proporção de fraudes seja a mesma nos dois grupos
    )

    # -------------------------------------------------------------------------
    # ITEM 4.4: PREPARAÇÃO E EXPORTAÇÃO DOS ARQUIVOS DE TREINO E TESTE
    # -------------------------------------------------------------------------

    # OBS: Balanceamento será feito no script de treinamento
    # Neste trecho serão preparados os arquivos finais para o modelo.

    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    print(f"Fraudes no treino: {y_train.sum()} | Fraudes no teste: {y_test.sum()}")

    # Salvando os arquivos processados no Banco de Dados
    print("Salvando dados processados no banco de dados...")
    save_data(X_train, "X_train", if_exists="replace")
    save_data(X_test, "X_test", if_exists="replace")
    # Para salvar Series (y_train, y_test), precisamos converter para DataFrame
    save_data(y_train.to_frame(), "y_train", if_exists="replace")
    save_data(y_test.to_frame(), "y_test", if_exists="replace")

    print("Processamento concluído! Dados salvos em 'fraud_detection.db'")


if __name__ == "__main__":
    executar_pre_processamento()
