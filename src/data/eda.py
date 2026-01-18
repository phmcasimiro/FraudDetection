# eda.py
# Script para análise exploratória de dados (EDA)
# Elaborado por: phmcasimiro

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def run_eda():
    # 1. Carregar os dados
    df = pd.read_csv("src/data/creditcard.csv")

    # Criar pasta para salvar gráficos (se não existir)
    os.makedirs("artifacts", exist_ok=True)

    # 2. Analisar Distribuição da Classe (Target)
    print("Analisando distribuição de classes...")
    plt.figure(figsize=(8, 6))  # Define o tamanho da figura
    sns.countplot(x="Class", data=df)  # Cria o gráfico de contagem
    plt.title("Distribuição: 0 (Normal) vs 1 (Fraude)")  # Define o título do gráfico
    plt.savefig("artifacts/distribuicao_classe.png")  # Salva o gráfico

    # Imprimir proporção no terminal
    print(df["Class"].value_counts(normalize=True))

    # 3. Analisar Correlação
    print("Gerando matriz de correlação...")
    # Calculando correlação de todas as variáveis com a Classe
    # correlations = df.corr()['Class'].sort_values(ascending=False)

    # Gerar Heatmap das correlações
    plt.figure(figsize=(12, 10))  # Define o tamanho da figura
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")  # Cria o heatmap
    plt.title("Matriz de Correlação Global")  # Define o título do gráfico
    plt.savefig("artifacts/matriz_correlacao.png")  # Salva o gráfico

    print("EDA concluída. Gráficos salvos em /artifacts")


if __name__ == "__main__":
    run_eda()
