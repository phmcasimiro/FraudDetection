# Fraud Detection ML API

 - This is a Machine Learning API that uses a Random Forest model to predict fraud based on a dataset of credit card transactions.

 - Essa é uma API de Machine Learning que usa um modelo de Random Forest para prever fraudes com base em um conjunto de dados de transações de cartão de crédito.

# 1. SETUP

## 1.1 Versões de Python e pip do sistema:

 ```bash
python --version
python3 --version
```
 - Verificar os caminhos executáveis de Python instalados no sistema:

```bash
which -a python python3
```

 - Verificar versões instaladas via pyenv:

```bash
pyenv versions
```

 - Verificar versão do ambiente virtual:

```bash
python -V
```
 -  Verificar a versão do pacote pip:

```bash
pip --version
```

## 1.2 Versionador de código (Git e GitHub):

```bash
git --version
git config --global user.name "phmcasimiro"
git config --global user.email "phmcasimiro@gmail.com"
```

 - Criar Repositório no GitHub e sincronizar com o repositório local:

```bash
# GitHub - Criação de repositório na Nuvem

# 1. Acessar o GitHub
<https://github.com/phmcasimiro>

# 2. Escolha o nome do repositório
<FraudDetection>

# 3. Escolha o nível de acesso (público ou privado)
<Público>

# 4. Crie o repositório (sem marcar nada em "initialize this repository with...")
<https://github.com/phmcasimiro/FraudDetection.git>

# Git - Sincronização de repositório local com o repositório na Nuvem
# 1. Inicialize o Git localmente
git init

# 2. Adicione os arquivos ao "palco"
git add .

# 3. Crie o primeiro commit (ponto de partida)
git commit -m "feat: Estrutura inicial do projeto de Detecção de Fraudes"

# 4. Renomeie a branch para main
git branch -m main

# 5. Conecte o repositório local ao GitHub
# Substitua pela URL que você copiou
git remote add origin https://github.com/phmcasimiro/FraudDetection.git

# 6. Envie o código para o GitHub
git push -u origin main
```

## 1.3 GERENCIADOR DE PROJETOS (GitHub Projects)
 - O GitHub Projects é uma ferramenta que permite gerenciar projetos de forma colaborativa.
 - Utilizaremos neste projeto para gerenciar as tarefas de desenvolvimento e construir um histórico de progresso.

```bash
# 1. Entrar no GitHub Projects
<https://github.com/users/phmcasimiro/projects/2>

# 2. Canto superior direito, clicar em "settings"
<https://github.com/users/phmcasimiro/projects/2/settings>

# 3. Selecionar Default Repository "FraudDetection"
```

## 1.4 ESTRUTURA DE DIRETÓRIOS

```bash
# Linux - Criação de diretórios
mkdir src
mkdir src/api
mkdir src/data
mkdir src/models
mkdir tests
mkdir logs
mkdir artifacts
mkdir artifacts/models
```
``` bash
# Estrutura de diretórios FraudDetection
├── artifacts/
│   └── models/
├── logs/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── data/
│   │   ├── __init__.py
│   ├── models/
│   |   └── __init__.py
│   ├── schemas/
│   |   ├── __init__.py
│   │   └── schemas.py
│   ├── services/
│   |   ├── __init__.py
│   │   └── services.py
├── tests/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 1.5 CRIAR ARQUIVOS `__init__.py`
 - Arquivos .py são considerados módulos em Python, e o arquivo `__init__.py` é um arquivo especial que define um diretório como um pacote Python.

```bash
# Linux - Criação de arquivos .py
touch src/__init__.py
touch src/api/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch tests/__init__.py
```

## 1.6 CRIAR ARQUIVOS DE CONFIGURAÇÃO DO PROJETO

**`.gitignore`**

```
venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
logs/*.log
.env
*.pkl
.DS_Store
```

**`requirements.txt`**
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.6.0
pytest==7.4.3
pytest-cov==4.1.0
black==24.1.1
ruff==0.1.15
mypy==1.8.0
httpx==0.26.0
scikit-learn==1.4.0
```
### 1.7 INSTALAR BIBLIOTECAS
```bash
pip install -r requirements.txt
```

### 1.8 TESTE FASTAPI
 - Criar arquivo **`test_api.py`**
```bash
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
return {"message": "Setup funcionando!"}
```
- Executar o arquivo **`test_api.py`**
```bash
uvicorn test_api:app --reload
```
 - Após o teste, **deletar** arquivo **`test_api.py`**

### 1.9 VERIFICAÇÃO FINAL DE AMBIENTE

```bash
python --version
pip --version
git --version
pip show fastapi
pip show pytest
```
------

### 2. Extração, Ingestão e Análise Exploratória (EDA)

#### 2.1 Extração e Ingestão de Dados
- Criar um script `src/data/download_data.py` para baixar o dataset mais atualizado .

```bash
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
```

#### 2.2 Análise Exploratória
- Antes de começar a codificar a API, é necessário entender os dados. 
- Verificar a correlação das variáveis `V1` a `V28` e a distribuição da variável alvo `Class`.
- Criar um script `src/data/eda.py` para implementar uma análise exploratória de dados (EDA) .
- **OBS:** Serão usadas bibliotecas de visualização que salvam gráficos como arquivos na pasta `artifacts/`.

```bash
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
    plt.figure(figsize=(8, 6)) # Define o tamanho da figura
    sns.countplot(x='Class', data=df) # Cria o gráfico de contagem
    plt.title("Distribuição: 0 (Normal) vs 1 (Fraude)") # Define o título do gráfico
    plt.savefig("artifacts/distribuicao_classe.png") # Salva o gráfico
    
    # Imprimir proporção no terminal
    print(df['Class'].value_counts(normalize=True))

    # 3. Analisar Correlação
    print("Gerando matriz de correlação...")
    # Calculando correlação de todas as variáveis com a Classe
    correlations = df.corr()['Class'].sort_values(ascending=False)
    
    # Gerar Heatmap das correlações
    plt.figure(figsize=(12, 10)) # Define o tamanho da figura
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm') # Cria o heatmap
    plt.title("Matriz de Correlação Global") # Define o título do gráfico
    plt.savefig("artifacts/matriz_correlacao.png") # Salva o gráfico
    
    print("EDA concluída. Gráficos salvos em /artifacts")

if __name__ == "__main__":
    run_eda()
```

#### Por que observar a correlação e a distribuição?

 - **Análise da Distribuição da Variável Alvo (`Class`):**
	 - A coluna `Class` possui dois valores: **0** (legítima) e **1** (fraude).
	 - É essencial verificar o balanceamento/desbalanceamento das variáveis, então, usaremos um gráfico de barras (Distribuição de Classes) para visualizar isso.
	 - Em fraudes, a classe "1" (fraude) costuma ser uma fração mínima, isto é, há uma barra enorme no 0 e uma quase invisível no 1.
	 - Em Machine Learning, isso é uma **Classe Desbalanceada**. Caso o modelo seja treinado assim, aprenderá que "quase sempre não é fraude" e ignorará as fraudes reais.

- **Correlação das variáveis `V1` a `V28`:** 
	- Neste dataset as variáveis `V1` a `V28` são resultado de um **PCA** (Principal Component Analysis).
	- O PCA transforma variáveis originais em novos componentes que são **independentes** entre si. 
	- Logo, se fizermos uma matriz de correlação entre as variáveis, a correlação será próxima de zero (0).
	- O Foco da Análise é a correlação das variáveis com a variável `Class/Fraude`, ou seja, descobrir quais variáveis `V` têm mais poder preditivo (influência) para determinar fraudes.
    
-  **1. Distribuição da Variável Alvo (`Class`)**

<p align="center">
  <img src="artifacts/distribuicao_classe.png" alt="Distribuição de Classes" width="600">
</p>

- Analisando o gráfico `distribuicao_classe.png` é possível verificar uma coluna gigante no valor **0** (Transações Legítimas) e uma quase invisível no valor **1** (Fraudes).
    
-   Em Machine Learning, isto exemplifica o conceito de **Desbalanceamento de Classe Severo**, isto é, no  dataset, menos de 0.2% dos dados são fraude.
    
-   Se o desbalanceamento não for tratado, o modelo de Random Forest aprenderá a sempre classificar como "0" (Transação Legítima), pois ele terá 99.8% de acurácia fazendo isso, mesmo falhando em detectar todas as fraudes.
   
- **Importante:** Vamos focar na métrica **Recall** a fim de não ignorar nenhuma fraude, mesmo que isso gere alguns alarmes falsos (Falsos Positivos).



-  **2. Correlação das Variáveis `V1` a `V28`**

<p align="center">
  <img src="artifacts/matriz_correlacao.png" alt="Correlação das Variáveis" width="600">
</p>

- O gráfico utiliza uma **escala de cores** para demonstrar a **força da relação entre as variáveis**.

- **Vermelho Forte** indica uma **correlação positiva perfeita (valor 1)**, isto é, quando uma variável aumenta, a outra também aumenta. Por isso há uma **linha vermelha na diagonal principal**, a qual representa a **correlação de uma variável consigo mesma**.

- **Azul Forte** indica uma **correlação negativa forte (valor -1)**, isto é, quando uma variável aumenta, a outra tende a diminuir.

- **Cores Claras/Neutras** Indicam **correlações fracas ou nulas (valor 0)**, ou seja, as variáveis apresentam comportamento independente entre si.

- O **Centro do gráfico** é marcado por cores neutras e azul clara, isto é, as variáveis não possuem correlação entre si em razão da técnica PCA aplicada ao conjunto de dados. Em outras palavras, as variáveis `V1` a `V28` têm correlação zero entre si (as células do mapa de calor fora da diagonal são neutras em sua maioria). A razão desse fenômeno é a aplicação da técnica **PCA** (Análise de Componentes Principais), uma técnica de engenharia de features que transforma dados correlacionados em componentes independentes.

- A análise de variância é uma técnica estatística que permite avaliar a variação de uma variável em relação a outra variável. Em outras palavras, verifica-se quais variáveis possuem maior variação quando a `Class/Fraude` é `1` e quando é `0`. Essas variáveis serão as mais importantes para o seu modelo de Random Forest.

Ao analisar a última linha (ou coluna) da matriz, que mostra a correlação com a `Class/Fraude`, verificamos que as variáveis `V17`, `V14` e `V12` apresentam correlações negativas relativamente fortes com a `Class/Fraude`. 


