# Fraud Detection ML API

 - Essa é uma API de Machine Learning que usa um modelo de Random Forest para prever fraudes com base em um conjunto de dados de transações de cartão de crédito.

- Pipeline de Dados do Modelo de Machine Learning:
    - Definição do Problema
    - Extração e Ingestão de Dados
    - Análise Exploratória dos Dados
    - Limpeza e Preparação dos Dados
    - Treinamento do Modelo (Experimentar Modelos)
    - Avaliação do Modelo (Validação do Modelo)
    - Deploy do Modelo (Produção)
    - Monitoramento do Modelo (Produção)
    - Atualização do Modelo (Produção)

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

### 2. Extração e Ingestão de Dados

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

### 3. Análise Exploratória de Dados (EDA)
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

#### **3.1 Distribuição da Variável Alvo (`Class`):**

<p align="center">
  <img src="artifacts/distribuicao_classe.png" alt="Distribuição de Classes" width="600">
</p>

- A coluna `Class` possui dois valores: **0** (legítima) e **1** (fraude).

- É essencial verificar o balanceamento/desbalanceamento das variáveis, então, usaremos um gráfico de barras (Distribuição de Classes) para visualizar isso.

- Em fraudes, a classe "1" (fraude) costuma ser uma fração mínima, isto é, há uma barra enorme no 0 e uma quase invisível no 1.

- Em Machine Learning, isso é uma **Classe Desbalanceada**. Caso o modelo seja treinado assim, aprenderá que "quase sempre não é fraude" e ignorará as fraudes reais.

- Analisando o gráfico `distribuicao_classe.png` é possível verificar uma coluna gigante no valor **0** (Transações Legítimas) e uma quase invisível no valor **1** (Fraudes).
    
-   Em Machine Learning, isto exemplifica o conceito de **Desbalanceamento de Classe Severo**, isto é, no  dataset, menos de 0.2% dos dados são fraude.
    
-   Se o desbalanceamento não for tratado, o modelo de Random Forest aprenderá a sempre classificar como "0" (Transação Legítima), pois ele terá 99.8% de acurácia fazendo isso, mesmo falhando em detectar todas as fraudes.
   
- **Importante:** Vamos focar na métrica **Recall** a fim de não ignorar nenhuma fraude, mesmo que isso gere alguns alarmes falsos (Falsos Positivos).


#### **3.2 Correlação das Variáveis `V1` a `V28`**

<p align="center">
  <img src="artifacts/matriz_correlacao.png" alt="Correlação das Variáveis" width="600">
</p>

- Neste dataset as variáveis `V1` a `V28` são resultado de um **PCA** (Principal Component Analysis).

- O PCA transforma variáveis originais em novos componentes que são **independentes** entre si. 

- Logo, se fizermos uma matriz de correlação entre as variáveis, a correlação será próxima de zero (0).

- O Foco da Análise é a correlação das variáveis com a variável `Class/Fraude`, ou seja, descobrir quais variáveis `V` têm mais poder preditivo (influência) para determinar fraudes.

- O gráfico utiliza uma **escala de cores** para demonstrar a **força da relação entre as variáveis**.

- **Vermelho Forte** indica uma **correlação positiva perfeita (valor 1)**, isto é, quando uma variável aumenta, a outra também aumenta. Por isso há uma **linha vermelha na diagonal principal**, a qual representa a **correlação de uma variável consigo mesma**.

- **Azul Forte** indica uma **correlação negativa forte (valor -1)**, isto é, quando uma variável aumenta, a outra tende a diminuir.

- **Cores Claras/Neutras** Indicam **correlações fracas ou nulas (valor 0)**, ou seja, as variáveis apresentam comportamento independente entre si.

- O **Centro do gráfico** é marcado por cores neutras e azul clara, isto é, as variáveis não possuem correlação entre si em razão da técnica PCA aplicada ao conjunto de dados. Em outras palavras, as variáveis `V1` a `V28` têm correlação zero entre si (as células do mapa de calor fora da diagonal são neutras em sua maioria). A razão desse fenômeno é a aplicação da técnica **PCA** (Análise de Componentes Principais), uma técnica de engenharia de features que transforma dados correlacionados em componentes independentes.

- A análise de variância é uma técnica estatística que permite avaliar a variação de uma variável em relação a outra variável. Em outras palavras, verifica-se quais variáveis possuem maior variação quando a `Class/Fraude` é `1` e quando é `0`. Essas variáveis serão as mais importantes para o seu modelo de Random Forest.

Ao analisar a última linha (ou coluna) da matriz, que mostra a correlação com a `Class/Fraude`, verificamos que as variáveis `V17`, `V14` e `V12` apresentam correlações negativas relativamente fortes com a `Class/Fraude`. 

### 4. LIMPEZA E PRÉ-PROCESSAMENTO DOS DADOS

- A partir da identificação das variáveis críticas (V12, V14 e V17) e do desbalanceamento do dataset, a **limpeza e pré-processamento dos dados** garantirão que o modelo de Random Forest seja treinado com dados de qualidade, evitando que o modelo seja enganado pelo ruído dos dados e que ele aprenda a classificar como "0" (Transação Legítima) mesmo falhando em detectar todas as fraudes. 

#### **4.1 VERIFICAÇÃO DE INTEGRIDADE (DATA CLEANING)**

- O objetivo é garantir que todas as entradas estejam em uma escala comparável e que o modelo consiga "enxergar" a fraude apesar da escassez de exemplos de fraudes.

- **TRATAMENTO DE NULOS:** Confirmar que não existem valores NaN ou vazios. Havendo, decidir se deve-se aplicar uma técnica de imputação ou simplesmente remover as linhas com valores nulos.

- **REMOÇÃO DE DUPLICATAS:** Transações identicas podem causar overfitting, ou seja, o modelo decora o dado em vez de aprender o padrão, portanto, devem ser removidas. 

#### **4.2 ESCALONAMENTO DE ATRIBUTOS (FEATURE SCALING)**

- As variáveis `V1` a `V28` já estão em uma escala similar devido ao PCA. Contudo, as colunas `Time` e `Amount` possuem escalas completamente diferentes (ex: `Amount` pode ir de 0 a 25.000).

- O Random Forest é menos sensível à escala do que modelos lineares, mas o escalonamento ajuda na convergência e na comparação de importância de features.

- Pode-se aplicar **RobustScaler** ou **StandardScaler** apenas nestas duas colunas para que fiquem na mesma "faixa" das variáveis V.

- **RobustScaler** é mais recomendado para dados com muitos outliers.

- **StandardScaler** é mais recomendado para dados com poucos outliers.

#### **4.3 DIVISÃO DE CONJUNTOS (SPLITING)**

- O dataset será dividido em dois conjuntos: **treino** e **teste**.

- Separar os dados antes de qualquer técnica de balanceamento para evitar o **Data Leakage** (vazamento de dados do teste para o treino).

- O conjunto de **treino** será usado para treinar o modelo.

- O conjunto de **teste** será usado para avaliar o desempenho do modelo (separados para não serem usados no treinamento).

 - **Estratificação**: Deve-se usar o parâmetro `stratify=y` para garantir que a proporção de fraudes (minúscula) seja mantida nos conjuntos de treino e teste.

#### **4.4 BALANCEAMENTO DOS DADOS (SAMPLING)**

- Como a classe 1 (Fraude) é minúscula, o Random Forest precisa de ajuda para dar peso às fraudes.

- **Balanceamento dos dados** é uma técnica que visa equilibrar a proporção de fraudes (minúscula) nos conjuntos de treino e teste.

**Opção A: Oversampling (SMOTE - Synthetic Minority Over-sampling Technique)**

- Em vez de apenas duplicar as fraudes existentes (o que causaria overfitting), o SMOTE cria "fraudes novas" artificiais.

 - Por meio de uma fraude real, avalia as fraudes "vizinhas" mais parecidas e cria um ponto intermediário entre elas. É como se ele interpolasse as características para criar uma fraude que "poderia existir".

 - A vantagem é que não há perda de informação (você mantém todos os dados legítimos). A desvantagem é que pode criar dados ruidosos se as fraudes estiverem misturadas com transações legítimas, confundindo o modelo.

**Opção B: Undersampling (Random Undersampling)**

- Você joga fora aleatoriamente a maioria das transações legítimas até ficar com uma quantidade parecida com a de fraudes (ex: 50/50).

- Se você tem 400 fraudes e 200.000 legítimas, você escolhe 400 legítimas aleatórias e descarta as outras 199.600.

- A vantagem é que treinamento fica ultra-rápido e o modelo foca muito em distinguir as classes. A desvantagem é o perigo de jogar fora 99% dos dados legítimos. O modelo pode deixar de aprender padrões importantes das transações legítimas e começar a dar muito "Falso Positivo" (bloquear cartão de cliente bom).

**Opção C: Pesos de Classe (Class Weights)**

- Abordagem "matemática" que não mexe nos dados, isto é, não cria dados artificiais.

- Configura-se um algoritmo de modo que "Se errar uma transação legítima, a penalidade é 1, mas se errar uma fraude, a penalidade é 500". O resultado é que o algoritmo se esforça 500x mais para acertar as fraudes.

- A vantagem é que é computacionalmente eficiente e não altera a distribuição original dos dados. A desvantagem é que pode não ser suficiente se o desbalanceamento for extremo (ex: 1 fraude em 1 milhão).

**Opção D: Métodos Híbridos (SMOTE + Tomek Links ou SMOTE + ENN)**

- Este é o método mais vencedor em competições de Machine Learning.

- Inicialmente aplica-se a técnica de **OverSampling/Smote** para criar fraudes artificiais e equilibrar o jogo. Posteriormente, aplica-se uma técnica de limpeza (Ex: **Tomek Links**) para remover dados que ficaram "na fronteira" confusa entre fraude e não-fraude.

- O resultado é um aumento na quantidade de fraudes, contudo, remove-se a sujeira que o **OverSampling/SMOTE** criou, deixando a separação entre as classes mais limpa para o modelo.

**Opção E: Detecção de Anomalias (Isolation Forest / One-Class SVM)**

- Há uma mudança na forma de pensar, em vez de classificar "A vs B", você treina o modelo apenas com transações normais, ou seja, o modelo aprende perfeitamente o que é o "comportamento normal" de um cliente e, consequentemente, identificará um comportamento que desvia muito desse padrão como anomalia (fraude).

- Este método é aplicado quando há pouquíssimas fraudes (ou nenhuma) para treinar, ou quando os padrões de fraude mudam tão rápido que o modelo supervisionado fica obsoleto.

### 5. TREINAMENTO DO MODELO

- O objetivo desta etapa é fazer com que o computador aprenda a distinguir uma transação legítima de uma fraude usando o Random Forest Classifier.

#### 5.1. **Fluxo do Script `train.py`**:

- **Leitura:** Carrega X_train.csv e y_train.csv.
- **Sintetização:** Aplica o SMOTE para equilibrar as quantidades (50/50).
- **Limpeza:** Aplica Tomek Links para remover as sobreposições.
- **Treino:** O Random Forest é treinado sobre este novo conjunto de dados "limpo e equilibrado".
- **Exportação:** Salva o model.pkl.

#### 5.2. **Random Forest**
- **O que é o Random Forest?**
    - É um algoritmo de aprendizado de máquina do tipo "Ensemble" (Conjunto) que constrói uma "floresta" composta por múltiplas Árvores de Decisão. Cada árvore é treinada com uma parte diferente dos dados e, ao final, todas "votam". A classe que receber mais votos (Fraude ou Legítima) é a decisão final do modelo.
    - Por exemplo, imagine que, em vez de perguntar a opinião de apenas um especialista (uma Árvore de Decisão), você pergunta a 100 especialistas diferentes. Cada um analisa partes diferentes dos dados. No final, eles fazem uma votação: se a maioria disser "Fraude", o modelo classifica como "Fraude". Isto torna o sistema muito mais robusto e menos propenso a erros bobos.
- **Por que usar o Random Forest?**
    - **Robustez (Menos Overfitting):** Enquanto uma única árvore de decisão tende a "decorar" os dados (overfitting), a combinação de muitas árvores reduz esse erro, criando um modelo que generaliza melhor para dados novos.
    - **Captura de Padrões Complexos:** Consegue identificar relações não-lineares entre as variáveis, o que é essencial em fraudes onde o comportamento criminoso não segue uma regra simples.
    - **Importância das Variáveis:** Permite identificar quais colunas (ex: V12, V14, V17, Amount) são as mais decisivas para detectar a fraude, oferecendo uma explicação do porquê o modelo tomou aquela decisão

#### 5.3. **Tratamento do Desbalanceamento (SMOTETomek)**
- **1ª Etapa: SMOTE (Oversampling)**
    - O SMOTE não apenas duplica as fraudes. Ele olha para uma fraude real, identifica seus "vizinhos" e cria uma nova fraude em um ponto aleatório entre eles. Isso ajuda o modelo a aprender a região onde a fraude ocorre, em vez de decorar pontos específicos.
- **2ª Etapa: Tomek Links (Cleaning/Undersampling)**
    - Ao criar dados sintéticos, o SMOTE pode acabar gerando fraudes muito próximas de transações legítimas, criando uma "zona cinzenta" confusa. O Tomek Links identifica pares de pontos de classes diferentes que são os vizinhos mais próximos um do outro e remove o exemplo da classe majoritária (ou ambos). Isso limpa a fronteira de decisão.

- **OBS:** A aplicação destas técnicas melhora a generalização, isto é, o modelo aprende fronteiras mais claras e precisas. 

- **OBS:** Esta técnica foca no Recall, ou seja,  aumenta a presença da classe 1, tornando o Random Forest muito mais sensível a padrões de fraude que antes seriam ignorados como "ruído".

#### 5.4. **O Produto do Treinamento (arquivo .pkl)**
- O resultado do treinamento não é um código, mas um **arquivo binário `model.pkl`**. Este arquivo contém todos os cálculos e caminhos que as 100 árvores aprenderam.

Este arquivo é o **produto** do treinamento e será carregado na API em produção evitando treinar o modelo novamente (demora e consome CPU). Você vai apenas carregar este arquivo e ele dará a resposta instantâneas às consultas.

#### 5.5. **SCRIPT `train.py`**

- A classe `SMOTETomek` verifica o dataset de treino, no qual havia pouquíssimas fraudes, e gera novas amostras baseadas na vizinhança das fraudes reais. Logo após "limpa" o dataset removendo pontos que ficaram muito sobrepostos, deixando a fronteira de decisão mais nítida para o Random Forest.

- O Parâmetro `n_jobs=-1` garante que o computador use todos os núcleos do processador para terminar mais rápido.

- A `class_weight` não foi usada porque após o método SMOTETomek, há uma proporção de 50% de fraudes e 50% de legítimas no dataset, ou seja, o modelo entende naturalmente a importância das duas classes.

#### 5.6. **Configuração de Logs**

- No script `train.py` foram configurados dois tipos de logs, os ***logs de terminal*** e os ***arquivos de log***. Os últimos foram configurados para serem escritos em um arquivo chamado `train.log` salvo na pasta `FraudDetection/logs`.






### 6. **Avaliação do Modelo (Validação do Modelo)**

- 

### 7. **Deploy do Modelo (Produção)**

### 8. **Monitoramento do Modelo (Produção)**

### 9. **Atualização do Modelo (Produção)**


















