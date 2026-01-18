# CREDIT CARD FRAUD DETECTION / Detecção de Fraudes em Cartões de Crédito
DEVELOPED BY: _phmcasimiro_

MBA IN GENERATIVE ARTIFICIAL INTELLIGENCE - PCDF & IBMEC

## Software Engineering applied to Machine Learning | Engenharia de Software aplicada ao Aprendizado de Máquina

## Table of Contents | Sumário

1. [System Architecture Overview | Visão Geral da Arquitetura do Sistema](#system-architecture-overview--visão-geral-da-arquitetura-do-sistema)
2. [Data Pipeline | Pipeline de Dados](#pipeline-de-dados)
3. [Extraction & Ingestion | Extração e Ingestão](#1-extração-e-ingestão-de-dados)
4. [EDA | Análise Exploratória](#2-análise-exploratória-de-dados-eda)
5. [Preprocessing | Pré-processamento](#4-limpeza-e-pré-processamento-dos-dados)
6. [Training & MLOps | Treinamento e MLOps](#5-treinamento-e-mlops)
7. [Evaluation | Avaliação](#6-avaliação-do-modelo-validação-do-modelo)
8. [Infrastructure | Infraestrutura (Docker)](#7-infraestrutura-e-portabilidade-docker)
9. [Deploy | Deploy do Modelo](#8-deploy-do-modelo-produção)

### System Architecture Overview | Visão Geral da Arquitetura do Sistema

**1. User Interface Layer**

- Client/User: Makes HTTP POST requests to /predict endpoint

**1. Camada de Interface do Usuário**

- Cliente/Usuário: Faz requisições HTTP POST para o endpoint /predict

**2. API Layer (FastAPI)**

- FastAPI Application: Handles HTTP requests and responses

- Authentication: API key-based authentication

- Input Validation: Pydantic schemas for data validation


**2. Camada de API (FastAPI)**

- Aplicação FastAPI: Lida com requisições e respostas HTTP

- Autenticação: Autenticação baseada em chave de API

- Validação de Entrada: Esquemas Pydantic para validação de dados

**3. Application Core**

- Prediction Service: Orchestrates the prediction workflow

- Authentication: Secures API endpoints

- Data Validation: Ensures input data quality

**3. Núcleo da Aplicação**

- Serviço de Predição: Orquestra o fluxo de trabalho de predição

- Autenticação: Protege os endpoints da API

- Validação de Dados: Garante a qualidade dos dados de entrada

**4. Machine Learning Engine**

- Model Artifact: Serialized Random Forest model (.pkl file) and MLflow Model Registry.

- Data Preprocessing: Feature engineering and transformation

- Random Forest Model: Trained ML model for predictions

**4. Motor de Aprendizado de Máquina**

- Arquivo do Modelo: Modelo Random Forest serializado `model.pkl` e Registro de Modelos MLflow.

- Pré-processamento de Dados: Engenharia e transformação de recursos

- Modelo Random Forest: Modelo de aprendizado de máquina treinado para predições

**5. Data Pipeline (Offline)**

- **Data Storage:** SQLite database (`data/fraud_detection.db`) for structured storage.

- **Data Processing:** ETL pipeline reads from DB, cleans/transforms, and saves back to DB.

- **Model Training:** Offline training using data from SQLite.

**5. Pipeline de Dados (Offline)**

- **Armazenamento de Dados:** Banco de dados SQLite (`data/fraud_detection.db`) para armazenamento estruturado.

- **Processamento de Dados:** Pipeline ETL lê do banco, limpa/transforma e salva de volta no banco.

- **Treinamento do Modelo:** Treinamento offline usando dados do SQLite.


**6. Request Flow**

- **Request:** _User → POST /predict → FastAPI_

- **Validation:** _API validates input using Pydantic schemas_

- **Authentication:** _API key verification_

- **Prediction:** _Service loads model, preprocesses data, runs inference_

- **Response:** _Prediction result → JSON → User_

**6. Fluxo de Requisição**

- **Requisição:** _Usuário → POST /predict → FastAPI_

- **Validação:** _A API valida a entrada usando esquemas Pydantic_

- **Autenticação:** _Verificação da chave da API_

- **Previsão:** _O serviço carrega o modelo, pré-processa os dados e executa a inferência_

- **Resposta:** _Resultado da previsão → JSON → Usuário_


**7. Key Characteristics** 

- **Separation of Concerns:** _Clear separation between API, business logic, and ML components_

- **Offline Training:** _Model training is separate from serving_

- **Serialized Model:** _Uses pickle files for model persistence_

- **RESTful Design:** _Standard API patterns for ML serving_

**7. Características principais**

- **Separação de responsabilidades:** _Separação clara entre os componentes da API, da lógica de negócios e do aprendizado de máquina_

- **Treinamento offline:** _O treinamento do modelo é separado da sua disponibilização_

- **Modelo serializado:** _Usa arquivos pickle para persistência do modelo_

- **Design RESTful:** _Padrões de API padrão para disponibilização de aprendizado de máquina_

### 1. Data Pipeline & Training Architecture (Offline) | Arquitetura de Pipeline de Dados e Treinamento

This workflow covers the data lifecycle from ingestion to model registration, running inside a Dockerized environment.

Este fluxo de trabalho cobre o ciclo de vida dos dados, desde a ingestão até o registro do modelo, rodando dentro de um ambiente Dockerizado.

```mermaid
graph TD
    subgraph DockerEnv["Docker Environment"]
        
        subgraph DataLayer["Data Layer"]
            RawData[("Raw Data (CSV)")]
            ETL["ETL Process (preprocess.py)"]
            DB[("SQLite Database (fraud_detection.db)")]
        end

        subgraph TrainingLayer["Training & Evaluation"]
            Trainer["Training Script (train.py)"]
            Evaluator["Evaluation Script (evaluate.py)"]
            SMOTE["SMOTETomek (Balancing)"]
            RF["Random Forest Classifier"]
        end

        subgraph MLOpsLayer["MLOps (MLflow)"]
            Tracking["Experiment Tracking (Metrics/Params)"]
            Registry[("Model Registry (Production)")]
            Artifacts["Artifact Store (Plots/Logs)"]
        end
    end

    RawData -->|"Ingest & Clean"| ETL
    ETL -->|"Store Processed Data"| DB
    
    DB -->|"Load Training Data"| Trainer
    Trainer -->|"Apply Balancing"| SMOTE
    SMOTE -->|"Train"| RF
    
    RF -->|"Log Metrics & Model"| Tracking
    RF -->|"Register Version"| Registry
    
    DB -->|"Load Test Data"| Evaluator
    Evaluator -->|"Generate Metrics"| Tracking
    Evaluator -->|"Save Plots"| Artifacts
    
    style DockerEnv fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
```

### 2. Deployment & Inference Architecture (Online) | Arquitetura de Deploy e Inferência

This workflow illustrates how the API serves predictions, abstracting the training complexity.

Este fluxo de trabalho ilustra como a API fornece predições, abstraindo a complexidade do treinamento.

```mermaid
graph TD
    subgraph ClientSide["Client Side"]
        User["External System / User"]
    end

    subgraph ServerSide["Server Side (Docker Container)"]
        subgraph APIGateway["API Gateway (FastAPI)"]
            EntryPoint["POST /predict"]
            Auth["Auth Middleware (API Key)"]
            Validator["Pydantic Validator (Schema)"]
        end

        subgraph ServiceLayer["Core Services"]
            Predictor["FraudPredictor Service (Singleton)"]
            Logic["Business Logic (Thresholds)"]
        end
        
        subgraph ModelSource["Model Source"]
            Registry[("MLflow Registry (Production)")]
            Fallback["Local Fallback (.pkl)"]
        end
    end

    User -->|"1. Request (JSON)"| EntryPoint
    EntryPoint -->|"2. Validate Data"| Validator
    Validator -->|"3. Authenticate"| Auth
    
    Auth -->|"4. Forward Valid Request"| Predictor
    
    Predictor -->|"5. Load Model (On Startup)"| Registry
    Registry -.->|"If Connection Fails"| Fallback
    
    Predictor -->|"6. Inference"| Logic
    Logic -->|"7. Return Result (Class + Prob)"| EntryPoint
    EntryPoint -->|"8. Response (JSON)"| User
    
    style ServerSide fill:#4a148c,stroke:#6a1b9a,stroke-width:2px,color:#fff
```

---------------


### 1. Extração e Ingestão de Dados

#### 1.1 Extração e Ingestão de Dados
- O Script `download_data.py` baixa o dataset mais atualizado .

`script download_data.py`


### 2. Análise Exploratória de Dados (EDA)
- Antes de começar a codificar a API, é necessário entender os dados. 
- Verificar a correlação das variáveis `V1` a `V28` e a distribuição da variável alvo `Class`.

- O script `src/data/eda.py` implementa uma análise exploratória de dados (EDA).

- **OBS:** Gráficos para visualização foram feitos com as bibliotecas `seaborn` e `matplotlib` e salvos na pasta `artifacts/`.


#### **3.1 Distribuição da Variável Alvo `Class`:**

<p align="center">
  <img src="artifacts/distribuicao_classe.png" alt="Distribuição de Classes" width="600">
</p>

- A variável `Class` possui dois valores: **0** (legítima) e **1** (fraude).

- Antes de treinar o modelo é essencial verificar o balanceamento/desbalanceamento das variáveis, então, usaremos um gráfico de barras (Distribuição de Classes) com este fim.

- Em fraudes, a classe "1" (fraude) costuma ser uma fração mínima, o que se traduz em uma barra enorme no 0 e uma quase invisível no 1.

- Em Machine Learning, isso significa que o dataset possui **Classes Desbalanceadas**. Caso o modelo seja treinado com dados severamente desbalanceados (menos de 0.2% dos dados são fraudes) aprenderá que "quase sempre não é fraude" e ignorará as fraudes reais.
    
- Explicando de outra forma, se o desbalanceamento não for tratado, o modelo de Random Forest aprenderá a sempre classificar como "0" (Transação Legítima), pois ele terá 99.8% de acurácia fazendo isso, mesmo falhando em detectar todas as fraudes (0.2%).
   
- **Importante:** Neste projeto focaremos na métrica **Recall** _(detalhes em: docs/Teoria&Conceitos.md)_ a fim de não ignorar nenhuma fraude, mesmo que isso gere alguns alarmes falsos (Falsos Positivos).


#### **3.2 Correlação das Variáveis `V1` a `V28`**

<p align="center">
  <img src="artifacts/matriz_correlacao.png" alt="Correlação das Variáveis" width="600">
</p>

- Neste dataset as variáveis `V1` a `V28` são resultado de um **PCA** (Principal Component Analysis), que transforma variáveis originais em novos componentes que são **independentes** entre si _(detalhes em: docs/Teoria&Conceitos.md)_.

- Logo, se fizermos uma matriz de correlação entre as variáveis, a correlação entre elas será próxima de zero (0).

- O Foco da Análise é a correlação das variáveis com a variável `Class/Fraude`, ou seja, descobrir quais variáveis `V` têm mais poder preditivo (influência) para determinar fraudes.

- O gráfico utiliza uma **escala de cores** para demonstrar a **força da relação entre as variáveis**.

- **Vermelho Forte** indica uma **correlação positiva perfeita (valor 1)**, isto é, quando uma variável aumenta, a outra também aumenta. Por isso há uma **linha vermelha na diagonal principal**, a qual representa a **correlação de uma variável consigo mesma**.

- **Azul Forte** indica uma **correlação negativa forte (valor -1)**, isto é, quando uma variável aumenta, a outra tende a diminuir.

- **Cores Claras/Neutras** Indicam **correlações fracas ou nulas (valor 0)**, ou seja, as variáveis apresentam comportamento independente entre si.

- O **Centro do gráfico** é marcado por cores neutras e azul clara, isto é, as variáveis não possuem correlação entre si em razão da técnica PCA aplicada ao conjunto de dados. Em outras palavras, as variáveis `V1` a `V28` têm correlação zero entre si (as células do mapa de calor fora da diagonal são neutras em sua maioria). A razão desse fenômeno é a aplicação da técnica **PCA** (Análise de Componentes Principais), uma técnica de engenharia de features que transforma dados correlacionados em componentes independentes _(detalhes em: docs/Teoria&Conceitos.md)_.

- A análise de variância é uma técnica estatística que permite avaliar a variação de uma variável em relação a outra variável. Em outras palavras, verifica-se quais variáveis `V1` a `V28` possuem maior variação quando a `Class/Fraude` é `1` e quando é `0`. As variáveis identificadas serão as mais importantes para o modelo de Random Forest.

- Ao analisar a última linha (ou coluna) da matriz, que mostra a correlação com a `Class/Fraude`, verificamos que as variáveis `V17`, `V14` e `V12` apresentam correlações negativas relativamente fortes com a `Class/Fraude`. 

### **4. LIMPEZA E PRÉ-PROCESSAMENTO DOS DADOS**

- A partir da identificação das variáveis críticas (V12, V14 e V17) e do desbalanceamento do dataset, a **limpeza e pré-processamento dos dados** garantem que o modelo de Random Forest seja treinado com dados de qualidade, evitando que o modelo seja enganado pelo ruído dos dados e que ele aprenda a classificar como "0" (Transação Legítima) e como "1" (Fraudes), mesmo que falhe em detectar todas as fraudes. 

#### **4.1 VERIFICAÇÃO DE INTEGRIDADE (DATA CLEANING)**

- O objetivo é garantir que todas as entradas estejam em uma escala comparável e que o modelo consiga "enxergar" a fraude apesar da escassez de exemplos de fraudes.

- **TRATAMENTO DE NULOS:** Confirmar que não existem valores NaN ou vazios. Havendo, decidir se aplicaremos uma técnica de imputação ou simplesmente removeremos as linhas com valores nulos.

- **REMOÇÃO DE DUPLICATAS:** Transações identicas podem causar overfitting, ou seja, o modelo decora o dado em vez de aprender o padrão, portanto, devem ser removidas. 

#### **4.1.1 VALIDAÇÃO DE INTEGRIDADE DE DADOS (PANDERA)**

- Visando garantir um pipeline de dados integro e evitar o processamento de dados corrompidos, foi implementada uma camada de validação (schema) usando a biblioteca **Pandera**.

- Se, no futuro, o Kaggle mudar o formato do arquivo (_creditcard.csv_) ou alguma coluna seja disponibilizada como texto em vez de número, isto é, houver qualquer tipo de mudança de formato ou tipo de dados, sem o Pandera, o script falharia de forma silenciosa ou poderia haver um erro matemático no treino.

- O uso do Pandera cria um "vigilante" que verifica o DataFrame, avalia e alerta alterações no dataset, garantindo a integridade estatística do modelo. O Pandera funciona como um contrato de dados, um schema, um molde ao qual o dado que entra no pipeline deve adequar seu formato e tipo. Caso contrário, o Pandera interrompe o processo imediatamente, evitando que o modelo aprenda padrões errados de dados inadequados (Evita o "Garbage In, Garbage Out").

- Foi definido um contrato/schema rígido `src/data/pandera_schemas.py` que verifica:
    - **Tipagem:** Garante que colunas V1 a V28, Time e Amount sejam sempre `float`.
    - **Regras de Negócio:** Verifica se a coluna `Amount` possui valores negativos (inválidos para transações).
    - **Consistência do Alvo:** Garante que a coluna `Class` contenha apenas os valores 0 ou 1.
    - **Contrato de Interface:** O strict=True garante que o modelo sempre receba as mesmas 30 variáveis de entrada, evitando erros de dimensão no futuro.

- Assim o pipeline segue o princípio de "Falha Rápida" (Fail-Fast), interrompendo o processo imediatamente caso o contrato de dados (schema) seja violado.


#### **4.2 ESCALONAMENTO DE ATRIBUTOS (FEATURE SCALING)**

- As variáveis `V1` a `V28` já estão em uma escala similar devido ao PCA. Contudo, as colunas `Time` e `Amount` possuem escalas completamente distintas (ex: `Amount` pode ir de 0 a 25.000).

- O Random Forest é menos sensível à escala do que modelos lineares, mas o escalonamento ajuda na convergência e na comparação de importância de features.

- Pode-se aplicar **RobustScaler** ou **StandardScaler** apenas nestas duas colunas para que fiquem na mesma "faixa" das variáveis V.

- **RobustScaler** é mais recomendado para dados com muitos outliers.

- **StandardScaler** é mais recomendado para dados com poucos outliers.

- Neste projeto foi aplicado o RobustScaler no script `src/data/preprocess.py`

#### **4.3 DIVISÃO DE CONJUNTOS (SPLITING)**

- O dataset foi dividido em dois conjuntos: **treino** e **teste** antes de qualquer técnica de balanceamento a fim de evitar o **Data Leakage** (vazamento de dados do teste para o treino).

- O conjunto de **treino** foi usado para treinar o modelo.

- O conjunto de **teste** foi usado para avaliar o desempenho do modelo.

- **Estratificação**: Foi usado o parâmetro `stratify=y` para garantir que a proporção de fraudes (minúscula) seja mantida nos conjuntos de treino e teste.

### 5. TREINAMENTO & MLOPS

O objetivo desta etapa é treinar o modelo para distinguir transações legítimas de fraudes e garantir o gerenciamento profissional do ciclo de vida desse modelo.

#### 5.1 **Fluxo do Pipeline de Treinamento**

O script `src/models/train.py` executa os seguintes passos:

1.  **Carregamento:** Leitura de `X_train.csv` e `y_train.csv`.
2.  **Balanceamento:** Aplicação de SMOTE (Geração de dados sintéticos).
3.  **Limpeza:** Aplicação de Tomek Links (Refinamento de fronteiras).
4.  **Treinamento:** Ajuste do Random Forest aos dados balanceados.
5.  **Registro:** Salvamento dos artefatos e métricas no MLflow.

#### 5.2 **Técnicas de Modelagem**

##### **A) Balanceamento de Dados (SMOTETomek)**
Como a classe de fraude é minoritária, foi utilizado um método híbrido:
-   **SMOTE (Oversampling):** Cria fraudes sintéticas interpolando exemplos existentes, ajudando o modelo a aprender a "região" da fraude.
-   **Tomek Links (Undersampling):** Remove exemplos da classe majoritária que estão muito próximos da classe minoritária, limpando a fronteira de decisão e reduzindo ruído.

##### **B) Algoritmo Random Forest**
Foi utilizado um **Ensemble** de **Árvores de Decisão**:
-   **Funcionamento:** Centenas de árvores "votam" na classificação. A maioria vence.
-   **Vantagens:** Alta robustez contra overfitting e capacidade de capturar padrões não-lineares complexos, típicos de fraudes.

#### 5.3 **MLOps com MLflow**

O MLflow foi implementado para elevar o nível de maturidade do projeto, garantindo rastreabilidade e governança.

##### **5.3.1 Arquitetura Implementada**
-   **Backend Store:** SQLite (`mlflow.db`) para metadados (rápido e leve).
-   **Artifact Store:** Diretório local (`mlruns/`) para modelos e gráficos.
-   **Model Registry:** Gerenciamento centralizado de versões de modelos.

##### **5.3.2 O Produto do Treinamento**

O resultado do treinamento gera dois artefatos principais:

1.  **Arquivo Binário (`model.pkl`)**:
    -   É o formato tradicional (legacy). Contém todos os cálculos e caminhos que as 100 árvores aprenderam serializados.
    -   Serve como **backup/fallback** caso o MLflow esteja indisponível.

2.  **Modelo Registrado no MLflow**:
    -   O modelo é salvo no formato padrão do MLflow e registrado no **Model Registry** com versionamento automático (ex: `FraudDetectionRandomForest/Version 1`).
    -   É o método **principal** de carregamento em produção, garantindo que a API sempre use a versão correta e aprovada.

Ambos contêm a mesma inteligência (as árvores de decisão treinadas), mas o registro no MLflow oferece governança e facilidade de deploy.

##### **5.3.2 Alteração de Modelo (Baseline vs Produção)**
Para garantir segurança na migração para MLOps:

1.  **Baseline (Legacy):** O antigo `model.pkl` foi registrado como `FraudDetectionBaseline` (v1).
2.  **Produção (Novo):** O novo modelo treinado foi registrado como `FraudDetectionRandomForest` e promovido para **Production**.
3.  **API Híbrida:** O `predictor.py` tenta carregar do MLflow Registry, mas se falhar usa o `model.pkl` local como fallback.

##### **5.3.3 Validação da Alteração dos Modelos**
Comparamos o modelo legado (`model.pkl`) com o novo modelo registrado `FraudDetectionRandomForest` no MLflow para garantir consistência:

| Métrica | Produção (.pkl) | MLflow Registry | Diferença |
| :--- | :--- | :--- | :--- |
| **Recall (Fraude)** | 0.8105 | 0.8105 | +0.0000 |
| **Precision (Fraude)** | 0.6063 | 0.6063 | +0.0000 |
| **F1-Score (Fraude)** | 0.6937 | 0.6937 | +0.0000 |
| **AUC-ROC** | 0.9786 | 0.9786 | +0.0000 |

**Conclusão:** Os modelos são idênticos. A migração foi bem-sucedida e segura.

### 6. **Avaliação e Validação do Modelo em Produção**

- Nesta etapa, o objetivo é avaliar o desempenho do modelo treinado com dados que não foram usados no treinamento (x_teste, y_teste).

#### **6.1 TAREFAS/ETAPAS (Automatizadas no `evaluate.py`)**

1.  **Execução do Script de Avaliação**
    -   O script `src/models/evaluate.py` é executado, carregando o modelo (do MLflow ou local) e os dados de teste.
    -   Todo o processo é rastreado automaticamente pelo MLflow.

2.  **Cálculo de Métricas**
    -   São calculadas as métricas: **Precision**, **Recall**, **F1-Score**, **Acurácia** e **AUC-ROC**.
    -   Os valores são registrados no MLflow para comparação histórica.

3.  **Geração de Artefatos Visuais** 
    
    -   **Matriz de Confusão:** Salva como imagem e registrada no MLflow.
    -   **Curva ROC:** Gráfico da performance do classificador, também salvo e registrado.
    -   **Relatório Técnico:** Arquivo de texto com o detalhamento das métricas. 
    - **Armazenamento:** `artifacts/evaluation`

#### **6.2 CONSIDERAÇÕES SOBRE AVALIAÇÃO DE FRAUDES**

**LEGENDA:** 

`VN = Verdadeiro Negativo`

`FP = Falso Positivo`

`FN = Falso Negativo`

`VP = Verdadeiro Positivo`

- **ACCURACY**: 

    - Fórmula: `(VN+VP)/(VN+VP+FP+FN)`

    - Percentual de acertos totais (tanto de fraudes quanto de legítimas).

    - Não é a melhor métrica para avaliar modelos construídos a partir de dados desbalanceados porque pode levar a um alto número de falsos negativos.

    - Por exemplo, se 99,8% das transações são legítimas, um modelo mal treinado e mal avaliado dirá que "NUNCA É FRAUDE" mesmo com 99,8% de acurácia, ou seja, terá um alto número de falsos negativos, isto é, o modelo erra em detectar fraudes.

- **PRECISION** 

    - Fórmula: `VP/(VP+FP)`

    - Responde à pergunta: "De todas as vezes que o modelo deu alerta de fraude, quantas eram fraudes reais?".

    - O objetivo desta métrica é reduzir Falsos Positivos, ou seja, evitar que o banco bloqueie o cartão de um cliente honesto por engano.

    - Exemplo: Uma precisão de 0.90 significa que, a cada 10 alertas de bloqueio, 9 eram realmente fraudes e 1 foi um alarme falso.    
    - Importância: Fundamental para medir a "irritação" gerada ao cliente legítimo.

- **RECALL**

    - Fórmula: `VP/(VP+FN)`

    - Responde à pergunta: "De todas as fraudes que realmente ocorreram, quantas o modelo conseguiu detectar?".

    - O objetivo desta métrica é reduzir Falsos Negativos, ou seja, evitar que o dinheiro do banco/cliente seja roubado.

    - Exemplo: Um recall de 0.90 significa que, de cada 10 fraudes que aconteceram, o modelo detectou 9 e deixou passar 1.

    - Importância: É a métrica de segurança. O aumento do Recall tende a baixar a Precision (mais bloqueios preventivos), exigindo um ponto de equilíbrio.

- **F1-SCORE** 

    - Fórmula: `2*(Precision * Recall)/(Precision + Recall)`

    - Média Harmônica entre Precision e Recall

    - Busca equilibrar Precision e Recall. Se o Recall for excelente (1.0) mas a Precision for péssima (0.1), o F1-Score será baixo. 

    - É a métrica única mais utilizada para comparar modelos em datasets desbalanceados.

- **AUC-ROC** 

    - Fórmula: `Área sob a curva ROC`

    - A curva ROC plota a "Taxa de Verdadeiros Positivos" contra a "Taxa de Falsos Positivos" para diferentes limiares de decisão do modelo.

    - Sabendo que a área sob a Curva ROC varia entre 0 e 1, um valor de 1 indica um modelo perfeito (todas as fraudes são detectadas e nenhum legítimo é bloqueado), um valor de 0.5 indica um modelo aleatório e um valor de 0 indica um modelo completamente incapaz de distinguir fraude de transação legítima, chegando a inverter as predições.

    - Considerando o Modelo de Detecção de Fraude deste projeto, uma AUC-ROC de 0.95 indicaria que o modelo tem uma capacidade muito alta de distinguir o que é fraude do que é legítimo, independentemente da proporção de classes. Já um AUC-ROC de 0.5 indicaria um modelo aleatório.


**IMPORTANTE:** RECALL é considerado uma métrica primária de segurança (pois queremos pegar o ladrão), enquanto a PRECISION é a métrica de "qualidade de atendimento" (não irritar o cliente). 


#### RESULTADOS

- **RELATÓRIO DE MÉTRICAS (Classe 1 Fraude):** 
    - **Recall (0.81):** Significa que de cada 100 fraudes reais, seu modelo detectou 81. Considerando que este é um primeiro modelo, construído com dados desbalanceados, esse resultado atingiu a expectativa.

    - **Precision (0.61):** Significa que quando o modelo diz "fraude", ele está certo em 61% das vezes. Os outros 39% são falsos positivos. Para bancos, este é um percentual aceitável, pois é preferível verificar uma transação legítima do que deixar uma fraude passar. Contudo, há de ser melhorado porque cada transação legítima bloqueada é um incômodo para o cliente.

    - **F1-Score (0.69):** É o equilíbrio entre os dois anteriores. O valor alcançado, próximo de 0.70, é um ponto de partida aceitável para um sistema de detecção de anomalias que utiliza um dataset muito desbalanceado.

    - **AUC-ROC (0.9786):** Este é o melhor resultado entre as métricas utilizadas, porque indica que o modelo possui uma capacidade excelente de separar as classes. Analisando comparativamente com os erros de Precision, Recall e F1-Score, estes últimos podem ser melhorados se ajustado o limiar (threshold) de decisão sobre a classificação de fraude.

- **MATRIZ DE CONFUSÃO:** 
    
    - A Matriz de Confusão permite visualizar onde o modelo acertou e onde e como ele errou.

    - **Verdadeiros Negativos (Superior Esquerdo - 56601):** 
    - Representa as transações legítimas que o modelo classificou corretamente como legítimas. Como esperado, é o maior volume de dados.

    - **Falsos Positivos (Superior Direito - 50):** 
    - Estes são os "Alarmes Falsos". O modelo previu fraude (1), mas a transação era legítima (0). No seu caso, são apenas 50 casos em mais de 56 mil, o que indica uma alta Precisão.

    - **Falsos Negativos (Inferior Esquerdo - 18):** 
    - São 18 fraudes que ocorreram na vida real, mas o modelo classificou como transação normal. O objetivo é reduzir esse número ao máximo para evitar prejuízo financeiro.

    - **Verdadeiros Positivos (Inferior Direito - 77):** 
    - São fraudes reais que o modelo conseguiu detectar com sucesso.

    - **Conclusão da Matriz:** O modelo conseguiu capturar a grande maioria das fraudes (77 de 95 totais no teste), mantendo um número muito baixo de clientes legítimos incomodados por alarmes falsos.

<p align="center">
  <img src="artifacts/evaluation/confusion_matrix_20260111_121048.png" alt="Matriz de Confusão" width="600">
</p>

- **CURVA ROC:**

    - A Curva ROC (Receiver Operating Characteristic) mostra a capacidade do modelo de distinguir entre as duas classes (Fraude vs. Legítima) conforme alteramos o "limiar" de decisão.

    - **Linha Azul (Modelo):** Quanto mais próxima ela estiver do canto superior esquerdo, melhor. Note que ela sobe quase verticalmente no início, o que indica que o modelo consegue identificar muitos Verdadeiros Positivos (Recall) sem aumentar drasticamente os Falsos Positivos.

    - **Linha Tracejada Preta:** Representa um modelo puramente aleatório (como jogar uma moeda). Como a linha azul está muito acima, significa que o modelo é muito superior ao acaso.

    - **Valor AUC = 0.9786:** A área sob a curva (AUC) é de quase 98%. Significa que se analisarmos, aleatoriamente, uma transação fraudulenta e uma legítima, há 97,86% de chance do modelo atribuir uma pontuação de risco maior para a fraude.

<p align="center">
  <img src="artifacts/evaluation/roc_curve_20260111_121048.png" alt="Curva ROC" width="600">
</p>

### 7. **INFRAESTRUTURA E PORTABILIDADE (DOCKER)**

Para garantir que o projeto seja executado de forma idêntica em qualquer ambiente (desenvolvimento, teste ou produção), foi implementada a conteinerização utilizando **Docker**, isto é, foi criada uma imagem do projeto que pode ser executada em qualquer ambiente que suporte Docker, eliminando o problema do _"na minha máquina funciona"_ e isolando todas as dependências e configurações.

**Por que o Docker nesta etapa do projeto?**

Considerando o Fluxo de MLOps, começamos pela **Experimentação**, que inclui experimentação, construção e avaliação do modelo de ML (Etapas 1 a 6). Seguimos para o Empacotamento (Etapa 7) e, por fim, para a Disponibilização como Serviço (Etapa 8).

**IMPLEMENTAÇÃO DO DOCKER:**

- **Imagem Base:** Utilização do `python:3.12-slim`.

- **Isolamento de Ambiente:** Gerenciamento automático de dependências via `requirements.txt`.

- **Configuração de PYTHONPATH:** Padronização dos caminhos de importação, resolvendo conflitos de módulos entre scripts.

- **Reprodutibilidade:** Garantia de que o pré-processamento, treino e avaliação ocorram sob as mesmas versões de bibliotecas (Pandas, Scikit-Learn, Pandera, etc).

- **CRIAR ARQUIVO `.dockerignore`**

```bash
venv/
__pycache__/
*.pyc
.git/
.env
logs/*.log
```

- **CRIAR ARQUIVO `dockerfile`**

```bash
# Dockerfile
# Criado por phmcasimiro
# Data: 2026-01-09

# 1. Imagem base oficial do Python
FROM python:3.12-slim

# 2. Diretório de trabalho
WORKDIR /app

# 3. Configurações de ambiente
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# 4. Instalar ferramentas de compilação (necessárias para o imbalanced-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Instalar dependências (aproveitando o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar o código do projeto
COPY . .

# 7. Comando padrão (abre um terminal)
CMD ["bash"]
```

- **EXECUTAR O COMANDO `docker build`**

```bash
docker build -t fraud-detection-app .
```

- **EXECUTAR O COMANDO `docker run`**

```bash
docker run -it fraud-detection-app
```

- Ao executar o comando acima, inicia-se um container baseado na imagem fraud-detection-app construída.

- Ao verificar o terminal, verá que o container foi iniciado e está rodando. A linha de comando apresentará: **root@567e69a3149f:/app#**

- **CRIAR ARQUIVO `docker-compose.yml`**

```bash
# docker-compose.yml
# Criado por phmcasimiro
# Data: 2026-01-09

version: '3.8'

services:
  app:
    build: . # Indica ao Compose para usar o Dockerfile da pasta atual.

    container_name: fraud_detection_container # Nome do container.

    # Sincronização em tempo real (Volumes)
    volumes:
      - .:/app # O ponto (.) é a máquina local, e /app é o container. Agora, qualquer vírgula atualizada no VS Code/Editor será refletida instantaneamente dentro do container Docker.

    stdin_open: true # Mantém o container aberto para comandos interativos

    tty: true # Mantém o container aberto para comandos interativos

    environment:
      - PYTHONPATH=/app
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
```


### 7.1 **Considerações sobre o Docker:**
 
- O projeto roda dentro de containers Docker para garantir que funcione igual em qualquer máquina.
- Utilizamos o **Docker Compose** para orquestrar a aplicação e criar um volume (espelho) entre sua pasta local e o container, permitindo desenvolvimento em tempo real.

- Para um guia completo de comandos, gerenciamento e resolução de problemas (troubleshooting), consulte o nosso **[Guia de Docker](docs/docker.md)**.

### 7.2 **Comandos Rápidos**

- **Iniciar:** `docker-compose up --build -d`
- **Parar:** `docker-compose down`
- **Logs:** `docker-compose logs -f app`
- **Status:** `docker ps`

### 8. **Deploy do Modelo (Produção)**

- No ponto de **Deploy** há uma transição do trabalho de um Cientista de Dados para um Engenheiro de Machine Learning. Nesta etapa, transformaremos o modelo estático `model.pkl` em um serviço web acessível (API), ou seja, colocaremos o modelo de **desenvolvimento** em **produção**, permitindo que sistemas externos (bancos) enviem transações e recebam previsões de fraude em tempo real.

- Sem o Deploy, o modelo de ML é apenas um arquivo inútil. Com o Deploy, o modelo se torna um serviço web acessível via API.

#### **8.1 Panorama Geral do Deploy**

**1 - Contrato/Schemas Pydantic**
    - Arquivo `src/api/schemas.py` garante que a API só receba dados no formato esperado, evitando erros devido a dados inválidos. 

**2 - Services**
    - Arquivo `src/models/predictor.py` implementa a lógica de negócio (predição de fraude). Este script é uma interface entre as chamadas da internet na API e o modelo Estatístico de machine learning.

**3 - FastAPI**
    - Arquivo `src/api/main.py` implementa a API utilizando o framework **FastAPI**. É o servidor que fica "ouvindo" a internet. Quando recebe uma requisição de predição, ele valida com o Schema e chama a função de predição.

**4 - Servidor de Aplicação (Uvicorn)**
    - O FastAPI cria a lógica, mas o **Uvicorn** é o servidor ASGI de alta performance que gerencia as conexões de rede, processos paralelos (workers) e garante que a API aguente milhares de requisições simultâneas.

**5 - Docker**
    - Atualizar a `Dockerfile` e `docker-compose.yml` para expor a porta da API (ex: 8000).
    - Garantir que o serviço suba automaticamente pronto para receber requisições.

**6 - CI/CD (Testes Automatizados)**
    - Implementar testes unitários e de integração com **pytest**.
    - Garantir que a API responda corretamente a dados válidos e trate erros para dados inválidos (ex: valores negativos) antes de adicionar camadas de segurança.

**7 - Segurança e Autenticação**
    - Implementação de barreiras de segurança (como API Keys ou OAuth2) para garantir que apenas clientes autorizados (ex: o sistema do banco) possam enviar transações para análise, protegendo o modelo contra uso indevido.

#### **8.2 Exemplo API FraudDetection**

- Se um sistema bancário enviar um JSON para a API:

```json
{ "Time": 1024, "V1": -1.2, ..., "Amount": 5000.0 }
```

- A API valida os dados com o Schema, chama a função de predição e retorna um JSON com o resultado:

```json
{ "is_fraud": 1, "probability": 0.98, "status": "ALTO RISCO" }
```

#### **8.3 - Schemas (Pydantic)**

- O Pydantic é uma biblioteca que ajuda a validar e converter dados em Python. Ele é comumente usado para validar dados de entrada e saída em APIs.

- O arquivo `src/api/schemas.py` define os contratos de entrada e saída da API. Em APIs, o cliente e o servidor precisam concordar exatamente com o formato dos dados ou nada funciona.

- ***TransactionInput***: Define o formato de entrada da API.
- ***PredictionOutput***: Define o formato de saída da API.
    - Herança `BaseModel`: Ao herdar de BaseModel, o Pydantic automaticamente transforma o JSON que chega na API em um objeto Python e valida os tipos de dados.
    - `Field`: Ajuda a documentar os dados e fornece exemplos.
    - `...`: Indica que o campo é obrigatório.
    - `description`: Ajuda a documentar os dados e fornece exemplos.
    - `example`: Ajuda a documentar os dados e fornece exemplos.
    - `ge`: Indica que o campo deve ser maior ou igual a um valor.
    - `nullable`: Indica que o campo pode ser nulo.

##### **Resumo do Fluxo:**

```mermaid
graph LR
    Step1["1. Sistema Externo\n(Envia JSON)"]
    Step2["2. Validação Pydantic\n(Verifica Regras)"]
    Step3["3. Predictor Service\n(Calcula Risco)"]
    Step4["4. Resposta API\n(JSON Final)"]

    Step1 -->|"Transação"| Step2
    Step2 -->|"Dados OK"| Step3
    Step3 -->|"Resultado"| Step4
    
    style Step1 fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
    style Step2 fill:#4a148c,stroke:#6a1b9a,stroke-width:2px,color:#fff
    style Step3 fill:#4a148c,stroke:#6a1b9a,stroke-width:2px,color:#fff
    style Step4 fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
```

#### **8.4 Serviço de Predição**

- O script `src/models/predictor.py` contém a lógica de predição. Seu objetivo é carregar o modelo de detecção de fraude de forma inteligente (priorizando o MLflow Registry) e oferecer uma função que recebe os dados da API e devolve a classificação.

##### **8.4.1 Considerações sobre o script de Predição**

- **Carregamento Híbrido (MLflow + Fallback)**:
    - O serviço foi projetado para buscar sempre a versão de **Produção** do modelo diretamente do **MLflow Model Registry**. Isso garante que a API esteja sempre usando a versão mais aprovada e auditada.
    - Caso o MLflow esteja indisponível, ele executa um **Fallback** automático, carregando o modelo `model.pkl` local (serializado via `joblib`).
    - Todo esse processo ocorre no `__init__`, garantindo que o modelo seja carregado apenas uma vez na memória RAM (Singleton), evitando latência a cada nova transação.

- **model_dump()**: Esse método do Pydantic transforma o objeto JSON que a API recebeu em um dicionário Python puro, que o Pandas aceita facilmente para criar o DataFrame.

- **Probabilidade vs Classe**: O modelo de detecção de fraude (Random Forest) não classifica apenas em 0 ou 1. Ele gera um percentual de probabilidade (ex: _"92% de chance de fraude"_). Para sistemas financeiros, esse score de risco é mais valioso para tomada de decisão (bloqueio automático vs análise humana) do que a classificação binária simples.


#### **8.5 API (FastAPI)**

- **O que é uma API?**
    - API (Application Programming Interface) é um conjunto de regras que permite que diferentes softwares se comuniquem. No caso desta API, é a interface que viabiliza um site ou aplicativo enviar dados de uma transação e receber resposta se a transação é fraude ou legítima, sem precisar conhecer o código complexo do modelo.

- **O que é o FastAPI?**
    - É um framework moderno e de alta performance para construção de APIs em Python. É extremamente rápido (comparável a NodeJS e Go), fácil de usar e gera automaticamente a documentação interativa (Swagger UI), o que facilita os testes.

- **O que é o Uvicorn?**
    - Uvicorn é um servidor ASGI (Asynchronous Server Gateway Interface) para aplicações FastAPI. Ele é responsável por receber as requisições HTTP e enviar as respostas para o cliente.

- **Implementação da API (FastAPI) e do Servidor de Aplicações (Uvicorn)?**
    - A API (FastAPI) foi configurada no arquivo `src/api/main.py`.
    - O Servidor de Aplicação (Uvicorn) foi configurado no arquivo `src/api/main.py`, mas iniciado dentro do container Docker no arquivo `docker-compose.yml` com Hot-Reload ativado.

- **Como subir a API?**
    - Com o `docker-compose.yml` configurado, basta rodar:
    ```bash
    docker-compose up --build -d
    ```
    - Isso constrói a imagem e inicia o servidor Uvicorn na porta 8000.

- **Visualizando Logs:**
    - Para ver o que está acontecendo (ex: "Recebendo nova transação..."), use:
    ```bash
    docker-compose logs -f app
    ```
    - O `-f` (follow) faz com que os logs apareçam em tempo real.

- **Testar o Reload Automático (Hot-Reload):**
    - O Docker foi configurado com um "espelho" (Volume). Caso o arquivo `src/api/main.py` seja editado e salvo, por exemplo, com uma alteração da mensagem de status de `"online"` para `"rodando"` (linha 27), o terminal de logs mostrará que o Uvicorn detecta a mudança e reinicia a API automaticamente, mostrando o log no terminal.

- **Sair dos Logs:**
    - Pressione `Ctrl + C`. 
    - Para de exibir os logs, mas o container continua rodando em segundo plano.

- **Testar com Swagger UI:**
    1.  Acesse `http://localhost:8000/docs` no seu navegador.
    2.  Você verá a documentação automática.
    3.  Clique no endpoint `POST /predict` -> `Try it out`.
    4.  Edite o JSON de exemplo e clique em `Execute`.
    5.  A API processará os dados e retornará a previsão (fraude ou legítima) logo abaixo. 


#### **8.6 CI/CD (Testes Automatizados)**

- **Panorama Geral sobre CI/CD**
    - De acordo com as boas práticas de desenvolvimento atuais, não basta apenas escrever código, é necessário garantir que o software esteja funcionando a cada edição, na sua conclusão e em cada alteração. O CI/CD (Continuous Integration/Continuous Delivery) é a esteira automática que valida e entrega do código.
    - No contexto de Machine Learning (MLOps), isso se torna ainda mais crítico porque há três variáveis que podem sofrer mudanças: Código, Dados e Modelo.
- **CI (Continuous Integration - Integração Contínua):** É a prática de integrar mudanças de código frequentemente. A cada "Save" ou "Push" para o repositório, um robô roda todos os testes automaticamente. Se algo não funcionar como planejado, o desenvolvedor é recebe um warning na hora.
- **CD (Continuous Delivery/Deployment - Entrega Contínua):** É a automação do deploy. Se os testes do CI passarem, o sistema atualiza a API em produção automaticamente, sem intervenção humana.
- **Testes Automatizados:** São scripts que "fingem" ser um usuário para verificar se o sistema está funcionando.
    - *Unitários:* Testam uma função isolada (ex: "A função de soma retorna 4 para 2+2?").
    - *Integração:* Testam se as partes conversam bem (ex: "A API recebe o JSON e o Modelo devolve a predição?").

- **A Importância para Engenharia de Software em ML**
    1.  **Confiabilidade:** Evita que um erro bobo de código (bug) derrube o sistema de detecção de fraudes em produção, o que poderia causar prejuízo financeiro real.
    2.  **Agilidade:** Permite que o Cientista de Dados melhore o modelo e coloque em produção em minutos, com segurança, ou seja, sem intervenção humana.
    3.  **Documentação Viva:** Os testes devem simular exatamente como o sistema deve se comportar. Se o teste espera receber um `float`, isso pode ser interpretado como uma documentação de que o campo deve ser do tipo `float`.

##### **8.6.1 Testes Automatizados**

- O script de testes está em `tests/test_api.py`
- Utiliza a biblioteca `pytest` e o `TestClient` do FastAPI para simular requisições HTTP e testar a API

- `TestClient(app)`: Neste trecho o FastAPI cria um "navegador virtual" (client), que permite enviar requisições para a API (app) sem precisar executar o servidor (sem uvicorn, sem docker), ou seja, é uma operação simulada na memória, tornando o teste ultra-rápido.

- `test_predict_legitimate_transaction():` Testa uma transação legítima que deve ser processada com sucesso (200 OK).


- `test_predict_invalid_amount():` Testa o funcionamento do contrato Pydantic (src/api/schemas.py), isto é, se a API retorna um erro (422 Unprocessable Entity) quando recebe valores inválidos em um objeto JSON encaminhado por um request/consulta de um sistema externo.



#### **8.7 CI/CD com GitHub Actions**

- Considerando que este é o primeiro projeto da Pós-Graduação, não implementamos CI/CD em etapas anteriores por priorizar a compreensão da construção do Modelo de Machine Learning e da API. Nesta etapa 8 (Deploy), implementaremos CI/CD com GitHub Actions para verificar se os scripts seguem as normas PEP8 (código limpo), verificar se a lógica do predictor.py está correta e verificar se a API está funcionando corretamente.

- **Tipos de Testes:**
    - **Linter:** Verificação da adequação do seu código às normas PEP8 (código limpo) utilizando a ferramenta **Ruff**.
    - **Unit Tests:** Verificação da adequação da lógica do predictor.py.
    - **Integration Tests:** Sobe a API e testa o endpoint /predict com um JSON real.

- **Implementação:**
    - Script de Implementação: `.github/workflows/main.yml`
    - Script de Testes: `tests/test_predict.py`
    - Atenção: A API_KEY deve ser configurada no seu repositório do GitHub porque não sobe para o GitHub.
    - Configuração de Variáveis de Ambiente: `secrets.API_KEY`
        - No seu repositório do GitHub: Settings > Secrets and variables > Actions.
        - Criar um ***New Repository Secret*** chamado `API_KEY` e colar o valor.

### 9. **Autenticação e Segurança**

#### 9.1 **Panorama Geral Autenticação e Segurança**
- **O que é?**
    - Autenticação é o processo de verificar *quem* está tentando acessar o sistema. 
    - Segurança envolve proteger a API contra acessos não autorizados, ataques maliciosos e vazamento de dados.
    - Sem isso, qualquer pessoa na internet poderia enviar dados para sua API ou tentar derrubá-la.

- **Opções de Mercado:**
    1.  **API Key (Chave de API):** Um código secreto simples enviado no cabeçalho da requisição.
        - *Prós:* Simples de implementar, fácil de usar para comunicação entre servidores (Machine-to-Machine).
        - *Contras:* Se a chave vazar, precisa ser trocada manualmente. Não tem gestão de usuários complexa.
    2.  **JWT (JSON Web Token):** Um token criptografado com validade temporária.
        - *Prós:* Mais seguro, permite expiração automática e carrega informações do usuário. Padrão para apps com login de usuário.
        - *Contras:* Mais complexo de implementar (requer endpoint de login).
    3.  **OAuth2:** A solução mais segura no mercado atualmente (ex: "Logar com Google").
        - *Prós:* Altamente seguro e padronizado.
        - *Contras:* Complexidade muito alta para uma API interna ou microserviço simples.

- **Nossa Escolha: API Key**
    - **Motivo:** Para este projeto de Detecção de Fraude, o cenário provável é que a API seja consumida por outro sistema interno de banco (Ex: sistema de transações), e não por um usuário final num navegador.
    - Para comunicação "Máquina x Máquina" (M2M) em ambiente controlado, a **API Key** oferece o melhor equilíbrio entre segurança e simplicidade. Ela garante que apenas quem tem o segredo (o sistema do banco) consiga solicitar predições, sem a sobrecarga de gerenciar logins e tokens temporários do OAuth2/JWT.

#### 9.2 **Implementação da API Key**
- **Segurança das Credenciais (.env):**
    - A chave de acesso `API_KEY` **não** é salva no código fonte. Ela é armazenada em um arquivo oculto `.env` que fica apenas no servidor e é ignorado pelo Git `.gitignore`.
    - O arquivo `.env.example` serve como modelo para outros desenvolvedores.
    - No script `main.py` foram usadas as bibliotecas `python-dotenv` e `os.getenv` para ler essa senha de forma segura. 
    - Se a senha não existir, o sistema trava propositalmente (Fail Fast) para evitar brechas de segurança `main.py:L32-33`.

- **Validação no FastAPI:**
    - No script `main.py:L35` foi usada a classe `APIKeyHeader` do FastAPI para validar a API Key.
    - No `main.py:L39-48` foi criada uma dependência `get_api_key` que intercepta todas as requisições ao endpoint `/predict` a fim de validar a API Key .
    - Se o cabeçalho `access_token` estiver ausente ou incorreto, a API retorna imediatamente um erro `403 Forbidden` ("Não foi possível validar as credenciais"), protegendo o modelo de uso indevido.

- **Atualização dos Testes:**
    - No `test_api.py:L24` foi feita uma atualização para carregar a chave real do ambiente de teste `test_api.py:L14-22` e enviá-la no cabeçalho das requisições .
    - No `test_api.py:L144-185` foi adicionado um teste específico `test_predict_unauthorized` para garantir que a porta de segurança está trancada para invasores.

### 10. **Monitoramento de Performance (Infraestrutura e Software)**

- Este monitoramento foca na saúde da API enquanto sistema web e responde à pergunta: "O sistema está disponível e rápido para o usuário?"

#### 10.1 **Métricas de Monitoramento de Performance**

- **Latência (Tempo de Resposta):** Mede o tempo de processamento de uma requisição pela API. Se o modelo de Random Forest começar a demorar 2 segundos em vez de 100ms, a experiência do usuário será ruim.

- **Taxa de Erros (HTTP 4xx e 5xx):** Monitora quantas requisições falham. Por exemplo, um aumento nos erros 500 pode indicar que o container Docker está ficando sem memória ou que o arquivo .pkl foi corrompido.

- **Throughput (Vazão):** Quantas predições por segundo a API está processando. Ajuda a decidir a necessidade de escalar o container Docker para mais instâncias.

- **Uso de Recursos (CPU/RAM):** Verifica se o carregamento do modelo está "pesando" no servidor. Redes Neurais tendem a consumir muita RAM, enquanto Random Forests são mais leves, mas exigem monitoramento de CPU.

#### 10.2 **Planejamento de Implementação de Monitoramento de Performance**

- **IMPORTANTE:**
    - Optou-se por não implementar o Monitoramento de Performance visto que este projeto é um projeto de estudo e não tem um ambiente de produção, contudo, o planejamento descrito abaixo é válido para projetos de produção.

- **OBJETIVO:**
    - Implementar um monitoramento profissional para rastrear latência da API, taxas de erro, throughput e recursos de infraestrutura (CPU/RAM).

- **ARQUITETURA:**
    - **Prometheus:** Banco de dados de séries temporais para coletar e armazenar métricas.
    - **Grafana:** Plataforma de visualização para dashboards.
    - **cAdvisor:** Ferramenta da Google para exportar métricas de uso de recursos de contêineres (CPU/RAM).
    - **FastAPI Instrumentator:**  Ferramenta de monitoramento e coleta de dados de telemetria de aplicações construídas com o framework FastAPI. Utilizada para observar, possibilita rastrear chamadas, métricas de desempenho e erros, o que facilita a depuração e otimização, ou seja, ajuda a entender o que acontece dentro da API. 

- **IMPLEMENTAÇÃO:**
    - Adicionar o pacote ***Prometheus-Fastapi-Instrumentator*** ao arquivo ***requirements.txt***.
    - Atualizar o script `src/api/main.py` para inicializar o Instrumentator.
    - Expor o endpoint `/metrics`.

- **INFRAESTRUTURA (DOCKER COMPOSE):**
    - Adicionar o serviço Prometheus (Porta 9090).
    - Adicionar o serviço Grafana (Porta 3000).
    - Adicionar o serviço cAdvisor (Porta 8080) para estatísticas de contêineres.

- **CONFIGURAÇÃO:**
    - Crie o arquivo `prometheus.yml` para coletar dados dos alvos (aplicativo, cAdvisor).

- **DOCUMENTAÇÃO:**
    - Crie o arquivo `docs/monitoring.md` com instruções de configuração e guia do painel de controle.

### 11. **Monitoramento de Modelo (Data Drift)**

- O **MONITORAMENTO DO MODELO** foca na qualidade das predições ao longo do tempo. Ele responde à pergunta: "O modelo ainda é inteligente ou ficou 'burro'?"

- O **DATA DRIFT (Desvio de Dados)** ocorre quando os dados que chegam na API hoje são estatisticamente diferentes dos dados de treino.

- O **CONCEPT DRIFT (Desvio de Conceito)** ocorre quando a definição de "fraude" muda.

- O **MODEL DECAY (Degradação)** é a queda natural das métricas conforme o modelo fica desatualizado.

- Para implementar o monitoramento utilizados a ferramenta **Evidently AI** para detectar **Data Drift** (Desvio de Dados). A ferramenta compara estatisticamente os dados de Treino (Referência) com os novos dados que chegam na API . O resultado é um relatório HTML interativo que é salvo no MLflow e permite visualizar facilmente quais colunas mudaram de distribuição.

#### 11.1 **Detalhes da Solução Implementada**

- **ARQUITETURA:**
    - **Evidently AI:** Biblioteca open-source utilizada para calcular métricas de drift e gerar relatórios visuais.
    - **Dados de Referência:** O dataset de treino (`X_train`) armazenado no SQLite, representando o padrão aprendido pelo modelo.
    - **Dados Atuais:** O dataset de teste (`X_test`) ou novos dados de produção, que são comparados contra a referência.
    - **MLflow:** Atua como repositório de artefatos, armazenando os relatórios HTML gerados para histórico e auditoria.

- **IMPLEMENTAÇÃO:**
    - Adicionado o pacote `evidently` ao arquivo `requirements.txt`.
    - Criado o script `src/models/monitor_drift.py` que carrega os dados do banco, simula um drift (para demonstração) e executa a análise.
    - O script utiliza a API Legacy do Evidently (`DataDriftPreset`) para compatibilidade e robustez.

- **INFRAESTRUTURA:**
    - O monitoramento é executado sob demanda (Job) via container Docker ou ambiente local, não exigindo um serviço "always-on" dedicado como o Prometheus.
    - Integra-se nativamente com a infraestrutura de banco de dados SQLite existente (`data/fraud_detection.db`).

- **CONFIGURAÇÃO:**
    - O script está configurado para utilizar o `DataDriftPreset`, que seleciona automaticamente os testes estatísticos adequados (ex: Kolmogorov-Smirnov para numéricos) com base no tipo de dado.

- **DOCUMENTAÇÃO:**
    - Criado o arquivo `docs/model_monitoring.md` com instruções passo-a-passo de execução e guia de interpretação dos gráficos de drift.

Para mais detalhes sobre como executar e interpretar o monitoramento, consulte o guia: [docs/model_monitoring.md](docs/model_monitoring.md).

### **12. PLANEJAMENTO DE DEPLOY EM NUVEM (GCP)**

- Esse projeto **não foi implementado em nuvem**, abaixo está o planejamento.

#### **12.1 CONSIDERAÇÕES E AJUSTES NECESSÁRIOS** 

- Fazer deploy de uma aplicação de Machine Learning que usa Docker, banco de dados e MLflow exige alguns ajustes para sair do ambiente local (onde tudo é salvo no seu disco) para a nuvem (containers são temporários).

- Considerando que vamos utilizar o **GCP**, precisamos nos atentar para o funcionamento Stateless do Cloud Run, ou seja, não podemos salvar dados no disco do container, porque o Container hospedado no Cloud Run nasce para responder uma requisição e morre logo após, ou seja, tudo que for salvo no container é perdido.

#### **12.1.1 Preparação da Aplicação (O "Docker de Produção")**

- Ajustar o Dockerfile para executar o servidor Uvicorn diretamente.

- Garantir que a porta exposta seja a definida pela variável de ambiente $PORT (exigência do Cloud Run) ou fixar em 8000.

#### **12.1.2 Externalização do Banco de Dados**

- Como o SQLite é um arquivo local, ele será apagado a cada deploy. A solução é migrar de SQLite para PostgreSQL. Há duas soluções possíveis, a primeira é usar o **Cloud SQL** (gerenciado), a segunda é subir um container de Postgres em uma VM (Compute Engine). 

- No código é necessário alterar a string de conexão no `src/data/db.py` e no MLflow para ler uma variável de ambiente `DATABASE_URL` em vez de fixar em sqlite:///....

#### **12.1.3 Armazenamento de Artefatos (Modelos)**

- Os **modelos .pkl** e os **gráficos do MLflow** também não podem ficar no disco local do container.

A solução é usar os **Buckets** do Google Cloud Storage (GCS).

- O MLflow tem suporte nativo, bastando configurar o `artifact_uri` para `gs://meu-bucket-mlflow/`.

#### **12.1.4 Armazenamento do Modelo ML em Produção**

- Treinar o modelo localmente, gerar o model.pkl, copiar para dentro da imagem Docker (COPY artifacts/model.pkl /app/artifacts/) e fazer o deploy.

A vantagem desta solução é que a API não depende de conexão externa para carregar o modelo. É uma solução simples, rápida e robusta.

A outra opção envolve o **MLflow Server Remoto**, ou seja, a API conecta num banco remoto e baixa o modelo do Bucket na inicialização. A vantagem é a possibilidade de trocar o modelo sem precisar fazer redeploy da API. A desvantagem é a complexidade de configurar a autenticação e rede.

#### **12.1.5 O Pipeline de Deploy (CI/CD)**

- Automatizar o processo usando GitHub Actions.

    - **Build:** O GitHub monta a imagem Docker.
    - **Push:** Envia a imagem para o Google Artifact Registry.
    - **Deploy:** Manda o Cloud Run atualizar o serviço com a nova imagem.













