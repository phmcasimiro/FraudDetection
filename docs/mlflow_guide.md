# Guia de Uso do MLflow

## Introdução

O MLflow foi implementado no projeto FraudDetection para rastrear experimentos de treinamento, gerenciar versões de modelos e facilitar a reprodutibilidade.

## Instalação

A biblioteca MLflow foi incluída no `requirements.txt`. 

Para instalar:

```bash
pip install -r requirements.txt
```

## Configuração

O MLflow está configurado para usar:
- **Backend Store**: SQLite `mlflow.db`
- **Artifact Store**: Sistema de arquivos local `mlruns/`

## Executando o Treinamento com MLflow

Ao executar o script de treinamento, o MLflow automaticamente registrará `Parâmetros`, `Métricas`, `Tags` e `Artefatos`.

```bash
python -m src.models.train
```

### O que é registrado:

#### **Parâmetros (Hyperparameters)**
- `n_estimators`: Número de árvores do Random Forest
- `max_depth`: Profundidade máxima das árvores
- `random_state`: Seed para reprodutibilidade
- `algorithm`: Tipo de algoritmo (RandomForest)
- `balancing_method`: Método de balanceamento (SMOTETomek)
- `dataset_size`: Tamanho do dataset de treino
- `num_features`: Número de features
- `fraud_percentage`: Porcentagem de fraudes no dataset

#### **Métricas**
- `train_accuracy`: Acurácia no conjunto de treino

#### **Tags**
- `autor`: phmcasimiro
- `projeto`: FraudDetection
- `versao`: Versão do modelo

#### **Artefatos**
- Modelo treinado (formato MLflow)
- Snapshot do código

## Visualizando Experimentos no MLflow UI

### Importante: Comando para Iniciar

Como configuramos o MLflow para usar um banco de dados **SQLite** (`mlflow.db`), É necessário usar o seguinte comando para iniciar a interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

> **Nota**: Rodando apenas `mlflow ui`, ele procurará arquivos na pasta padrão e não encontrará seus experimentos (aparecerá "No Experiments Exist").

### Iniciar em porta customizada (opcional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080
```

## Estrutura do MLflow UI

Ao acessar a interface, você verá:

### 1. **Experiments** (Experimentos)
- Lista de todos os experimentos:
  - `fraud_detection_training`
  - `fraud_detection_evaluation`

### 2. **Runs** (Execuções)
- Cada treinamento cria um "run"
- Informações de cada run:
  - **Start Time**: Quando o treino iniciou
  - **Duration**: Tempo de execução
  - **Parameters**: Hiperparâmetros utilizados
  - **Metrics**: Métricas registradas
  - **Tags**: Metadados adicionais

### 3. **Comparação de Runs**
- Selecione múltiplos runs
- Clique em "Compare"
- Visualize diferenças em parâmetros e métricas lado a lado

### 4. **Artifacts**
- Modelos salvos
- Gráficos e visualizações
- Arquivos de log

## Comparando Experimentos

### Via UI
1. Selecione 2 ou mais runs (checkboxes)
2. Clique no botão **Compare**
3. Veja a tabela comparativa de parâmetros e métricas

### Via Python (Opcional)

```python
import mlflow

# Buscar todos os runs do experimento
experiment = mlflow.get_experiment_by_name("fraud_detection_training")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Ordenar por métrica
best_run = runs.sort_values("metrics.train_accuracy", ascending=False).iloc[0]
print(f"Melhor run: {best_run['run_id']}")
print(f"Acurácia: {best_run['metrics.train_accuracy']}")
```

## Carregando Modelos do MLflow

### Carregar modelo registrado

```python
import mlflow.sklearn

# Carregar a versão mais recente do modelo
model_uri = "models:/FraudDetectionRandomForest/latest"
modelo = mlflow.sklearn.load_model(model_uri)

# Fazer predições
predictions = modelo.predict(X_test)
```

### Carregar modelo de um run específico

```python
import mlflow

# Copie o run_id da UI
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"
modelo = mlflow.sklearn.load_model(model_uri)
```

## Model Registry

O modelo é automaticamente registrado no **Model Registry** com o nome:
- `FraudDetectionRandomForest`

### Promover modelo para produção (via UI)

1. Acesse **Models** no menu lateral
2. Clique em `FraudDetectionRandomForest`
3. Selecione uma versão
4. Clique em **Stage** → **Transition to Production**

### Promover modelo via Python

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="FraudDetectionRandomForest",
    version=1,
    stage="Production"
)
```

## Workflow Recomendado

### 1. Experimentos (Development)
```bash
# Treinar com diferentes hiperparâmetros
# Cada execução cria um novo run no MLflow
python -m src.models.train
```

### 2. Avaliação
```bash
# Visualizar no MLflow UI
mlflow ui

# Comparar runs
# Selecionar o melhor modelo
```

### 3. Registro
```bash
# O modelo já é registrado automaticamente
# Verificar no Model Registry via UI
```

### 4. Produção
```python
# Carregar modelo de produção na API
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/FraudDetectionRandomForest/Production")
```

## Boas Práticas

### Sempre registre
- Parâmetros modificados
- Métricas de validação
- Informações do dataset

### Use tags descritivas
- Autor do experimento
- Objetivo do teste
- Versão do código

### Compare antes de promover
- Sempre compare métricas de múltiplos runs
- Verifique se há melhoria significativa

### Documente mudanças
- Use a descrição do run para explicar mudanças
- Adicione notas sobre resultados inesperados

## Comandos Úteis

### Listar experimentos
```bash
mlflow experiments list
```

### Buscar runs
```bash
mlflow runs list --experiment-name fraud_detection_training
```

### Servir modelo via API
```bash
mlflow models serve -m models:/FraudDetectionRandomForest/Production -p 5001
```

## Troubleshooting

### Erro: "No module named 'mlflow'"
```bash
pip install mlflow==2.18.0
```

### MLflow UI não abre
- Verifique se a porta 5000 está disponível
- Tente outra porta: `mlflow ui --port 8080`

### Modelo não aparece no Registry
- Verifique se `mlflow.sklearn.log_model()` foi executado
- Confirme que `registered_model_name` está especificado

## Recursos Adicionais

- [Documentação Oficial do MLflow](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)
- [Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
