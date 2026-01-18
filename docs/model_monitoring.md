# Guia de Monitoramento de Modelo (Data Drift)

Este documento explica como funciona o sistema de monitoramento de **Data Drift** implementado no projeto, utilizando a biblioteca **Evidently AI**.

## 1. O que é Data Drift?

**Data Drift** (Desvio de Dados) ocorre quando a distribuição estatística dos dados de entrada em produção muda significativamente em relação aos dados usados para treinar o modelo.

Isso pode acontecer por vários motivos:
-   Mudança no comportamento do consumidor (ex: Black Friday).
-   Novos tipos de fraude.
-   Erros no pipeline de dados.

Quando ocorre Data Drift, a performance do modelo tende a cair, pois ele está fazendo predições sobre dados que ele "não conhece bem".

## 2. Ferramenta: Evidently AI

Utilizamos o **Evidently AI** para comparar dois conjuntos de dados:
1.  **Reference (Referência):** O dataset de **Treino** (`X_train`), que representa o "mundo conhecido" pelo modelo.
2.  **Current (Atual):** O dataset de **Teste** (`X_test`) ou novos dados de produção, que queremos validar.

## 3. Como Executar o Monitoramento

O script de monitoramento está localizado em `src/models/monitor_drift.py`.

### Comando
```bash
python -m src.models.monitor_drift
```

### O que o script faz?
1.  Carrega `X_train` (Referência) e `X_test` (Atual) do banco de dados SQLite.
2.  **Simula um Drift Artificial:** Para fins de demonstração, o script altera propositalmente 50% dos dados de teste (multiplica `Amount` por 5 e desloca `V1`).
3.  Calcula o **Data Drift Report** usando testes estatísticos (ex: Kolmogorov-Smirnov).
4.  Gera um relatório HTML interativo.
5.  Loga o relatório e as métricas no **MLflow**.

## 4. Interpretando o Relatório

O relatório é salvo em `artifacts/drift_report.html`. Ao abri-lo no navegador, você verá:

### 4.1 Data Drift Summary
-   **Drifted Columns:** Quantas colunas sofreram desvio estatístico.
-   **Dataset Drift:** Se o dataset como um todo é considerado "desviado" (geralmente se > 50% das colunas tiverem drift).

### 4.2 Drift por Coluna
-   Clique em uma coluna (ex: `Amount`) para ver o gráfico de distribuição.
-   **Verde (Reference):** Distribuição original.
-   **Vermelho/Cinza (Current):** Distribuição atual.
-   Se as curvas estiverem muito separadas, houve drift.

## 5. Integração com MLflow

O monitoramento é registrado no experimento `fraud_detection_monitoring`.
Você pode acessar o MLflow UI para ver o histórico de drifts ao longo do tempo:

```bash
mlflow ui
```

No MLflow, você encontrará:
-   **Métrica `drift_share`:** Porcentagem de colunas com drift.
-   **Artefato `drift_report.html`:** O relatório completo para download/visualização.
