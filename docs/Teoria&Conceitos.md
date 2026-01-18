# **CREDIT CARD FRAUD DETECTION ML API**

 - Essa é uma API de Machine Learning que usa um modelo de Random Forest para prever fraudes com base em um conjunto de dados de transações de cartão de crédito.

# **O QUE É MACHINE LEARNING?**

Técnica na qual sistemas de computador aprendem com dados e melhoram seu desempenho em uma tarefa específica sem serem explicitamente programados para ela. Em vez de receber um conjunto de regras rígidas, o algoritmo é "treinado" com grandes volumes de dados, encontrando padrões e criando um modelo preditivo ou de decisão.

# **TIPOS DE MACHINE LEARNING**

## **APRENDIZADO SUPERVISIONADO (SUPERVISED LEARNING)**

- O algoritmo é treinado com **dados rotulados**, ou seja, **pares de entrada e saída esperada**. O algoritmo recebe um conjunto de dados de entrada **($X$)** e as respostas corretas correspondentes **($y$)**. O objetivo é **aprender uma função** **$f(X) = y$** que consiga **prever o rótulo** de novos dados nunca vistos.

#### **CLASSIFICAÇÃO**

- Na Classificação, o objetivo é prever a categoria (rótulo discreto) a que um dado pertence, com base em características observadas. A Classificação responde a perguntas como: "Issa transação é Legítima ou é Fraude?".

- **Fluxo da Classificação**

    - **Conjunto de Treinamento:** Dados com rótulos conhecidos (Ex: Dados de transações de Cartões de Crédito com a etiqueta "Legítima" ou "Fraude").

    - **Mapeamento de Funções:** O algoritmo tenta encontrar a fronteira de decisão que melhor separa as classes.

    - **Generalização:** O modelo recebe dados novos e, com base na fronteira aprendida, atribui uma classe.

- **Algoritmos de Classificação:**

    - **Regressão Logística:** Apesar do nome, é utilizada em classificações. Estima a probabilidade de um evento pertencer a uma classe (0 a 1). Utilizada como baseline.

    - **K-Nearest Neighbors (KNN):** Classifica um ponto com base na "votação" dos vizinhos mais próximos. Simples, mas custoso para grandes volumes de dados.

    - **Árvores de Decisão (Decision Trees):** Cria regras lógicas (Se-Então) baseadas nos atributos. Altamente explicável.

    - **Random Forest:** Um "Floresta" de árvores de decisão que votam entre si, aumentando a precisão e evitando o overfitting.

    - **Support Vector Machines (SVM):** Busca o "hiperplano" que maximize a margem de separação entre as classes.

- **Métricas de Avaliação**

    - Na Classificação usamos a **Matriz de Confusão** para entender onde o modelo está errando:

    - **Acurácia:** 
        - Proporção total de acertos. 
        - Pode ocasionar interpretações erradas em dados desbalanceados.

    - **Precisão:** 
        - De todos que o modelo disse ser "Classe A", quantos realmente eram? (Evita Falsos Positivos).

    - **Recall (Revocação):** 
        - De todos que eram "Classe A" na realidade, quantos o modelo conseguiu pegar? (Evita Falsos Negativos).

    - **F1-Score:** 
        - A média harmônica entre Precisão e Recall.

#### **REGRESSÃO**

- Na Regressão, o objetivo é prever um valor numérico contínuo, ou seja, diferente da classificação (que escolhe uma categoria), a regressão tenta estimar um número exato em uma escala infinita ou dentro de um intervalo.

- O algoritmo busca encontrar uma função matemática que descreva a relação entre as variáveis de entrada ($X$) e a variável de saída ($y$).

- Equação da Reta (Regressão Linear Simples):$$y = \beta_0 + \beta_1x + \epsilon$$Onde $\beta_0$ é o intercepto, $\beta_1$ é o coeficiente (peso) da variável e $\epsilon$ é o erro (resíduo).

- OBS: Dependendo da complexidade dos dados, diferentes abordagens são necessárias

- **TIPOS DE REGRESSÕES LINEARES**

    - **Regressão Linear Simples:** Uma única variável de entrada prevendo o resultado.

    - **Regressão Linear Múltipla:** Várias variáveis influenciando o resultado (ex: área, bairro e idade de uma casa para prever o preço).

    - **Regressão Polinomial:** Usada quando a relação entre os dados não é uma linha reta, mas uma curva.

    - **Modelos de Árvore (Decision Tree Regressor):** Em vez de uma fórmula única, divide os dados em "caixas" e tira a média dos valores em cada uma.

    - **SVR (Support Vector Regression):** Uma variante do SVM focada em prever valores contínuos dentro de uma margem de erro tolerável.

- **MÉTRICAS DE ERRO**

- **MAE (Mean Absolute Error):** Média aritmética da distância absoluta entre a previsão e o real. Fácil de interpretar porque usa a mesma unidade dos dados.

- **MSE (Mean Squared Error):** Média dos erros ao quadrado. Penaliza erros grandes de forma muito mais severa.

- **RMSE (Root Mean Squared Error):** Raiz quadrada do MSE. Traz o erro de volta para a unidade original dos dados.

- **$R^2$ (Coeficiente de Determinação):** Indica o quanto do modelo explica a variabilidade dos dados. Vai de 0 a 1 (quanto mais próximo de 1, melhor).

## **APRENDIZADO NÃO SUPERVISIONADO (UNSUPERVISED LEARNING)**

- O modelo recebe apenas os dados de entrada **($X$)**, sem nenhuma resposta ou rótulo. O objetivo não é prever um valor, mas encontrar uma **estrutura oculta** ou **padrões desconhecidos** no dataset.

- O algoritmo atua como um explorador, agrupando dados por similaridade ou reduzindo a complexidade sem intervenção humana sobre o "que" cada grupo significa.


### **Clusterização**

- Clusterização é o processo de dividir um conjunto de objetos em grupos/clusters, de modo que os objetos no mesmo cluster sejam mais semelhantes entre si do que com os de outros clusters. Diferente da classificação, aqui o algoritmo não sabe previamente o que cada cluster representa.

- Clusterização está relacionada à três conceitos principais: 
    - **Métrica de Distância**: Função que mede a similaridade entre dois pontos, ou seja, o algoritmo calcula a "distância" entre eles (ex: Distância Euclidiana ou Manhattan). 
    - **Centroides**: Pontos centrais que representam a "média" ou o centro de um cluster.
    - **Inércia**: Medida de quão compactos são os clusters (menor inércia geralmente indica clusters melhor definidos).

- **ALGORÍTIMOS DE CLUSTERIZAÇÃO**:

    - **K-Means (Baseado em Partição):**
    - Divide o dataset em K grupos/clusters pré-definidos. A quantidade de grupos/clusters é informada ao algoritmo utilizando técnicas como o Método do Cotovelo.
    - Escolhe K pontos aleatórios como centroides iniciais, atribui cada dado ao centroide mais próximo e recalcula a posição do centroide repetidamente até estabilizar.

    - **Agrupamento Hierárquico:**
    - Cria uma árvore de clusters (Dendrograma), ou seja, não é necessário definir o número de clusters de anteriormente.
    - **Árvore Divisiva:** inicia o processo com todos os dados em um grupo e vai dividindo.
    - **Árvore Aglomerativa:** Começa com cada dado sendo um cluster e vai juntando os mais próximos.

    - **DBSCAN (Baseado em Densidade):**
    - Muito utilizado em datasets com formatos irregulares ou quando há muito ruído.
    - Identifica áreas de "alta densidade" e agrupa pontos próximos. Pontos em áreas isoladas são classificados como Outliers (ruído), facilitando a limpeza do dataset.

- **CASOS DE USO**:

    - **ANÁLISE DE HOTSPOTS DE CRIME**:

        - Ao agrupar coordenadas geográficas de ocorrências (latitude/longitude), o algoritmo identifica automaticamente áreas com maior incidência criminal sem que você precise definir bairros específicos.

    - **SEGMENTAÇÃO DE COMPORTAMENTO**:

        - No seu projeto de detecção de fraudes, você pode usar a clusterização para agrupar clientes por "perfil de gasto". Transações que caírem fora de qualquer cluster conhecido podem ser sinalizadas como suspeitas para análise posterior.

    - **PROCESSAMENTO DE LINGUAGEM NATURAL (NLP)**:

        - Agrupar boletins de ocorrência ou processos judiciais por similaridade de conteúdo textual para identificar padrões de modus operandi de grupos criminosos.

### **Redução de Dimensionalidade**

- A Redução de Dimensionalidade é o processo de reduzir o número de variáveis aleatórias sob consideração, obtendo um conjunto de variáveis "principais". 
- No contexto não supervisionado, isso é vital porque trabalhamos com dados que não possuem rótulos (labels), e a estrutura oculta pode estar mascarada pelo ruído de muitas colunas.

- **TÉCNICAS DE REDUÇÃO DE DIMENSIONALIDADE**:
    - **PCA (Principal Component Analysis):** 
        - Técnica linear mais comum. Transforma as variáveis originais em Componentes Principais (ortogonais entre si). O primeiro componente captura a maior variância possível dos dados.
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding) e UMAP (Uniform Manifold Approximation and Projection):** 
        - Técnicas não-lineares focadas em visualização de dados de alta dimensionalidade. Utilizadas para manter pontos que estão próximos no espaço de alta dimensão também próximos no espaço reduzido (2D ou 3D). Técnicas focadas em identificação de clusters complexos.

#### **PCA aplicada à Detecção de Fraudes em Cartões de Crédito:** 

- Neste projeto de detecção de fraudes em cartões de crédito, a redução de dimensionalidade é aplicada da seguinte forma:

    - **Problema:** Um dataset de transações pode ter centenas de colunas/variáveis (localização, valor, hora, tipo de estabelecimento, histórico do cliente, etc.).

    - **Solução:** Muitas dessas variáveis são correlacionadas, então, ao aplicar o PCA, você pode reduzir 30 variáveis para apenas 10 componentes principais que ainda explicam 95% da variação dos dados.

    - **Benefício:** O modelo de detecção de anomalias/fraudes rodará muito mais rápido e terá menos chances de sofrer overfitting (decorar o ruído dos dados).

### **Aprendizado por Reforço (Reinforcement Learning)**

- O algoritmo (Agente) aprende por meio da interação com um ambiente dinâmico, recebendo recompensas por ações desejadas e punições por ações indesejadas.

- O objetivo é maximizar a recompensa cumulativa.

- Exemplos: 
	- Treinamento de robôs, sistemas de jogos e veículos autônomos.

### **4. LIMPEZA E PRÉ-PROCESSAMENTO DOS DADOS**

#### **4.4 BALANCEAMENTO DOS DADOS (SAMPLING)**

**Opção A: Oversampling (SMOTE - Synthetic Minority Over-sampling Technique)**

- Em vez de apenas duplicar as fraudes existentes (o que causaria overfitting), o SMOTE cria **fraudes novas artificiais**.

 - Por meio de uma fraude real, avalia as fraudes "vizinhas" mais parecidas e cria um ponto intermediário entre elas. É como se ele interpolasse as características para criar uma fraude que "poderia existir".

 - A vantagem é que não há perda de informação (você mantém todos os dados legítimos). A desvantagem é que pode criar dados ruidosos se as fraudes estiverem misturadas com transações legítimas, confundindo o modelo.

**Opção B: Undersampling (Random Undersampling)**

- Você **joga fora aleatoriamente** a maioria das transações legítimas até ficar com uma quantidade parecida com a de fraudes (ex: 50/50).

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
