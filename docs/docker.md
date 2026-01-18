# Guia Completo de Docker para Iniciantes

Este guia foi elaborado para ajudar você a entender, gerenciar e solucionar problemas no ambiente Docker do projeto **FraudDetection**.

---

## 1. Conceitos Fundamentais (O "Porquê")

Imagine que o Docker é uma **caixa mágica** que contém tudo o que seu projeto precisa para funcionar (Python, bibliotecas, banco de dados, etc.).

-   **Dockerfile:** É a **receita do bolo**. Um arquivo de texto com instruções passo a passo de como montar essa caixa.
-   **Imagem (Image):** É o **bolo pronto**. O resultado da receita. É um arquivo estático e imutável. Você não edita a imagem, você cria uma nova.
-   **Container:** É a **fatia do bolo sendo servida**. É a execução da imagem. Você pode ter vários containers rodando a partir da mesma imagem.
-   **Volume:** É um **portal mágico** entre seu computador e o container. Se você alterar um arquivo no seu computador, a alteração aparece instantaneamente dentro do container.
-   **Docker Compose:** É o **maître**. Ele coordena os containers, garantindo que tudo suba na ordem certa e com as configurações corretas (portas, volumes, variáveis de ambiente).

---

## 2. Comandos Essenciais

### 2.1 Iniciar o Projeto (Dia a Dia)

Para rodar o projeto e deixar a API disponível:

```bash
docker-compose up --build -d
```

-   `up`: Sobe os serviços.
-   `--build`: Força a reconstrução da imagem (garante que novas dependências sejam instaladas).
-   `-d`: Detached mode (roda em segundo plano, liberando seu terminal).

### 2.2 Parar o Projeto

Quando terminar de trabalhar:

```bash
docker-compose down
```

-   Para e remove os containers, mas mantém os dados do banco de dados (se estiverem salvos em volumes persistentes ou arquivos locais mapeados).

### 2.3 Ver o que está acontecendo (Logs)

Se algo der errado ou você quiser ver os `print()` do seu código:

```bash
docker-compose logs -f app
```

-   `-f`: Follow (acompanha em tempo real).
-   `app`: Nome do serviço definido no `docker-compose.yml`.
-   Para sair, pressione `Ctrl + C`.

### 2.4 Acessar o Terminal do Container

Às vezes você precisa entrar no container para rodar scripts manuais (como `python src/models/train.py`):

```bash
docker exec -it fraud_detection_container bash
```

-   `exec`: Executa um comando em um container que já está rodando.
-   `-it`: Interativo (permite digitar).
-   `fraud_detection_container`: Nome do container.

---

## 3. Gerenciamento e Limpeza

### 3.1 Listar Containers Ativos
```bash
docker ps
```

### 3.2 Listar Todos os Containers (inclusive parados)
```bash
docker ps -a
```

### 3.3 Limpar Tudo (Faxina Geral)
Se você estiver com problemas estranhos e quiser "resetar" o Docker (apaga containers, redes e imagens não usadas):

```bash
docker system prune -a
```
*Cuidado: Isso apaga todas as imagens não utilizadas do seu computador.*

---

## 4. Troubleshooting (Resolução de Problemas)

### 4.1 "Adicionei uma biblioteca nova no requirements.txt, mas o código diz que ela não existe."
**Causa:** O container está usando uma imagem antiga que não tinha essa biblioteca.
**Solução:** Reconstrua a imagem.
```bash
docker-compose up --build -d
```

### 4.2 "Mudei o nome de uma pasta e agora o Docker dá erro de 'File not found'."
**Causa:** O Dockerfile pode estar copiando arquivos de locais que não existem mais, ou o cache está atrapalhando.
**Solução:**
1. Verifique se o `Dockerfile` e o `docker-compose.yml` refletem a nova estrutura de pastas.
2. Force o rebuild sem cache:
```bash
docker-compose build --no-cache
docker-compose up -d
```

### 4.3 "Erro de conflito de dependência (Dependency Conflict)"
**Exemplo Real:** Ocorreu recentemente com `mlflow` e `packaging`.
**Sintoma:** O build falha com uma mensagem vermelha gigante explicando que a versão X exige a versão Y.
**Solução:**
1. Leia o erro com atenção. Ele geralmente diz: "package A requires package B < 2.0".
2. Abra o `requirements.txt`.
3. Fixe a versão da biblioteca problemática (ex: mudar `packaging==25.0` para `packaging==24.2`).
4. Rode `docker-compose up --build -d` novamente.

### 4.4 "A porta 8000 já está em uso"
**Causa:** Outro serviço (ou um container "zumbi") está usando a porta da API.
**Solução:**
1. Tente parar os containers: `docker-compose down`.
2. Se não resolver, procure o processo "ladrão" e mate-o:
   ```bash
   sudo lsof -i :8000
   kill -9 <PID>
   ```

---

## 5. Resumo da Arquitetura Docker Atual

-   **Imagem Base:** `python:3.12-slim` (Leve e rápida).
-   **Diretório de Trabalho:** `/app`.
-   **Volume:** A pasta atual do projeto (`.`) é espelhada para `/app`.
    -   *Vantagem:* Você edita no VS Code e o container vê a mudança na hora.
-   **Portas:** A porta `8000` do container é exposta na `8000` do seu PC.
-   **Comando Padrão:** Inicia o servidor `uvicorn` com *hot-reload*.
