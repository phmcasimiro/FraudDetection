# Fraud Detection ML API

 - This is a Machine Learning API that uses a Random Forest model to predict fraud based on a dataset of credit card transactions.

 - Essa é uma API de Machine Learning que usa um modelo de Random Forest para prever fraudes com base em um conjunto de dados de transações de cartão de crédito.

## Setup

 - Verificar as versões de Python no sistema:

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
 - Configurar o versionador de código (Git e GitHub):

```bash
git --version
git config --global user.name "phmcasimiro"
git config --global user.email "phmcasimiro@gmail.com"
```
 - Criar Repositório no GitHub e sincronizar com o repositório local:

```bash
# GitHub

# 1. Acessar o GitHub
<https://github.com/phmcasimiro>

# 2. Escolha o nome do repositório
<FraudDetection>

# 3. Escolha o nível de acesso (público ou privado)
<Público>

# 4. Crie o repositório (sem marcar nada em "initialize this repository with...")
<https://github.com/phmcasimiro/FraudDetection.git>

# Git
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

## Criar Estrutura de Diretórios

```bash
mkdir src
mkdir src/api
mkdir src/data
mkdir src/models
mkdir tests
mkdir logs
mkdir artifacts
mkdir artifacts/models
```

## Criar arquivos `__init__.py`

```bash
# Linux
touch src/__init__.py
touch src/api/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch tests/__init__.py
```






git config --global user.name "Seu Nome"
git config --global user.email "seu.email@exemplo.com"
Verificação:
git --version
Comandos essenciais que usaremos:
git clone <url> # Copiar repositório
git add . # Adicionar mudanças
git commit -m "mensagem" # Salvar versão
git push # Enviar para servidor
git status # Ver estado atual
1.5 VS Code - Editor de Código
Download: https://code.visualstudio.com/
Extensões Essenciais
Após instalar VS Code, adicione estas extensões:
Python (Microsoft)
Busque "Python" no marketplace de extensões
Clique Install
Pylance (Microsoft)
1.
2.
3.
4.
