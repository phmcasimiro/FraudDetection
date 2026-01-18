
# 1. SETUP

## 1.1 Verificação das Versões de Python e pacote Pip do sistema:

 ```bash
python --version
python3 --version
```

- Verificar a versão do pacote pip:

```bash
pip --version
```

- Criar ambiente virtual:

```bash
python3 -m venv venv
```

- Ativar ambiente virtual:

```bash
source venv/bin/activate
```

- Verificar versão do ambiente virtual:

```bash
python -V
```

## 1.2 Versionamento de código (Git e GitHub):

1.2.1 Verificar a versão do pacote git:

```bash
git --version
```

1.2.2 Configurar o usuário e o email do git:

```bash
git config --global user.name "phmcasimiro"
git config --global user.email "phmcasimiro@gmail.com"
```

1.2.3 Criar Repositório no GitHub

- Criar repositório no GitHub (Nuvem)

- Acessar o GitHub: ***<https://github.com/phmcasimiro>***

- Clicar em Repositórios e nomear o novo repositório para:  ***FraudDetection***

- Escolher o nível de acesso (público ou privado): ***Público***

- Criar um novo repositório (sem marcar nada em "initialize this repository with...")
    - **OBS:** ***Sem README e sem .gitignore***
    - ***<https://github.com/phmcasimiro/FraudDetection.git>***

1.2.4 Sincronizar repositório local com o repositório da Nuvem (***FraudDetection***)

- Inicializar o Git localmente:

```bash 
git init
```

- Renomear a branch para main

```bash
git branch -m main
```

- Adicionar os arquivos ao "palco"

```bash
git add .
```

- Criar o primeiro commit (ponto de partida)

```bash
git commit -m "feat: Estrutura inicial do projeto de Detecção de Fraudes"
```

- Conectar o repositório local ao GitHub

- **OBS:** Substitua pela URL que você copiou

```bash
git remote add origin https://github.com/phmcasimiro/FraudDetection.git
```

- Enviar código para o GitHub

```bash
git push -u origin main
```

## 1.3 GERENCIADOR DE PROJETOS (GitHub Projects)

- **OBS:** O GitHub Projects é uma ferramenta que permite gerenciar projetos de forma colaborativa. Neste projeto será utilizado para gerenciar as tarefas de desenvolvimento e construir um histórico de progresso.


- Entrar no GitHub Projects

***<https://github.com/users/phmcasimiro/projects/2>***

- Canto superior direito, clicar em "settings"

***<https://github.com/users/phmcasimiro/projects/2/settings>***

- Selecionar ***Default Repository*** "FraudDetection"


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