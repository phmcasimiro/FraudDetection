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