FROM python:3.11-slim

WORKDIR /app

# Emplacements et dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copie intégrale des sources
COPY . .

# Ports exposés pour FastAPI (8000) et MLflow UI (5000)
EXPOSE 8000 5000

# Commande par défaut : Serveur FastAPI
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
