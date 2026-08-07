# 🚀 Talys-Trade — MLOps & GenAI Sales Forecasting Platform

![Talend](https://img.shields.io/badge/ETL-Talend-FF5722?style=flat&logo=talend&logoColor=white)
![MySQL](https://img.shields.io/badge/Data%20Mart-MySQL-4479A1?style=flat&logo=mysql&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6C00?style=flat&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-01875F?style=flat&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/GenAI-Groq%20Llama%203.3-F05032?style=flat&logo=meta&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=githubactions&logoColor=white)
![Power BI](https://img.shields.io/badge/Analytics-Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)

 Plateforme industrielle de prévision des ventes journalières, combinant un pipeline **ETL Talend**, un **Data Mart MySQL**, un pipeline **MLOps complet (XGBoost + MLflow + CI/CD)**, un **Agent IA Copilot conversationnel (Groq LLM + Tool Calling)**, un moteur **RAG (ChromaDB)** et des rapports décisionnels automatisés exposés sur **FastAPI** et **Power BI**.

---


## ✨ Fonctionnalités Clés

### 1. 🔄 Pipeline ETL & Data Integration (Talend Open Studio)
- **Extraction & Normalisation** : Extraction des transactions brutes, nettoyage et chargement dans le Data Mart MySQL (modèle en étoile : `fact_sales`, `dim_date`, `dim_product`, `dim_client`, `dim_localisation`).

### 2. 📈 Pipeline MLOps & Prévision des Ventes
- **Modèle ML** : `XGBRegressor` avec tuning hyperparamétrique via `RandomizedSearchCV` et découpage temporel `TimeSeriesSplit`.
- **Ingénierie des Caractéristiques** : Variables calendaires, Lags ($1, 7, 30$ jours), Moyennes mobiles ($7, 30$ jours).
- **Tracking & Versioning** : Métriques ($MAE, RMSE, MAPE$) enregistrées dans une base de données SQLite `mlflow.db` avec sauvegarde du meilleur modèle.

### 3. 🤖 Agent IA Copilot Sales & Tool Calling (GenAI)
- **Modèle LLM** : **Llama 3.3 70B** propulsé par **Groq API** (réponses ultra-rapides en < 1 sec).
- **Capacité Tool Calling** : L'Agent choisit dynamiquement les outils à exécuter :
  - `get_sales_forecast` : Prédictions 30j.
  - `get_historical_sales` : Ventes réelles passées.
  - `get_model_metrics` : Précision du modèle ($MAPE, MAE$).
  - `search_product_catalog` : Recherche sémantique dans le catalogue produits.

### 4. 📚 Moteur RAG (Retrieval-Augmented Generation)
- Base de données vectorielle **ChromaDB** indexant les produits de `dim_product.csv` pour permettre des requêtes sémantiques en langage naturel.

### 5. 📄 Génération Automatique de Rapports Exécutifs LLM
- À la fin de chaque réentraînement, un rapport de synthèse décisionnel est rédigé automatiquement en Markdown par le LLM (`data/reports/executive_report_latest.md`).

### 6. 🐳 Conteneurisation & CI/CD
- **Docker Compose** : Orchestration multi-services (FastAPI REST Server + MLflow UI).
- **GitHub Actions** : Workflow automatisé d'intégration continue déclenché sur modification des données ou calendrier.

---

## ⚡ Démarrage Rapide

### Option A : Lancement avec Docker Compose (Recommandé)

```bash
# 1. Cloner le projet
git clone https://github.com/Aziz216-ui/talys-ml.git
cd talys-ml

# 2. Configurer la clé Groq (dans .env)
echo "GROQ_API_KEY=votre_cle_groq_ici" > .env

# 3. Lancer toute l'infrastructure
docker compose up --build
```
- **FastAPI Documentation (Swagger UI)** : [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Dashboard** : [http://localhost:5000](http://localhost:5000)

---

### Option B : Lancement Local Python

```powershell
# 1. Créer et activer l'environnement virtuel
python -m venv .env
.\.venv\Scripts\Activate.ps1

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Exécuter le pipeline complet de réentraînement
python src\main.py

# 4. Lancer le serveur API FastAPI
uvicorn src.api.app:app --reload --port 8000

# 5. Lancer l'interface MLflow UI (dans un autre terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 🧪 Simulation de Production & Tests

Pour simuler l'arrivée de nouvelles ventes réelles en production et observer le réentraînement automatique :

```powershell
# Simule 800 nouvelles ventes et réentraîne le modèle
python src\simulate_production.py

# Lancer la suite de tests unitaires
pytest tests/
```

---

## 📌 Référence de l'API REST FastAPI

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Vérification de l'état de santé du service |
| `POST` | `/api/predict` | Obtenir les prédictions XGBoost sur $N$ jours |
| `POST` | `/api/copilot/chat` | Interroger l'Agent IA Copilot en langage naturel |
| `GET` | `/api/reports/latest` | Consulter le dernier rapport de synthèse exécutif |

---

## 👨‍💻 Auteur
Développé par **Aziz** — AI / MLOps Engineer.