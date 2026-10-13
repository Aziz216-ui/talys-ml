# 🚀 Talys-Trade — MLOps & GenAI Sales Forecasting Platform

![Talend](https://img.shields.io/badge/ETL-Talend-FF5722?style=flat&logo=talend&logoColor=white)
![MySQL](https://img.shields.io/badge/Data%20Mart-MySQL-4479A1?style=flat&logo=mysql&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6C00?style=flat&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-01875F?style=flat&logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/GenAI-Groq%20GPT--OSS-F05032?style=flat&logoColor=white)
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
- **Modèle LLM** : **GPT-OSS 120B** via **Groq API**, avec **Google Gemini** en option.
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

## 🔁 Réentraînement automatique orchestré par Prefect

Le pipeline lit le Data Mart MySQL `talys_datamart` et publie ses résultats dans MySQL
(`ml_predictions`, `ml_ventes_combinees`, `ml_training_runs`, `ml_reports`), tables lues directement par Power BI.

```powershell
# 1. Initialiser le Data Mart MySQL à partir des exports CSV (une seule fois)
python src\load_mysql.py

# 2. Démarrer le serveur Prefect (tableau de bord : http://127.0.0.1:4200)
prefect server start

# 3. Démarrer le déploiement planifié (tous les jours à 02:00)
python flowsetrain_flow.py --serve
#    ou, pour une démonstration, une vérification toutes les minutes :
python flowsetrain_flow.py --serve --interval 1

# Déclenchement événementiel (ex. à la fin du job Talend)
prefect deployment run "talys-retrain/talys-retrain-planifie"
```

Le flow `talys-retrain` détecte les nouvelles journées de ventes ; sans nouvelle donnée il s'arrête,
sinon il réentraîne, compare le challenger au modèle en production sur la même période de test
et ne déploie que s'il est meilleur. Un verrou MySQL empêche deux réentraînements simultanés.

## 🧪 Simulation de Production & Tests

```powershell
# Ajoute 3 nouvelles journées de ventes dans MySQL, une par minute
python src\simulate_production.py --days 3 --every 60

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