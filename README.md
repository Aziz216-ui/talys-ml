# 🚀 Talys-Trade — MLOps & GenAI Sales Forecasting Platform

[![CI](https://github.com/Aziz216-ui/talys-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Aziz216-ui/talys-ml/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://talys-trade.streamlit.app)

![Talend](https://img.shields.io/badge/ETL-Talend-FF5722?style=flat&logo=talend&logoColor=white)
![MySQL](https://img.shields.io/badge/Data%20Mart-MySQL%20(Aiven)-4479A1?style=flat&logo=mysql&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6C00?style=flat&logo=xgboost&logoColor=white)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-01875F?style=flat&logo=mlflow&logoColor=white)
![Prefect](https://img.shields.io/badge/Orchestration-Prefect-024DFD?style=flat&logo=prefect&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/GenAI-Groq%20GPT--OSS-F05032?style=flat&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit%20Cloud-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Power BI](https://img.shields.io/badge/Analytics-Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)

Plateforme décisionnelle de prévision des ventes journalières : un **ETL Talend** alimente un
**Data Mart MySQL** hébergé dans le cloud ; un pipeline **MLOps** (XGBoost, MLflow) orchestré par
**Prefect** se réentraîne à l'arrivée de nouvelles données ; les résultats sont restitués dans
**Power BI** et dans un **Copilot IA conversationnel** (LLM + appel d'outils) déployé sur **Streamlit Cloud**.

👉 **Application en ligne : [talys-trade.streamlit.app](https://talys-trade.streamlit.app)**

---

## 🏗️ Architecture

```
 Sources ──▶ Talend (ETL) ──▶ MySQL talys_datamart (Aiven, connexion chiffrée)
                                 │  schéma en étoile + tables de sortie ML
                                 │
           Prefect ◀─────────────┤  détecte les nouvelles journées de ventes
     (flow talys-retrain)        │
   entraînement XGBoost ─────────▶  ml_predictions · ml_ventes_combinees
   Champion / Challenger            ml_training_runs · ml_reports
   rapport exécutif LLM              │
                                     ├──▶ Power BI (6 pages, ODBC chiffré)
                                     ├──▶ Copilot IA Streamlit (Streamlit Cloud)
                                     └──▶ API REST FastAPI (+ page de chat)

 GitHub Actions : tests unitaires à chaque push (CI)
```

**Source unique de vérité** : Power BI, l'application Streamlit, l'API et le rapport LLM lisent tous
le même Data Mart MySQL ; les chiffres sont identiques partout.

---

## ✨ Fonctionnalités clés

### 1. 🔄 ETL & Data Warehouse
- **Talend Open Studio** : extraction, nettoyage et chargement du Data Mart en étoile
  (`fact_sales`, `dim_date`, `dim_product`, `dim_client`, `dim_localisation`).
- **MySQL hébergé (Aiven)** : connexion TLS avec vérification du certificat ; vue
  `v_ventes_quotidiennes` qui fournit la série temporelle au modèle.

### 2. 📈 Prévision des ventes (Machine Learning)
- **XGBRegressor** optimisé par `RandomizedSearchCV` avec validation temporelle `TimeSeriesSplit`.
- **Variables** : calendrier, retards (1, 7, 30 jours), moyennes mobiles (7, 30 jours).
- **Prévision autorégressive** sur 30 jours.

### 3. 🔁 Réentraînement automatique (MLOps)
- **Prefect** : flow `talys-retrain` planifié chaque jour à 02:00 ou déclenché à la fin du job Talend ;
  il s'arrête si aucune nouvelle journée de ventes n'est arrivée.
- **Champion / Challenger** : le nouveau modèle est comparé au modèle en production **sur la même
  période de test** et n'est déployé que s'il est meilleur (MAPE).
- **MLflow** : suivi des paramètres, métriques et modèles de chaque entraînement.
- **Robustesse** : verrou MySQL contre les exécutions parallèles, publication atomique des résultats,
  nouvelles tentatives automatiques sur les accès base de données.

### 4. 🤖 Copilot IA (GenAI)
- **LLM** : GPT-OSS 120B via **Groq** (Google Gemini en option), avec appel d'outils :
  `get_sales_forecast`, `get_historical_sales`, `get_model_metrics`, `search_product_catalog`.
- **Catalogue produits** : recherche sémantique RAG (**ChromaDB**) en local, recherche par
  catégorie / sous-catégorie / branche sur l'application déployée.
- **Rapport exécutif** rédigé par le LLM après chaque réentraînement et affiché dans Power BI.

### 5. 📊 Power BI
6 pages : Ventes, Produits, Clients, Prévisions, MLOps & Performance du modèle, IA Copilot.
Les métriques affichées sont celles du **modèle en production** ; un bouton ouvre le Copilot IA en ligne.

---

## ⚡ Démarrage

### Configuration (`.env`)
Copier `.env.example` en `.env`, puis renseigner les clés LLM et la connexion MySQL :
```
DATA_SOURCE=mysql
MYSQL_URL=mysql+pymysql://avnadmin:MOT_DE_PASSE@HOTE.aivencloud.com:PORT/talys_datamart
MYSQL_SSL_CA=certs/ca.pem
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
```

### Installation et initialisation du Data Mart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Crée le schéma et charge le Data Mart depuis les exports CSV (une seule fois)
python src\load_mysql.py
```

### Réentraînement orchestré par Prefect
```powershell
prefect server start                               # tableau de bord : http://127.0.0.1:4200
python flows\retrain_flow.py --serve               # déploiement planifié (tous les jours à 02:00)
python flows\retrain_flow.py --serve --interval 1  # démonstration : vérification chaque minute

# Déclenchement événementiel (ex. à la fin du job Talend)
prefect deployment run "talys-retrain/talys-retrain-planifie"
```

### Simulation de production
```powershell
# Ajoute 3 nouvelles journées de ventes dans MySQL, une par minute
python src\simulate_production.py --days 3 --every 60
```

### Applications
```powershell
streamlit run src\chat_app.py                       # Copilot IA : http://localhost:8501
uvicorn src.api.app:app --port 8000                 # API : http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # MLflow : http://localhost:5000
```
Avec Docker : `docker compose up --build` démarre l'API (port 8000) et MLflow UI (port 5000).

### Tests
```powershell
pytest
```

---

## 📌 API REST (FastAPI)

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | État du service |
| `POST` | `/api/predict` | Prévisions XGBoost sur N jours |
| `POST` | `/api/copilot/chat` | Question en langage naturel au Copilot IA |
| `GET` | `/api/reports/latest` | Dernier rapport exécutif |
| `GET` | `/chat` | Page web de chat |

---

## 📁 Structure

```
flows/retrain_flow.py      Flow Prefect de réentraînement
src/main.py                Étapes du pipeline (chargement, entraînement, prévisions, publication)
src/db.py                  Connexion MySQL (SSL)          src/data_access.py   Lecture des résultats
src/load_mysql.py          Initialisation / migration du Data Mart
src/simulate_production.py Simulation de nouvelles journées de ventes
src/ai_agent/              Agent Copilot, outils, RAG     src/reports/         Rapport LLM
src/api/app.py             API FastAPI                    src/chat_app.py      Application Streamlit
sql/schema.sql             Schéma MySQL (Data Mart + tables et vues ML)
```

---

## 👨‍💻 Auteur
**Melki Mohamed Aziz** — Projet de fin d'études, stage ingénieur 2026.
