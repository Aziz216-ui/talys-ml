# Talys-Trade — Forecasting des ventes

Pipeline de prévision du montant total des ventes journalières (XGBoost),
intégré au data mart MySQL (issu du pipeline ETL Talend) et exposé via Power BI.

## Structure
- `src/` : code source du pipeline (data loading, features, training, prediction)
- `models/` : modèles entraînés et historique des métriques
- `notebooks/` : exploration initiale (Colab)