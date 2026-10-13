"""
Acces unique aux resultats du pipeline (predictions, ventes combinees, metriques, rapport).

En mode MySQL (DATA_SOURCE=mysql), tout est lu dans talys_datamart : l'API, le Copilot,
Streamlit, le rapport LLM et Power BI affichent donc exactement les memes chiffres.
En mode CSV (CI GitHub), les fichiers exportes par le pipeline sont utilises.
"""

import json
import os

import pandas as pd

from db import BASE_DIR, get_engine, use_mysql

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")


def read_predictions():
    """Predictions a 30 jours : date_transaction, montant_predit."""
    if use_mysql():
        df = pd.read_sql("SELECT date_transaction, montant_predit FROM ml_predictions ORDER BY date_transaction",
                         get_engine())
    else:
        path = os.path.join(DATA_DIR, "predictions_ventes.csv")
        if not os.path.exists(path):
            return pd.DataFrame(columns=["date_transaction", "montant_predit"])
        df = pd.read_csv(path)
    df["date_transaction"] = pd.to_datetime(df["date_transaction"])
    return df


def read_combined_sales():
    """Serie reel + predit : date_transaction, Montant, Type."""
    if use_mysql():
        df = pd.read_sql("SELECT date_transaction, Montant, Type FROM ml_ventes_combinees ORDER BY date_transaction",
                         get_engine())
    else:
        path = os.path.join(DATA_DIR, "ventes_combinees.csv")
        if not os.path.exists(path):
            return pd.DataFrame(columns=["date_transaction", "Montant", "Type"])
        df = pd.read_csv(path)
    df["date_transaction"] = pd.to_datetime(df["date_transaction"])
    return df


def read_products():
    """Catalogue produits (dim_product)."""
    if use_mysql():
        return pd.read_sql("SELECT * FROM dim_product", get_engine())
    path = os.path.join(DATA_DIR, "dim_product.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def read_production_metrics():
    """Metriques du modele en production lors du dernier reentrainement, ou None."""
    if use_mysql():
        df = pd.read_sql(
            "SELECT date_run, production_mae AS mae, production_rmse AS rmse, production_mape AS mape, "
            "(SELECT COUNT(*) FROM ml_training_runs) AS nombre_total_runs "
            "FROM ml_training_runs ORDER BY date_run DESC LIMIT 1",
            get_engine(),
        )
        return df.iloc[0].to_dict() if not df.empty else None

    path = os.path.join(MODELS_DIR, "metrics_history.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        history = json.load(f)
    return {**history[-1], "nombre_total_runs": len(history)} if history else None


def read_latest_report():
    """Dernier rapport executif Markdown, ou None."""
    if use_mysql():
        df = pd.read_sql("SELECT markdown_content FROM v_latest_report", get_engine())
        return df.iloc[0, 0] if not df.empty else None

    path = os.path.join(REPORTS_DIR, "executive_report_latest.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()
