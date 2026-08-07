"""
Outils métier pour l'Agent IA Copilot Sales & Forecasting.
Fournit un accès aux prédictions, à l'historique des ventes, aux métriques MLflow et aux produits.
"""

import json
import os
import pandas as pd
from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@tool
def get_sales_forecast(horizon_days: int = 30) -> str:
    """
    Obtient les prédictions du montant total des ventes journalières pour les prochains jours.
    Utile quand l'utilisateur demande les prévisions, les ventes futures ou le chiffre d'affaires projeté.
    """
    pred_path = os.path.join(DATA_DIR, "predictions_ventes.csv")
    if not os.path.exists(pred_path):
        return "Aucune prédiction disponible. Veuillez réentraîner le modèle."

    df = pd.read_csv(pred_path)
    df["date_transaction"] = pd.to_datetime(df["date_transaction"])
    df = df.head(horizon_days)

    total_pred = df["montant_predit"].sum()
    avg_daily = df["montant_predit"].mean()
    min_day = df.loc[df["montant_predit"].idxmin()]
    max_day = df.loc[df["montant_predit"].idxmax()]

    summary = {
        "horizon_jours": len(df),
        "date_debut": df["date_transaction"].min().strftime("%Y-%m-%d"),
        "date_fin": df["date_transaction"].max().strftime("%Y-%m-%d"),
        "montant_total_predit": round(total_pred, 2),
        "moyenne_journaliere_predite": round(avg_daily, 2),
        "jour_ventes_min": {
            "date": min_day["date_transaction"].strftime("%Y-%m-%d"),
            "montant": round(min_day["montant_predit"], 2),
        },
        "jour_ventes_max": {
            "date": max_day["date_transaction"].strftime("%Y-%m-%d"),
            "montant": round(max_day["montant_predit"], 2),
        },
    }

    return json.dumps(summary, indent=2, ensure_ascii=False)


@tool
def get_historical_sales(days: int = 30) -> str:
    """
    Obtient les statistiques sur l'historique récent des ventes réelles.
    Utile pour analyser les tendances passées, les meilleures journées ou le montant total vendu.
    """
    comb_path = os.path.join(DATA_DIR, "ventes_combinees.csv")
    if not os.path.exists(comb_path):
        return "Historique des ventes indisponible."

    df = pd.read_csv(comb_path)
    df_reel = df[df["Type"] == "Reel"].copy()
    df_reel["date_transaction"] = pd.to_datetime(df_reel["date_transaction"])
    df_reel = df_reel.sort_values("date_transaction").tail(days)

    total_real = df_reel["Montant"].sum()
    avg_real = df_reel["Montant"].mean()

    summary = {
        "periode_jours": len(df_reel),
        "date_debut": df_reel["date_transaction"].min().strftime("%Y-%m-%d"),
        "date_fin": df_reel["date_transaction"].max().strftime("%Y-%m-%d"),
        "montant_total_reel": round(total_real, 2),
        "moyenne_journaliere_reelle": round(avg_real, 2),
    }

    return json.dumps(summary, indent=2, ensure_ascii=False)


@tool
def get_model_metrics() -> str:
    """
    Obtient les métriques de performance actuelles du modèle XGBoost (MAE, RMSE, MAPE).
    Utile si l'utilisateur demande la précision du modèle ou l'erreur moyenne de prédiction.
    """
    metrics_path = os.path.join(MODELS_DIR, "metrics_history.json")
    if not os.path.exists(metrics_path):
        return "Aucune métrique disponible."

    with open(metrics_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return "Historique des métriques vide."

    latest = history[-1]
    result = {
        "dernier_reentrainement": {
            "MAE": round(latest.get("mae", 0), 2),
            "RMSE": round(latest.get("rmse", 0), 2),
            "MAPE": f"{round(latest.get('mape', 0), 2)}%",
        },
        "nombre_total_runs": len(history),
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def search_product_catalog(query: str) -> str:
    """
    Recherche dans le catalogue de produits (dim_product) par mot-clé, nom de catégorie ou libellé.
    Exemple de query: 'Clavier', 'Électronique', 'Imprimante'.
    """
    prod_path = os.path.join(DATA_DIR, "dim_product.csv")
    if not os.path.exists(prod_path):
        return "Catalogue produit indisponible."

    df = pd.read_csv(prod_path)

    # Recherche dans les colonnes textuelles
    matches = df[
        df.apply(
            lambda row: row.astype(str).str.contains(query, case=False).any(),
            axis=1,
        )
    ].head(10)

    if matches.empty:
        return f"Aucun produit trouvé correspondant au mot-clé '{query}'."

    records = matches.to_dict(orient="records")
    return json.dumps(records, indent=2, ensure_ascii=False)
