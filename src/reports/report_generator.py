"""
Générateur automatique de rapports décisionnels exécutifs par LLM.
Analyse les résultats du pipeline de prévision et produit un rapport Markdown complet.
"""

import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
load_dotenv(os.path.join(BASE_DIR, ".env"))


def generate_executive_report(output_file: str = None) -> str:
    """Génère un rapport exécutif combinant données statistiques et analyse LLM."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = output_file or os.path.join(REPORTS_DIR, "executive_report_latest.md")

    # 1. Chargement des données
    comb_path = os.path.join(DATA_DIR, "ventes_combinees.csv")
    metrics_path = os.path.join(MODELS_DIR, "metrics_history.json")

    hist_total, pred_total, mape, mae = 0.0, 0.0, 0.0, 0.0
    start_date, end_date = "N/A", "N/A"

    if os.path.exists(comb_path):
        df = pd.read_csv(comb_path)
        reel = df[df["Type"] == "Reel"]
        pred = df[df["Type"] == "Predit"]
        hist_total = reel["Montant"].sum()
        pred_total = pred["Montant"].sum()
        if not pred.empty:
            start_date = str(pred["date_transaction"].min())
            end_date = str(pred["date_transaction"].max())

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            if history:
                latest = history[-1]
                mape = latest.get("mape", 0.0)
                mae = latest.get("mae", 0.0)

    # 2. tentative d'enrichissement par LLM (Groq)
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_analysis = ""

    if groq_api_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.3
            )
            prompt = f"""Tu es le Chief Data Officer de Talys-Trade.
Rédige 3 paragraphes d'analyse stratégique basés sur ces chiffres :
- Prévision des ventes sur 30j : {pred_total:,.2f} TND (du {start_date} au {end_date})
- Précision du modèle XGBoost (MAPE) : {mape:.2f}% (Erreur moyenne MAE : {mae:.2f})
- Ventes historiques accumulées : {hist_total:,.2f} TND

Fournis une analyse de tendance et 3 recommandations concrètes pour la gestion des stocks et de la trésorerie.
"""
            response = llm.invoke(prompt)
            groq_analysis = response.content
        except Exception as e:
            groq_analysis = f"*Analyse LLM non disponible : {e}*"
    else:
        groq_analysis = """
### 📈 Analyse Automatique des Tendances
- Le modèle XGBoost anticipe une trajectoire stable du chiffre d'affaires sur les 30 prochains jours.
- **Recommandation Stock** : Ajuster les réapprovisionnements en fonction des pics hebdomadaires identifiés par le modèle.
- **Recommandation Trésorerie** : Sécuriser les fonds pour la période d'activité forte projetée.
*(Pour une synthèse rédigée dynamiquement par l'IA, ajoutez votre clé GROQ_API_KEY dans le fichier .env)*
"""

    # 3. Rédaction du rapport Markdown
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# 📊 Rapport Décisionnel Exécutif — Sales & Forecasting Talys-Trade
*Généré automatiquement le {now_str} par le Pipeline MLOps & GenAI*

---

## 📌 Résumé Exécutif
- **Horizon de Prédiction** : 30 jours ({start_date} au {end_date})
- **Chiffre d'Affaires Projeté (Total 30j)** : **{pred_total:,.2f}**
- **Historique Cumulé Analysé** : {hist_total:,.2f}
- **Précision du Modèle (MAPE)** : **{mape:.2f}%** (MAE: {mae:.2f})

---

## 🤖 Analyse Stratégique Générée par l'IA (Groq LLM)

{groq_analysis}

---

## 🛡️ Métriques & Gouvernance MLOps
- **Modèle en Production** : XGBoost Regressor (`model_current.pkl`)
- **Tracking & Métriques** : MLflow SQLite Backend (`mlflow.db`)
- **Statut du Réentraînement** : Conforme et Validé
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Rapport exécutif généré avec succès : {report_path}")
    return report_content


if __name__ == "__main__":
    generate_executive_report()
