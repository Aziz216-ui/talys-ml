"""
Serveur API REST FastAPI pour Talys-Trade.
Expose des endpoints pour les prédictions XGBoost, l'Agent IA Copilot Sales, et la consultation de rapports.
"""

import os
import sys
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ai_agent.agent import create_sales_copilot
from reports.report_generator import REPORTS_DIR

app = FastAPI(
    title="Talys-Trade Sales & AI Copilot API",
    description="API REST d'inférence ML XGBoost et d'assistance décisionnelle GenAI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    horizon_days: Optional[int] = 30


class ChatRequest(BaseModel):
    message: str
    groq_api_key: Optional[str] = None


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Talys-Trade Sales & AI Copilot API",
        "version": "2.0.0",
        "endpoints": [
            "/api/predict (POST)",
            "/api/copilot/chat (POST)",
            "/api/reports/latest (GET)",
            "/docs (Swagger UI)"
        ],
    }


@app.post("/api/predict")
def predict_sales(req: PredictionRequest):
    """Retourne les prédictions XGBoost sur l'horizon spécifié (ex: 30j)."""
    pred_path = os.path.join(BASE_DIR, "data", "raw", "predictions_ventes.csv")
    if not os.path.exists(pred_path):
        raise HTTPException(
            status_code=404,
            detail="Fichier de prédictions introuvable. Exécutez le pipeline principal."
        )

    df = pd.read_csv(pred_path)
    df_head = df.head(req.horizon_days)
    records = df_head.to_dict(orient="records")

    return {
        "horizon_jours": len(records),
        "total_predit": round(float(df_head["montant_predit"].sum()), 2),
        "predictions": records,
    }


@app.post("/api/copilot/chat")
def chat_with_copilot(req: ChatRequest):
    """Dialogue en langage naturel avec l'Agent IA Copilot Sales."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message vide.")

    try:
        agent = create_sales_copilot(groq_api_key=req.groq_api_key)
        response = agent.invoke({"input": req.message})
        output_text = response.get("output", "")

        return {
            "query": req.message,
            "response": output_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Agent IA: {str(e)}")


@app.get("/api/reports/latest")
def get_latest_report():
    """Consulte le dernier rapport exécutif généré par le LLM."""
    report_path = os.path.join(REPORTS_DIR, "executive_report_latest.md")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Rapport non encore généré.")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "file": "executive_report_latest.md",
        "markdown_content": content,
    }
