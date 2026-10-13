"""
Serveur API REST FastAPI pour Talys-Trade.
Expose des endpoints pour les prédictions XGBoost, l'Agent IA Copilot Sales, et la consultation de rapports.
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ai_agent.agent import create_sales_copilot
from data_access import read_latest_report, read_predictions

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
    chat_history: Optional[list] = None


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
    df = read_predictions()
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Aucune prédiction disponible. Exécutez le pipeline principal."
        )

    df_head = df.head(req.horizon_days)
    records = df_head.assign(date_transaction=df_head["date_transaction"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")

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
        response = agent.invoke({
            "input": req.message,
            "chat_history": req.chat_history or []
        })
        output_text = response.get("output", "")
        tools_called = response.get("tools_called", [])

        return {
            "query": req.message,
            "response": output_text,
            "tools_called": tools_called,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Agent IA: {str(e)}")


from fastapi.responses import HTMLResponse

@app.get("/chat", response_class=HTMLResponse)
def get_chat_ui():
    """Interface Web de Chat réelle pour l'Agent IA Copilot Sales."""
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Talys-Trade — Agent IA Copilot Sales</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            body { background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
            header { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
            .logo { font-size: 20px; font-weight: 700; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .status { font-size: 13px; color: #34d399; background: rgba(52, 211, 153, 0.1); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(52, 211, 153, 0.2); }
            main { flex: 1; display: flex; flex-direction: column; max-width: 900px; width: 100%; margin: 0 auto; padding: 20px; overflow: hidden; }
            #chat-box { flex: 1; overflow-y: auto; padding-right: 8px; display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth; }
            .msg { display: flex; gap: 12px; max-width: 80%; }
            .msg.user { align-self: flex-end; flex-direction: row-reverse; }
            .msg.bot { align-self: flex-start; }
            .avatar { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; flex-shrink: 0; }
            .user .avatar { background: #6366f1; color: white; }
            .bot .avatar { background: #0284c7; color: white; }
            .bubble { background: #1e293b; padding: 14px 18px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.06); font-size: 14px; line-height: 1.6; white-space: pre-wrap; }
            .user .bubble { background: #4f46e5; border-color: transparent; border-bottom-right-radius: 4px; }
            .bot .bubble { border-bottom-left-radius: 4px; }
            .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
            .chip { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #cbd5e1; padding: 8px 14px; border-radius: 20px; font-size: 12px; cursor: pointer; transition: all 0.2s; }
            .chip:hover { background: rgba(56, 189, 248, 0.15); border-color: #38bdf8; color: #38bdf8; }
            .input-area { background: #1e293b; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 8px 12px; display: flex; gap: 10px; align-items: center; }
            input { flex: 1; background: transparent; border: none; outline: none; color: white; font-size: 14px; padding: 8px; }
            button { background: linear-gradient(135deg, #0284c7, #6366f1); border: none; color: white; font-weight: 600; padding: 10px 20px; border-radius: 12px; cursor: pointer; transition: opacity 0.2s; }
            button:hover { opacity: 0.9; }
            .loading { display: flex; gap: 4px; align-items: center; height: 20px; }
            .dot { width: 6px; height: 6px; background: #94a3b8; border-radius: 50%; animation: pulse 1.4s infinite ease-in-out; }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes pulse { 0%, 100% { opacity: 0.3; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1.2); } }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">🤖 Talys-Trade Copilot IA</div>
            <div class="status">● Connecté (Groq GPT-OSS · Gemini)</div>
        </header>

        <main>
            <div id="chat-box">
                <div class="msg bot">
                    <div class="avatar">IA</div>
                    <div class="bubble">Bonjour ! Je suis le Copilot IA Sales & Forecasting pour Talys-Trade.
Comment puis-je vous aider aujourd'hui ? Vous pouvez me poser des questions sur les prévisions de ventes, les données historiques ou les performances des modèles ML.</div>
                </div>
            </div>

            <div class="chips" style="margin-top:16px;">
                <div class="chip" onclick="sendPrompt(this.innerText)">Quel est le chiffre d'affaires prévisionnel pour le mois prochain ?</div>
                <div class="chip" onclick="sendPrompt(this.innerText)">Quelle est la précision du modèle XGBoost (MAPE, MAE) ?</div>
                <div class="chip" onclick="sendPrompt(this.innerText)">Recherche des détails sur la catégorie de produit Bio</div>
            </div>

            <div class="input-area">
                <input type="text" id="userInput" placeholder="Posez votre question à l'Agent IA...">
                <button id="sendBtn" type="button">Envoyer</button>
            </div>
        </main>

        <script>
            document.getElementById('sendBtn').addEventListener('click', function() {
                sendMessage();
            });

            document.getElementById('userInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    sendMessage();
                }
            });

            function sendPrompt(text) {
                document.getElementById('userInput').value = text;
                sendMessage();
            }

            function sendMessage() {
                var input = document.getElementById('userInput');
                var text = input.value.trim();
                if (!text) return;

                var chatBox = document.getElementById('chat-box');

                // User Message
                var userDiv = document.createElement('div');
                userDiv.className = 'msg user';
                userDiv.innerHTML = '<div class="avatar">Vous</div><div class="bubble">' + escapeHtml(text) + '</div>';
                chatBox.appendChild(userDiv);

                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Loading Bot Message
                var botDiv = document.createElement('div');
                botDiv.className = 'msg bot';
                botDiv.innerHTML = '<div class="avatar">IA</div><div class="bubble"><div class="loading"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>';
                chatBox.appendChild(botDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                fetch('/api/copilot/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                })
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    botDiv.querySelector('.bubble').innerHTML = formatMarkdown(data.response || "Erreur de réponse");
                    chatBox.scrollTop = chatBox.scrollHeight;
                })
                .catch(function(err) {
                    botDiv.querySelector('.bubble').innerText = "Erreur de connexion à l'Agent IA.";
                    chatBox.scrollTop = chatBox.scrollHeight;
                });
            }

            function escapeHtml(text) {
                return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            }

            function formatMarkdown(text) {
                return text
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/\n/g, '<br>');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)



@app.get("/api/reports/latest")
def get_latest_report():
    """Consulte le dernier rapport exécutif généré par le LLM."""
    content = read_latest_report()
    if not content:
        raise HTTPException(status_code=404, detail="Rapport non encore généré.")

    return {
        "file": "executive_report_latest.md",
        "markdown_content": content,
    }

