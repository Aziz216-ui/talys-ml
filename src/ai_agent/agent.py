"""
Agent IA Copilot Sales & Forecasting utilisant Groq LLM et le Tool Calling.
Capable de répondre de manière fluide en langage naturel aux questions des décideurs.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage

from .tools import (
    get_sales_forecast,
    get_historical_sales,
    get_model_metrics,
    search_product_catalog,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SYSTEM_PROMPT = """Tu es un Assistant IA Expert en Analytics, Ventes et Forecasting pour l'entreprise Talys.
Ta mission est d'aider les managers et décideurs à comprendre les performances passées, les prédictions futures du chiffre d'affaires et la précision des modèles ML (XGBoost).

Consignes :
1. Utilise toujours les outils mis à ta disposition pour obtenir des chiffres précis avant de répondre.
2. Réponds de façon synthétique, professionnelle et structurée en Français.
3. Mets en valeur les chiffres clés (Montants en TND/EUR/USD, métriques MAPE/MAE, dates clés).
4. Si l'utilisateur pose une question sur un produit ou une catégorie, recherche dans le catalogue produits.
5. Sois proactif : propose des explications métier simples si des variations de ventes sont détectées.
"""


class GroqSalesCopilot:


    def __init__(self, groq_api_key: str, model_name: str):
        self.tools_map = {
            "get_sales_forecast": get_sales_forecast,
            "get_historical_sales": get_historical_sales,
            "get_model_metrics": get_model_metrics,
            "search_product_catalog": search_product_catalog,
        }
        tools = list(self.tools_map.values())
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.2,
        )
        self.llm_with_tools = self.llm.bind_tools(tools)

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_input),
        ]

        response = self.llm_with_tools.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                tool_fn = self.tools_map.get(name)
                if tool_fn:
                    tool_result = tool_fn.invoke(args)
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

            final_response = self.llm.invoke(messages)
            return {"output": final_response.content}

        return {"output": response.content}


def create_sales_copilot(groq_api_key: str = None, model_name: str = None):
    """Crée et initialise l'agent IA Copilot Sales avec Groq API."""
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    selected_model = model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        print("Note: GROQ_API_KEY non configurée. L'agent utilisera un mode dégradé avec réponses simulées.")
        return MockSalesCopilot()

    return GroqSalesCopilot(groq_api_key=api_key, model_name=selected_model)



class MockSalesCopilot:

    def invoke(self, inputs: dict):
        user_input = inputs.get("input", "").lower()
        if "prévision" in user_input or "futur" in user_input or "prochain" in user_input:
            forecast = get_sales_forecast.invoke({})
            return {
                "output": f"📊 **Prévisions des ventes (30 prochains jours)** :\n{forecast}\n\n*Note : Clé GROQ_API_KEY non fournie. Réponse basée sur l'outil de prédiction.*"
            }
        elif "historique" in user_input or "passé" in user_input or "réel" in user_input:
            history = get_historical_sales.invoke({})
            return {
                "output": f"📈 **Historique des ventes récentes** :\n{history}\n\n*Note : Réponse générée depuis l'outil historique.*"
            }
        elif "métrique" in user_input or "mape" in user_input or "précision" in user_input:
            metrics = get_model_metrics.invoke({})
            return {
                "output": f"🎯 **Performance du modèle XGBoost** :\n{metrics}\n\n*Note : Métriques récupérées du fichier MLflow.*"
            }
        else:
            return {
                "output": "Bonjour ! Je suis le Copilot Ventes & Forecasting Talys. Vous pouvez me poser des questions sur les prévisions de ventes, les données historiques ou la précision du modèle XGBoost. (Ajoutez votre clé `GROQ_API_KEY` dans le fichier `.env` pour débloquer le raisonnement LLM complet !)"
            }
