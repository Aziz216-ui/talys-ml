import os
import json
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

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


class GeminiSalesCopilot:
    """Agent Copilot natif connecté à Google Gemini via REST API Google AI Studio."""

    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.tools_map = {
            "get_sales_forecast": get_sales_forecast,
            "get_historical_sales": get_historical_sales,
            "get_model_metrics": get_model_metrics,
            "search_product_catalog": search_product_catalog,
        }

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])

        # Construction du contexte pour l'API REST Google AI Studio
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if content and not content.startswith("⚠️ Note :") and not content.startswith("*🔑 Note :"):
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})

        messages.append({"role": "user", "content": user_input})

        # Schéma des outils métier pour Gemini API
        tools_def = [
            {
                "type": "function",
                "function": {
                    "name": "get_sales_forecast",
                    "description": "Obtient les prédictions du montant des ventes pour les 30 prochains jours (XGBoost).",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_historical_sales",
                    "description": "Obtient l'historique récent des ventes réelles.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_model_metrics",
                    "description": "Obtient les métriques de performance du modèle XGBoost (MAPE, MAE, RMSE).",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_product_catalog",
                    "description": "Recherche sémantique RAG dans le catalogue produits par mot-clé.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Mot-clé ou catégorie de produit"}},
                        "required": ["query"]
                    }
                }
            }
        ]

        # Liste des identifiants de modèles Gemini actifs sur Google AI Studio
        candidate_models = [
            self.model_name,
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ]
        # Éliminer les doublons
        candidate_models = list(dict.fromkeys([m for m in candidate_models if m]))

        # URL OpenAI-Compatible de Google AI Studio avec la clé en entête x-goog-api-key
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        headers = {
            "x-goog-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for model in candidate_models:
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools_def,
                "temperature": 0.2
            }

            try:
                res = requests.post(url, headers=headers, json=payload, timeout=30)
                if res.status_code == 200:
                    res_data = res.json()
                    choice = res_data["choices"][0]["message"]
                    tools_called = []

                    if "tool_calls" in choice and choice["tool_calls"]:
                        tool_calls = choice["tool_calls"]
                        messages.append(choice)
                        for tc in tool_calls:
                            fn_name = tc["function"]["name"]
                            fn_args_str = tc["function"].get("arguments", "{}")
                            try:
                                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                            except Exception:
                                fn_args = {}
                            
                            tools_called.append(fn_name)
                            tool_fn = self.tools_map.get(fn_name)
                            if tool_fn:
                                result = tool_fn.invoke(fn_args)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": str(result)
                                })

                        payload_2 = {
                            "model": model,
                            "messages": messages,
                            "temperature": 0.2
                        }
                        res_2 = requests.post(url, headers=headers, json=payload_2, timeout=30)
                        if res_2.status_code == 200:
                            out_text = res_2.json()["choices"][0]["message"]["content"]
                            return {"output": out_text, "tools_called": tools_called}

                    out_text = choice.get("content", "")
                    if out_text:
                        return {"output": out_text, "tools_called": tools_called}

                else:
                    last_error = f"Status {res.status_code}: {res.text}"
            except Exception as err:
                last_error = str(err)

        # Deuxième approche : API REST Native generateContent de Google AI Studio
        for model in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
            native_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            native_headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            native_payload = {
                "contents": [
                    {
                        "parts": [{"text": f"{SYSTEM_PROMPT}\n\nQuestion de l'utilisateur: {user_input}"}]
                    }
                ]
            }
            try:
                n_res = requests.post(native_url, headers=native_headers, json=native_payload, timeout=15)
                if n_res.status_code == 200:
                    n_data = n_res.json()
                    candidates = n_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text_resp = parts[0].get("text", "")
                            if text_resp:
                                return {"output": text_resp, "tools_called": []}
            except Exception:
                pass

        return MockSalesCopilot().invoke(inputs)


class GroqSalesCopilot:

    def __init__(self, groq_api_key: str, model_name: str):
        self.groq_api_key = groq_api_key
        self.requested_model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.tools_map = {
            "get_sales_forecast": get_sales_forecast,
            "get_historical_sales": get_historical_sales,
            "get_model_metrics": get_model_metrics,
            "search_product_catalog": search_product_catalog,
        }

    def _get_active_models(self) -> list:
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            models = client.models.list()
            active_ids = [m.id for m in models.data if hasattr(m, "id")]
            if active_ids:
                if self.requested_model in active_ids:
                    active_ids.remove(self.requested_model)
                    return [self.requested_model] + active_ids
                return active_ids
        except Exception as e:
            print(f"Détection des modèles Groq API: {e}")
        
        return [self.requested_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

    def _get_llm(self, model: str):
        tools = list(self.tools_map.values())
        llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model,
            temperature=0.2,
        )
        return llm, llm.bind_tools(tools)

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        
        active_models = self._get_active_models()

        last_exception = None
        for model in active_models:
            if not model:
                continue
            try:
                llm, llm_with_tools = self._get_llm(model)
                
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for msg in chat_history:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if content and not content.startswith("⚠️ Note :"):
                        if role == "user":
                            messages.append(HumanMessage(content=content))
                        elif role == "assistant":
                            messages.append(AIMessage(content=content))

                messages.append(HumanMessage(content=user_input))

                tools_called = []
                response = llm_with_tools.invoke(messages)

                if hasattr(response, "tool_calls") and response.tool_calls:
                    messages.append(response)
                    for tool_call in response.tool_calls:
                        name = tool_call["name"]
                        args = tool_call["args"]
                        tools_called.append(name)
                        tool_fn = self.tools_map.get(name)
                        if tool_fn:
                            tool_result = tool_fn.invoke(args)
                            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

                    final_response = llm.invoke(messages)
                    return {"output": final_response.content, "tools_called": tools_called}

                return {"output": response.content, "tools_called": tools_called}

            except Exception as e:
                last_exception = e
                print(f"Modèle Groq '{model}' non disponible ({e}), essai du suivant...")

        return MockSalesCopilot().invoke(inputs)


def create_sales_copilot(groq_api_key: str = None, model_name: str = None):
    """Crée et initialise l'agent IA Copilot Sales avec Gemini API ou Groq API."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        return GeminiSalesCopilot(
            api_key=gemini_key,
            model_name=model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        )

    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    selected_model = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if api_key:
        return GroqSalesCopilot(groq_api_key=api_key, model_name=selected_model)

    return MockSalesCopilot()


class MockSalesCopilot:
    """Agent autonome de secours qui exécute réellement les outils et formate les réponses."""

    def invoke(self, inputs: dict):
        user_input = inputs.get("input", "")
        low_input = user_input.lower()
        tools_called = []

        if any(k in low_input for k in ["prévision", "futur", "prochain", "projeté", "chiffre d'affaires"]):
            raw = get_sales_forecast.invoke({})
            tools_called.append("get_sales_forecast")
            try:
                data = json.loads(raw)
                output = (
                    f"📊 **Prévisions du Chiffre d'Affaires ({data.get('horizon_jours', 30)} jours)** :\n\n"
                    f"- **Période** : du `{data.get('date_debut')}` au `{data.get('date_fin')}`\n"
                    f"- 💰 **Montant total projeté** : `{data.get('montant_total_predit', 0):,.2f} TND`\n"
                    f"- 📈 **Moyenne journalière** : `{data.get('moyenne_journaliere_predite', 0):,.2f} TND`\n"
                    f"- 🚀 **Pic de ventes estimé** : `{data.get('jour_ventes_max', {}).get('montant', 0):,.2f} TND` le `{data.get('jour_ventes_max', {}).get('date')}`\n"
                    f"- 📉 **Jour le plus bas** : `{data.get('jour_ventes_min', {}).get('montant', 0):,.2f} TND` le `{data.get('jour_ventes_min', {}).get('date')}`\n\n"
                    f"*Note : Analyse générée à partir des prédictions réelles XGBoost.*"
                )
            except Exception:
                output = f"📊 **Prévisions des ventes** :\n{raw}"

            return {"output": output, "tools_called": tools_called}

        elif any(k in low_input for k in ["historique", "passé", "réel", "tendance", "précédent"]):
            raw = get_historical_sales.invoke({})
            tools_called.append("get_historical_sales")
            try:
                data = json.loads(raw)
                output = (
                    f"📈 **Analyse de l'Historique Récent des Ventes ({data.get('periode_jours', 30)} derniers jours)** :\n\n"
                    f"- **Période couverte** : du `{data.get('date_debut')}` au `{data.get('date_fin')}`\n"
                    f"- 💵 **Chiffre d'affaires total réel** : `{data.get('montant_total_reel', 0):,.2f} TND`\n"
                    f"- 📊 **Moyenne quotidienne réelle** : `{data.get('moyenne_journaliere_reelle', 0):,.2f} TND`\n"
                )
            except Exception:
                output = f"📈 **Historique des ventes** :\n{raw}"

            return {"output": output, "tools_called": tools_called}

        elif any(k in low_input for k in ["métrique", "mape", "mae", "rmse", "précision", "erreur", "performance"]):
            raw = get_model_metrics.invoke({})
            tools_called.append("get_model_metrics")
            try:
                data = json.loads(raw)
                latest = data.get("dernier_reentrainement", {})
                output = (
                    f"🎯 **Performance et Précision du Modèle XGBoost** :\n\n"
                    f"- 📐 **MAE (Mean Absolute Error)** : `{latest.get('MAE', 'N/A')} TND`\n"
                    f"- 📏 **RMSE (Root Mean Squared Error)** : `{latest.get('RMSE', 'N/A')} TND`\n"
                    f"- 📊 **MAPE (Mean Absolute Percentage Error)** : `{latest.get('MAPE', 'N/A')}`\n"
                    f"- 🔄 **Nombre total de réentraînements enregistrés** : `{data.get('nombre_total_runs', 1)}`\n"
                )
            except Exception:
                output = f"🎯 **Métriques du modèle** :\n{raw}"

            return {"output": output, "tools_called": tools_called}

        elif any(k in low_input for k in ["produit", "catalogue", "catégorie", "bio", "clavier", "recherche"]):
            query_term = "Bio" if "bio" in low_input else user_input
            raw = search_product_catalog.invoke({"query": query_term})
            tools_called.append("search_product_catalog")
            return {
                "output": f"🛒 **Résultats de recherche dans le catalogue produits (RAG / Base)** :\n\n```json\n{raw}\n```",
                "tools_called": tools_called
            }

        else:
            return {
                "output": (
                    f"Bonjour ! Je suis le **Copilot IA Sales & Forecasting de Talys-Trade**.\n\n"
                    f"Voici ce que je peux analyser pour vous :\n"
                    f"- 📊 **Prévisions des ventes futures** à 30 jours (XGBoost)\n"
                    f"- 📈 **Historique des ventes réelles** et tendances\n"
                    f"- 🎯 **Précision du modèle ML** (MAPE, MAE, RMSE)\n"
                    f"- 🛒 **Recherche sémantique RAG** dans le catalogue produit\n"
                ),
                "tools_called": []
            }
