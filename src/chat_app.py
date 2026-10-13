"""
Application Web de Chat Interactive Streamlit pour l'Agent IA Copilot Sales.
"""

import importlib.util
import os
import sys
import json
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ai_agent.agent import create_sales_copilot
from ai_agent.tools import get_model_metrics, load_combined_sales_df

st.set_page_config(
    page_title="Talys-Trade — Copilot IA Sales",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #38bdf8; margin-bottom: 5px; }
    .sub-header { font-size: 14px; color: #94a3b8; margin-bottom: 20px; }
    .stChatMessage { border-radius: 12px; margin-bottom: 10px; }
    .metric-card { background: #1e293b; padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); }
    .tool-badge { background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

# Session State initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis le **Copilot IA Sales & Forecasting de Talys-Trade**.\nComment puis-je vous aider aujourd'hui ? Posez-moi vos questions sur les prévisions, l'historique des ventes ou la précision des modèles ML.",
            "tools_called": [],
            "has_chart": False
        }
    ]

# Sidebar
with st.sidebar:
    st.title("🤖 Configuration & Status")
    
    # Cle Groq API
    groq_input_key = st.text_input(
        "🔑 Clé API Groq (Optionnel)",
        value="",
        type="password",
        help="Laissez vide pour utiliser GROQ_API_KEY du fichier .env ou le mode dégradé autonome."
    )
    
    active_key = groq_input_key.strip() or os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if gemini_key:
        st.success(f"🟢 Google Gemini {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')} : Actif (AI Studio)")
    elif active_key:
        st.success("🟢 Groq LLM : Actif")
    else:
        st.info("🔵 Mode Autonome Intégré")

    st.divider()

    # Metrics Section
    st.subheader("🎯 Métriques Modèle XGBoost")
    try:
        metrics_str = get_model_metrics.invoke({})
        metrics_data = json.loads(metrics_str)
        latest = metrics_data.get("dernier_reentrainement", {})
        mape_val = latest.get("MAPE", "N/A")
        mae_val = latest.get("MAE", "N/A")
        rmse_val = latest.get("RMSE", "N/A")
        
        st.markdown(f"""
        <div style="background: #1e293b; padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 13px;">Précision (MAPE)</span>
                <span style="color: #34d399; font-weight: 700; font-size: 14px;">{mape_val}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 13px;">Erreur (MAE)</span>
                <span style="color: #38bdf8; font-weight: 700; font-size: 14px;">{mae_val} TND</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8; font-size: 13px;">RMSE</span>
                <span style="color: #cbd5e1; font-weight: 600; font-size: 13px;">{rmse_val} TND</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Réentraînements enregistrés : {metrics_data.get('nombre_total_runs', 1)}")
    except Exception as e:
        st.info("Métriques MLflow indisponibles.")

    st.divider()

    # RAG Status
    st.subheader("🧠 Recherche Catalogue Produits")
    if importlib.util.find_spec("chromadb") is not None:
        st.caption("Recherche sémantique RAG (ChromaDB) : ● Active")
    else:
        st.caption("Recherche par catégorie, sous-catégorie et branche : ● Active")

    st.divider()

    # Suggestions & Quick Actions
    st.subheader("💡 Suggestions Rapides")
    if st.button("📊 Prévisions Ventes (30j)", use_container_width=True):
        st.session_state.prompt_trigger = "Quel est le chiffre d'affaires prévisionnel pour les 30 prochains jours ?"
    if st.button("📈 Graphique Réel vs Prédit", use_container_width=True):
        st.session_state.prompt_trigger = "Affiche-moi le graphique de l'historique et des prévisions de ventes."
    if st.button("🎯 Précision XGBoost", use_container_width=True):
        st.session_state.prompt_trigger = "Quelle est la précision du modèle XGBoost (MAPE et MAE) ?"
    if st.button("🛒 Produits Catégorie Bio", use_container_width=True):
        st.session_state.prompt_trigger = "Recherche les produits bio dans le catalogue"

    st.divider()
    if st.button("🗑️ Effacer la conversation", type="secondary", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation réinitialisée. Comment puis-je vous aider ?",
                "tools_called": [],
                "has_chart": False
            }
        ]
        st.rerun()


def copilot_page():
    # Header
    st.markdown('<div class="main-header">💬 Agent IA Copilot Sales & Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Assistant décisionnel intelligent connecté aux pipelines MLOps XGBoost et au catalogue produits RAG.</div>', unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        
            # Tools called badge
            tools_called = msg.get("tools_called", [])
            if tools_called:
                with st.expander("🔍 Outils exécutés", expanded=False):
                    for t in tools_called:
                        st.markdown(f'<span class="tool-badge">⚡ {t}</span>', unsafe_allow_html=True)
        
            # Chart display if flagged
            if msg.get("has_chart"):
                df_comb = load_combined_sales_df()
                if not df_comb.empty:
                    st.subheader("📊 Graphique Évolutif des Ventes (Réel & Prédit)")
                    chart_data = df_comb.pivot(index="date_transaction", columns="Type", values="Montant")
                    st.line_chart(chart_data, color=["#6366f1", "#38bdf8"])

    # Handle Inputs (either from chat_input or quick action buttons)
    prompt = st.chat_input("Posez votre question sur les ventes, prédictions ou produits...")

    if "prompt_trigger" in st.session_state and st.session_state.prompt_trigger:
        prompt = st.session_state.prompt_trigger
        del st.session_state["prompt_trigger"]

    if prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt, "tools_called": [], "has_chart": False})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Raisonnement de l'Agent IA & Appel d'outils..."):
                try:
                    agent = create_sales_copilot(groq_api_key=active_key)
                
                    # Pass chat history to maintain multi-turn memory
                    response = agent.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.messages[:-1]
                    })
                
                    output_text = response.get("output", "Aucune réponse générée.")
                    tools_called = response.get("tools_called", [])
                
                except Exception as e:
                    output_text = f"Erreur lors du traitement de l'Agent IA : {e}"
                    tools_called = []

                # Check if prompt or response requests chart rendering
                low_prompt = prompt.lower()
                has_chart = any(k in low_prompt for k in ["graphique", "courbe", "tendance", "visualis", "chart", "prévision", "historique"])

                st.markdown(output_text)
            
                if tools_called:
                    with st.expander("🔍 Outils exécutés", expanded=False):
                        for t in tools_called:
                            st.markdown(f'<span class="tool-badge">⚡ {t}</span>', unsafe_allow_html=True)
            
                if has_chart:
                    df_comb = load_combined_sales_df()
                    if not df_comb.empty:
                        st.subheader("📊 Graphique Évolutif des Ventes (Réel & Prédit)")
                        chart_data = df_comb.pivot(index="date_transaction", columns="Type", values="Montant")
                        st.line_chart(chart_data, color=["#6366f1", "#38bdf8"])

                # Save assistant message to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output_text,
                    "tools_called": tools_called,
                    "has_chart": has_chart
                })


def dashboards_page():
    st.markdown('<div class="main-header">📊 Tableaux de bord Power BI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ventes, produits, clients, prévisions, suivi MLOps et insights IA, alimentés par le même Data Mart MySQL que le Copilot.</div>', unsafe_allow_html=True)

    embed_url = os.getenv("POWERBI_EMBED_URL", "").strip()
    if embed_url:
        st.iframe(embed_url, height=760)
        st.link_button("Ouvrir les tableaux de bord en plein écran", embed_url, icon=":material/open_in_new:")
    else:
        st.info(
            "Les tableaux de bord apparaîtront ici une fois le rapport publié avec « Publier sur le web » "
            "dans Power BI Service et son lien ajouté dans la variable `POWERBI_EMBED_URL`.",
            icon=":material/info:",
        )


page = st.navigation(
    [
        st.Page(copilot_page, title="Copilot IA", icon=":material/smart_toy:", url_path="copilot", default=True),
        st.Page(dashboards_page, title="Tableaux de bord", icon=":material/bar_chart:", url_path="tableaux-de-bord"),
    ],
    position="top",
)
page.run()
