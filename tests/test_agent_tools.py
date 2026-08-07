"""
Tests unitaires pour les outils métier de l'Agent IA Copilot Sales.
"""

import json
import pytest
from src.ai_agent.tools import (
    get_sales_forecast,
    get_historical_sales,
    get_model_metrics,
    search_product_catalog,
)


def test_get_sales_forecast():
    result = get_sales_forecast.invoke({"horizon_days": 10})
    assert isinstance(result, str)
    data = json.loads(result)
    assert "montant_total_predit" in data
    assert "horizon_jours" in data


def test_get_historical_sales():
    result = get_historical_sales.invoke({"days": 15})
    assert isinstance(result, str)
    data = json.loads(result)
    assert "montant_total_reel" in data


def test_get_model_metrics():
    result = get_model_metrics.invoke({})
    assert isinstance(result, str)
    data = json.loads(result)
    assert "dernier_reentrainement" in data


def test_search_product_catalog():
    result = search_product_catalog.invoke({"query": "a"})
    assert isinstance(result, str)
