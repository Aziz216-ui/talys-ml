"""
Tests unitaires pour les endpoints FastAPI REST.
"""

from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "online"
    assert "version" in json_data


def test_predict_sales():
    response = client.post("/api/predict", json={"horizon_days": 7})
    assert response.status_code == 200
    json_data = response.json()
    assert "predictions" in json_data
    assert len(json_data["predictions"]) <= 7


def test_copilot_chat():
    response = client.post("/api/copilot/chat", json={"message": "Quelle est la précision du modèle ?"})
    assert response.status_code == 200
    json_data = response.json()
    assert "response" in json_data
    assert len(json_data["response"]) > 0


def test_get_latest_report():
    response = client.get("/api/reports/latest")
    assert response.status_code in [200, 404]
