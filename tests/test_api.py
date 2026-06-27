import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_status" in response.json()
