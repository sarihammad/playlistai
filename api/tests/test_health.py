import pytest
from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint returns 200."""
    response = client.get("/healthz")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "cache_connected" in data


def test_metrics_endpoint():
    """Test metrics endpoint returns 200."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "request_duration_ms" in response.text


def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "PlayListAI"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data

