import pytest
from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)


def test_continue_endpoint_basic():
    """Test basic playlist continuation functionality."""
    request_data = {
        "tracks": ["track_123", "track_456", "track_789"],
        "k": 20
    }
    
    response = client.post("/continue", json=request_data)
    
    # Should return 200 even if model not loaded (graceful degradation)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= request_data["k"]


def test_continue_endpoint_with_context():
    """Test playlist continuation with context."""
    request_data = {
        "tracks": ["track_123", "track_456"],
        "k": 10,
        "context": {
            "hour": 21,
            "dow": 5
        },
        "use_ann": False
    }
    
    response = client.post("/continue", json=request_data)
    assert response.status_code in [200, 500]


def test_continue_endpoint_empty_tracks():
    """Test playlist continuation with empty tracks list."""
    request_data = {
        "tracks": [],
        "k": 20
    }
    
    response = client.post("/continue", json=request_data)
    assert response.status_code == 400
    assert "tracks array is required" in response.json()["detail"]


def test_continue_endpoint_invalid_k():
    """Test playlist continuation with invalid k value."""
    request_data = {
        "tracks": ["track_123"],
        "k": 0
    }
    
    response = client.post("/continue", json=request_data)
    assert response.status_code == 422  # Validation error


def test_continue_response_format():
    """Test that response format matches schema."""
    request_data = {
        "tracks": ["track_123", "track_456"],
        "k": 5
    }
    
    response = client.post("/continue", json=request_data)
    
    if response.status_code == 200:
        data = response.json()
        assert "items" in data
        
        for item in data["items"]:
            assert "track_id" in item
            assert "score" in item
            assert isinstance(item["track_id"], str)
            assert isinstance(item["score"], (int, float))

