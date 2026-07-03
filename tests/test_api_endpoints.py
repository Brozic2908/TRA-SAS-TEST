import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Test endpoint GET /api/v1/health"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_query_validation_error():
    """Test POST /api/v1/query voi cau hoi rong"""
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 400
    assert "không được để trống" in response.json()["detail"]

@patch("app.main.graph.ainvoke", new_callable=AsyncMock)
def test_query_endpoint_success(mock_ainvoke):
    """Test POST /api/v1/query thanh cong va tra ve dung schema"""
    mock_ainvoke.return_value = {
        "question": "Thủ tục hải quan?",
        "documents": ["Nội dung Điều 16 Luật Hải quan 54/2014/QH13..."],
        "citations": [
            {
                "law_number": "54/2014/QH13",
                "article_number": "Điều 16",
                "title": "Địa điểm làm thủ tục hải quan",
                "status": "con_hieu_luc",
                "superseded_by": None
            }
        ],
        "generation": "Theo Điều 16 Luật Hải quan 54/2014/QH13...",
        "search_fallback": False
    }

    payload = {
        "question": "Thủ tục hải quan ở đâu?",
        "session_id": "test-session-123"
    }

    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["session_id"] == "test-session-123"
    assert data["question"] == "Thủ tục hải quan?"
    assert len(data["citations"]) == 1
    assert data["citations"][0]["law_number"] == "54/2014/QH13"
    assert data["citations"][0]["article_number"] == "Điều 16"
    assert data["search_fallback"] is False
