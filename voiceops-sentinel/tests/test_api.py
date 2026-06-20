from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_dashboard():
    """Verify that GET / returns the dashboard HTML successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "VoiceOps Sentinel" in response.text
    assert "Ingest Audio Call" in response.text


def test_get_health():
    """Verify that GET /health returns the correct JSON health check payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "voiceops-sentinel"
    assert "asr_backend" in response.json()
