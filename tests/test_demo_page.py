"""The demo page is HTML and does not leak the model key."""

from fastapi.testclient import TestClient

from src.api.main import app
from src.config import get_settings


def test_root_is_demo_page():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "CarePolicy" in body
    assert "/docs" in body
    assert get_settings().api_key in body
    assert "sk-" not in body
