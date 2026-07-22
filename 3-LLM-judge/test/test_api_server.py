from fastapi.testclient import TestClient

from api_server import create_app


def test_root_serves_frontend_html():
    app = create_app(api_key="test-key", model_path="../models/distilbert_sentinel")
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Prompt Injection Detector" in response.text
