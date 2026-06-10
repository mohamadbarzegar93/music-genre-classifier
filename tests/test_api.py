import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_valid_file():
    with open("tests/fixtures/blues.00000.wav", "rb") as f:
        response = client.post("/predict", files={"file":f})
    assert response.status_code == 200
    data = response.json()
    assert "genre" in data
    assert "confidence" in data
    assert 0<=data["confidence"]<=100
    assert data["genre"] in [
        "blues", "classical", "country", "disco",
        "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    ]


def test_predict_invalid_format():
    fake_file = io.BytesIO(b"not audio data")
    response = client.post(
        "/predict",
        files = {"file": ("test.txt", fake_file, "text/plain")}
    )
    assert response.status_code ==500
