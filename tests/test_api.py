import json
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data


def test_generate_scenarios():
    payload = {"regime": "normal", "length": 20, "num_scenarios": 1}
    resp = client.post('/scenarios/generate', json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 'scenario_id' in data
