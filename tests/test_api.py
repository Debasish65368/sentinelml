import sys
from pathlib import Path

import pandas as pd
import pytest
import shap
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_main
from src.train_models import _build_xgboost


def make_payload():
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload.update(
        {
            "transaction_id": "test-transaction",
            "Amount": 42.0,
            "hour_of_day": 12.0,
            "hour_sin": 0.0,
            "hour_cos": -1.0,
        }
    )
    return payload


def make_test_model():
    X = pd.DataFrame(
        [
            [0.0] * 32,
            [1.0] * 32,
            [-1.0] * 32,
            [2.0] * 32,
            [-2.0] * 32,
            [3.0] * 32,
        ],
        columns=api_main.FEATURE_COLUMNS,
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], name="Class")
    model = _build_xgboost(
        scale_pos_weight=1.0,
        n_estimators=5,
        max_depth=2,
        learning_rate=0.2,
    )
    model.fit(X, y)
    return model


@pytest.fixture
def client(monkeypatch):
    model = make_test_model()

    def fake_load_model_resources():
        return {
            "model": model,
            "threshold": 0.5,
            "shap_explainer": shap.TreeExplainer(model),
            "scaler": None,
            "model_version": "test-model",
        }

    monkeypatch.setenv("SENTINEL_API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "load_model_resources", fake_load_model_resources)
    monkeypatch.setattr(api_main, "generate_explanation", lambda *args, **kwargs: "Mock explanation.")
    monkeypatch.setattr(api_main, "validate_explanation_accuracy", lambda *args, **kwargs: True)
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_works(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["shap_loaded"] is True


def test_predict_returns_expected_schema(client):
    response = client.post("/predict", json=make_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"risk_score", "label", "transaction_id"}
    assert 0 <= body["risk_score"] <= 1
    assert body["label"] in {"fraud", "normal"}
    assert body["transaction_id"] == "test-transaction"


def test_predict_batch_returns_predictions_in_order(client):
    first_payload = make_payload()
    first_payload["transaction_id"] = "first"
    first_payload["V1"] = 0.0
    second_payload = make_payload()
    second_payload["transaction_id"] = "second"
    second_payload["V1"] = 2.0

    response = client.post(
        "/predict_batch",
        json={"transactions": [first_payload, second_payload]},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predictions"}
    assert len(body["predictions"]) == 2
    assert [prediction["transaction_id"] for prediction in body["predictions"]] == ["first", "second"]
    for prediction in body["predictions"]:
        assert set(prediction) == {"risk_score", "label", "transaction_id"}
        assert 0 <= prediction["risk_score"] <= 1
        assert prediction["label"] in {"fraud", "normal"}


def test_predict_batch_empty_batch_returns_400(client):
    response = client.post("/predict_batch", json={"transactions": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "No transactions provided."


def test_explain_returns_expected_schema(client):
    response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "test-secret-key"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"risk_score", "label", "shap_contributions", "explanation"}
    assert 0 <= body["risk_score"] <= 1
    assert body["label"] in {"fraud", "normal"}
    assert len(body["shap_contributions"]) == len(api_main.FEATURE_COLUMNS)
    assert {"feature", "value", "contribution"}.issubset(body["shap_contributions"][0])
    assert body["explanation"] == "Mock explanation."


def test_explain_rejects_missing_api_key(client):
    response = client.post("/explain", json=make_payload())
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing API Key"


def test_explain_rejects_incorrect_api_key(client):
    response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API Key"


def test_cors_headers(client):
    response = client.options("/explain", headers={
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_invalid_input_returns_422(client):
    payload = make_payload()
    payload.pop("V1")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
