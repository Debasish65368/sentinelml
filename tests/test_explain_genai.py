import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import shap
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.explain as explain
import api.main as api_main
from src.explain import generate_explanation
from starlette.testclient import TestClient


def make_breakdown():
    return pd.DataFrame(
        {
            "feature": ["V14", "V10", "Amount"],
            "value": [-3.2, -2.1, 99.0],
            "shap_contribution": [4.2, 2.0, -0.5],
            "abs_contribution": [4.2, 2.0, 0.5],
        }
    )


class SuccessfulGroq:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        message = SimpleNamespace(content="V14 and V10 raised concern, while Amount reduced it slightly.")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FailingGroq:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        raise TimeoutError("simulated timeout")


def test_generate_explanation_handles_successful_response(monkeypatch):
    monkeypatch.setattr(explain, "Groq", SuccessfulGroq)

    explanation = generate_explanation(make_breakdown(), 0.91, {"V14": -3.2, "V10": -2.1})

    assert "V14" in explanation
    assert "V10" in explanation


def test_generate_explanation_handles_failed_response(monkeypatch):
    monkeypatch.setattr(explain, "Groq", FailingGroq)

    explanation = generate_explanation(make_breakdown(), 0.91, {"V14": -3.2, "V10": -2.1})

    assert isinstance(explanation, str)
    assert explanation
    assert "fallback" in explanation.lower()


# ---------------------------------------------------------------------------
# API-level integration tests for validate_explanation_accuracy wired into /explain
# ---------------------------------------------------------------------------

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
    from src.train_models import _build_xgboost
    X = pd.DataFrame(
        [[0.0] * 32, [1.0] * 32, [-1.0] * 32, [2.0] * 32, [-2.0] * 32, [3.0] * 32],
        columns=api_main.FEATURE_COLUMNS,
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], name="Class")
    model = _build_xgboost(scale_pos_weight=1.0, n_estimators=5, max_depth=2, learning_rate=0.2)
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
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_explain_valid_explanation_returns_explanation(client, monkeypatch):
    monkeypatch.setattr(api_main, "generate_explanation", lambda *args, **kwargs: "Custom LLM explanation.")
    monkeypatch.setattr(api_main, "validate_explanation_accuracy", lambda *args, **kwargs: True)

    response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "test-secret-key"})
    body = response.json()
    assert body["explanation"] == "Custom LLM explanation."
    assert "risk_score" in body


def test_explain_invalid_explanation_returns_fallback(client, monkeypatch):
    monkeypatch.setattr(api_main, "generate_explanation", lambda *args, **kwargs: "Custom LLM explanation.")
    monkeypatch.setattr(api_main, "validate_explanation_accuracy", lambda *args, **kwargs: False)

    response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "test-secret-key"})
    body = response.json()
    assert body["explanation"] != "Custom LLM explanation."
    assert "fallback" in body["explanation"].lower()


def test_explain_validation_crash_returns_fallback(client, monkeypatch):
    def fake_validate(*args, **kwargs):
        raise ValueError("Simulated validation crash")
    monkeypatch.setattr(api_main, "validate_explanation_accuracy", fake_validate)

    response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "test-secret-key"})
    body = response.json()
    assert body["explanation"] != "Custom LLM explanation."
    assert "fallback" in body["explanation"].lower()


def test_explain_and_predict_scores_are_consistent(client):
    predict_response = client.post("/predict", json=make_payload())
    predict_body = predict_response.json()

    explain_response = client.post("/explain", json=make_payload(), headers={"X-API-Key": "test-secret-key"})
    explain_body = explain_response.json()

    assert abs(predict_body["risk_score"] - explain_body["risk_score"]) < 1e-6
    assert predict_body["label"] == explain_body["label"]
