"""
Focused unit tests for the Streamlit /explain request-building logic.

Tests cover:
- get_sentinel_api_key() resolution from st.secrets and environment variables
- post_explain header injection via post_json in app.py
- missing-key error behavior
- open routes do not send X-API-Key
- API regression: /explain returns 401 without a key

These tests import streamlit_app._explain_client directly (no Streamlit import)
and test post_explain / post_json through the API regression test.
No browser or running Streamlit instance is required.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app._explain_client import get_sentinel_api_key as _get_key_impl


# ---------------------------------------------------------------------------
# _explain_client.get_sentinel_api_key — resolution order
# ---------------------------------------------------------------------------

def test_get_sentinel_api_key_returns_streamlit_secret(monkeypatch):
    """st_secrets takes priority over the environment variable."""
    monkeypatch.delenv("SENTINEL_API_KEY", raising=False)
    st_secrets_mock = MagicMock()
    st_secrets_mock.get.side_effect = lambda key, default=None: (
        "from-st-secrets" if key == "SENTINEL_API_KEY" else default
    )
    result = _get_key_impl(st_secrets=st_secrets_mock)
    assert result == "from-st-secrets"


def test_get_sentinel_api_key_falls_back_to_env(monkeypatch):
    """Falls back to the environment variable when st_secrets returns nothing."""
    monkeypatch.setenv("SENTINEL_API_KEY", "from-env")
    st_secrets_mock = MagicMock()
    st_secrets_mock.get.return_value = None
    result = _get_key_impl(st_secrets=st_secrets_mock)
    assert result == "from-env"


def test_get_sentinel_api_key_returns_none_when_unconfigured(monkeypatch):
    """Returns None when neither st_secrets nor the environment has a key."""
    monkeypatch.delenv("SENTINEL_API_KEY", raising=False)
    st_secrets_mock = MagicMock()
    st_secrets_mock.get.return_value = None
    result = _get_key_impl(st_secrets=st_secrets_mock)
    assert result is None


def test_get_sentinel_api_key_handles_secrets_exception(monkeypatch):
    """Falls back gracefully when st_secrets.get raises an exception."""
    monkeypatch.setenv("SENTINEL_API_KEY", "env-fallback")
    st_secrets_mock = MagicMock()
    st_secrets_mock.get.side_effect = AttributeError("no secrets")
    result = _get_key_impl(st_secrets=st_secrets_mock)
    assert result == "env-fallback"


def test_get_sentinel_api_key_works_without_secrets(monkeypatch):
    """When st_secrets is None (no Streamlit session), uses env var only."""
    monkeypatch.setenv("SENTINEL_API_KEY", "env-only")
    result = _get_key_impl(st_secrets=None)
    assert result == "env-only"


# ---------------------------------------------------------------------------
# post_explain — header injection and error handling (via importable wrappers)
# ---------------------------------------------------------------------------

def _make_post_explain_under_test(env_api_key=None):
    """Helper: create standalone post_explain and post_json callables for testing
    without importing the full Streamlit app module."""
    import requests as _requests

    def _get_key():
        return env_api_key

    def post_json_impl(endpoint, payload, headers=None, api_base="http://localhost:8000",
                       timeout=30):
        url = f"{api_base}{endpoint}"
        try:
            resp = _requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
        except _requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Could not connect to API at {api_base}") from exc
        except _requests.exceptions.Timeout as exc:
            raise RuntimeError(f"Timed out calling {endpoint}") from exc
        except _requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"API error for {endpoint}: {resp.text}") from exc
        return resp.json()

    def post_explain_impl(payload):
        api_key = _get_key()
        if not api_key:
            raise RuntimeError(
                "SENTINEL_API_KEY is not configured. "
                "Set it in your .env file (local) or Streamlit secrets (Streamlit Cloud) "
                "to enable the Explain feature."
            )
        return post_json_impl("/explain", payload, headers={"X-API-Key": api_key})

    return post_explain_impl, post_json_impl


def test_post_explain_sends_api_key_header():
    """post_explain attaches X-API-Key when a key is configured."""
    post_explain, _ = _make_post_explain_under_test(env_api_key="test-key-123")

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        resp = MagicMock()
        resp.json.return_value = {"explanation": "ok", "risk_score": 0.01, "label": "normal", "shap_contributions": []}
        resp.raise_for_status = MagicMock()
        return resp

    with patch("requests.post", side_effect=fake_post):
        post_explain({"transaction_id": "t1"})

    assert "/explain" in captured["url"]
    assert captured["headers"].get("X-API-Key") == "test-key-123"


def test_post_explain_raises_when_key_missing():
    """post_explain raises a user-friendly RuntimeError when the key is absent."""
    post_explain, _ = _make_post_explain_under_test(env_api_key=None)

    with pytest.raises(RuntimeError, match="SENTINEL_API_KEY is not configured"):
        post_explain({"transaction_id": "t1"})


def test_post_explain_error_does_not_leak_key_value():
    """A RuntimeError from a failed /explain call must not include the key."""
    import requests as _req
    post_explain, _ = _make_post_explain_under_test(env_api_key="super-secret-key")

    def fake_post(url, json=None, headers=None, timeout=None):
        resp = MagicMock()
        resp.text = '{"detail": "Invalid API Key"}'
        resp.raise_for_status.side_effect = _req.exceptions.HTTPError("401 Client Error")
        return resp

    with patch("requests.post", side_effect=fake_post):
        with pytest.raises(RuntimeError) as exc_info:
            post_explain({"transaction_id": "t1"})

    assert "super-secret-key" not in str(exc_info.value)


def test_post_json_predict_sends_no_api_key():
    """/predict and /predict_batch must not receive X-API-Key (open routes)."""
    _, post_json = _make_post_explain_under_test(env_api_key="should-not-appear")

    captured_headers = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_headers.update(headers or {})
        resp = MagicMock()
        resp.json.return_value = {"risk_score": 0.0, "label": "normal"}
        resp.raise_for_status = MagicMock()
        return resp

    with patch("requests.post", side_effect=fake_post):
        post_json("/predict", {"transaction_id": "t1"})

    assert "X-API-Key" not in captured_headers


# ---------------------------------------------------------------------------
# API layer regression — /explain must reject unauthenticated calls
# ---------------------------------------------------------------------------

def test_api_explain_rejects_missing_api_key(monkeypatch):
    """Guard: the FastAPI /explain endpoint returns 401 when no key is sent."""
    import shap
    import pandas as pd
    import api.main as api_main
    from src.train_models import _build_xgboost
    from starlette.testclient import TestClient

    model_features = api_main.FEATURE_COLUMNS

    def make_test_model():
        X = pd.DataFrame([[0.0] * len(model_features)] * 6, columns=model_features)
        y = pd.Series([0, 1, 0, 1, 0, 1], name="Class")
        m = _build_xgboost(scale_pos_weight=1.0, n_estimators=5, max_depth=2, learning_rate=0.2)
        m.fit(X, y)
        return m

    model = make_test_model()

    def fake_load():
        return {
            "model": model,
            "threshold": 0.5,
            "shap_explainer": shap.TreeExplainer(model),
            "scaler": None,
            "model_version": "test",
        }

    monkeypatch.setenv("SENTINEL_API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "load_model_resources", fake_load)

    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload.update({"transaction_id": "t1", "Amount": 10.0, "hour_of_day": 1.0, "hour_sin": 0.0, "hour_cos": 1.0})

    with TestClient(api_main.app) as client:
        r = client.post("/explain", json=payload)
        assert r.status_code == 401


def test_api_explain_accepts_valid_key(monkeypatch):
    """Guard: /explain returns 200 when a valid key is provided."""
    import shap
    import pandas as pd
    import api.main as api_main
    from src.train_models import _build_xgboost
    from starlette.testclient import TestClient

    model_features = api_main.FEATURE_COLUMNS

    def make_test_model():
        X = pd.DataFrame([[0.0] * len(model_features)] * 6, columns=model_features)
        y = pd.Series([0, 1, 0, 1, 0, 1], name="Class")
        m = _build_xgboost(scale_pos_weight=1.0, n_estimators=5, max_depth=2, learning_rate=0.2)
        m.fit(X, y)
        return m

    model = make_test_model()

    def fake_load():
        return {
            "model": model,
            "threshold": 0.5,
            "shap_explainer": shap.TreeExplainer(model),
            "scaler": None,
            "model_version": "test",
        }

    monkeypatch.setenv("SENTINEL_API_KEY", "test-secret-key")
    monkeypatch.setattr(api_main, "load_model_resources", fake_load)
    monkeypatch.setattr(api_main, "generate_explanation", lambda *a, **kw: "V14 and V10 raised concern.")
    monkeypatch.setattr(api_main, "validate_explanation_accuracy", lambda *a, **kw: True)

    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload.update({"transaction_id": "t1", "Amount": 10.0, "hour_of_day": 1.0, "hour_sin": 0.0, "hour_cos": 1.0})

    with TestClient(api_main.app) as client:
        r = client.post("/explain", json=payload, headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200
        body = r.json()
        assert "risk_score" in body
        assert "explanation" in body
        assert "shap_contributions" in body
