import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.explain as explain
from src.explain import generate_explanation


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
