import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.train_models as train_models
from src.train_models import train_logistic_baseline, train_random_forest, train_xgboost


EXPECTED_METRIC_KEYS = {
    "method",
    "model_name",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
}


def make_modeling_data(rows=120):
    fraud_rows = rows // 5
    normal_rows = rows - fraud_rows
    y = pd.Series([0] * normal_rows + [1] * fraud_rows, name="Class")
    signal = np.r_[np.linspace(-2.0, 0.5, normal_rows), np.linspace(0.6, 3.0, fraud_rows)]
    X = pd.DataFrame(
        {
            "V1": signal,
            "V2": signal * 0.5,
            "Amount": np.linspace(1.0, 200.0, rows),
            "hour_of_day": np.arange(rows) % 24,
            "hour_sin": np.sin(2 * np.pi * (np.arange(rows) % 24) / 24),
            "hour_cos": np.cos(2 * np.pi * (np.arange(rows) % 24) / 24),
        }
    )
    return X, y


def make_train_val_data():
    X, y = make_modeling_data()
    train_index = list(range(72)) + list(range(96, 114))
    val_index = list(range(72, 96)) + list(range(114, 120))
    return X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[val_index]


def test_train_logistic_baseline_trains_and_returns_expected_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "PROJECT_ROOT", tmp_path)
    X_train, X_val, y_train, y_val = make_train_val_data()

    model, metrics = train_logistic_baseline(X_train, y_train, X_val, y_val)

    assert EXPECTED_METRIC_KEYS.issubset(metrics)
    assert hasattr(model, "predict_proba")


def test_train_logistic_baseline_prediction_probabilities_are_in_range(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "PROJECT_ROOT", tmp_path)
    X_train, X_val, y_train, y_val = make_train_val_data()

    model, _ = train_logistic_baseline(X_train, y_train, X_val, y_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_train_random_forest_trains_and_returns_expected_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "PROJECT_ROOT", tmp_path)
    X_train, X_val, y_train, y_val = make_train_val_data()

    model, metrics = train_random_forest(X_train, y_train, X_val, y_val)

    assert EXPECTED_METRIC_KEYS.issubset(metrics)
    assert hasattr(model, "predict_proba")


def test_train_xgboost_trains_and_returns_expected_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "PROJECT_ROOT", tmp_path)
    X_train, X_val, y_train, y_val = make_train_val_data()

    model, metrics = train_xgboost(X_train, y_train, X_val, y_val)

    assert EXPECTED_METRIC_KEYS.issubset(metrics)
    assert hasattr(model, "predict_proba")
