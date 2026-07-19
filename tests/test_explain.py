import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import compute_shap_values


def make_tree_data(rows=80, features=6):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"feature_{i}" for i in range(features)],
    )
    y = ((X["feature_0"] + X["feature_1"] * 0.5) > 0).astype(int)
    return X, y


def test_compute_shap_values_has_expected_shape():
    X, y = make_tree_data()
    model = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.2,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    shap_values = compute_shap_values(model, X.head(12))

    assert shap_values.shape == (12, X.shape[1])


def test_compute_shap_values_does_not_error_on_small_sample():
    X, y = make_tree_data()
    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.2,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    shap_values = compute_shap_values(model, X.head(3))

    assert np.isfinite(shap_values).all()
