import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import engineer_features, remove_duplicates, split_data


def make_sample_df(normal_rows=100, fraud_rows=20):
    rows = normal_rows + fraud_rows
    data = {
        "Time": np.arange(rows, dtype=float) * 1_000,
        "Amount": np.linspace(1.0, 250.0, rows),
        "Class": [0] * normal_rows + [1] * fraud_rows,
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.linspace(-1.0, 1.0, rows) + i

    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    return pd.DataFrame(data, columns=columns)


def test_engineer_features_output_has_no_nans():
    engineered_df, _, _ = engineer_features(make_sample_df())

    assert not engineered_df.isna().any().any()


def test_engineer_features_output_has_expected_columns():
    engineered_df, feature_columns, target_col = engineer_features(make_sample_df())
    expected_features = [f"V{i}" for i in range(1, 29)] + [
        "Amount",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
    ]

    assert "Time" not in engineered_df.columns
    assert "hour_sin" in engineered_df.columns
    assert "hour_cos" in engineered_df.columns
    assert target_col == "Class"
    assert feature_columns == expected_features


def test_hour_cyclical_features_are_between_minus_one_and_one():
    engineered_df, _, _ = engineer_features(make_sample_df())

    assert engineered_df["hour_sin"].between(-1, 1).all()
    assert engineered_df["hour_cos"].between(-1, 1).all()


def test_split_data_splits_sum_to_original_row_count():
    engineered_df, _, target_col = engineer_features(make_sample_df())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(engineered_df, target_col=target_col)

    split_rows = len(X_train) + len(X_val) + len(X_test)
    split_targets = len(y_train) + len(y_val) + len(y_test)

    assert split_rows == len(engineered_df)
    assert split_targets == len(engineered_df)


def test_split_data_preserves_approximate_fraud_ratio():
    engineered_df, _, target_col = engineer_features(make_sample_df())
    _, _, _, y_train, y_val, y_test = split_data(engineered_df, target_col=target_col)
    full_ratio = engineered_df[target_col].mean()

    for y_split in [y_train, y_val, y_test]:
        assert abs(y_split.mean() - full_ratio) <= 0.03


def test_remove_duplicates_reduces_row_count_when_duplicates_exist():
    df = make_sample_df(normal_rows=20, fraud_rows=10)
    df_with_duplicates = pd.concat([df, df.iloc[[0, 5, 25]]], ignore_index=True)

    deduplicated_df = remove_duplicates(df_with_duplicates)

    assert len(deduplicated_df) == len(df)
    assert len(deduplicated_df) < len(df_with_duplicates)
