from pathlib import Path
import os
from time import perf_counter

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRIC_COLUMNS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "inference_time_seconds",
]


def load_processed_splits(processed_dir=PROCESSED_DATA_DIR):
    """Load processed train/validation/test splits from CSV files."""
    processed_dir = Path(processed_dir)
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_val = pd.read_csv(processed_dir / "X_val.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["Class"]
    y_val = pd.read_csv(processed_dir / "y_val.csv")["Class"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["Class"]
    return X_train, X_val, X_test, y_train, y_val, y_test


def _configure_mlflow():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("model_training")


def _build_logistic_regression(method):
    class_weight = "balanced" if method == "class_weight" else None
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=5_000,
        solver="lbfgs",
        random_state=42,
    )


def _prepare_training_data(X_train, y_train, method):
    if method == "class_weight":
        return X_train, y_train, {"resampled_train_rows": len(y_train)}

    if method == "smote":
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled, {"resampled_train_rows": len(y_resampled)}

    raise ValueError("method must be either 'class_weight' or 'smote'")


def _evaluate_predictions(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _predict_and_evaluate(model, X_val, y_val):
    start_time = perf_counter()
    y_val_proba = model.predict_proba(X_val)[:, 1]
    inference_time = perf_counter() - start_time
    metrics = _evaluate_predictions(y_val, y_val_proba)
    metrics["inference_time_seconds"] = inference_time
    return metrics


def _print_results(method, metrics):
    print(f"=== {metrics.get('model_name', 'Model')}: {method} ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"Precision @ 0.5: {metrics['precision']:.6f}")
    print(f"Recall @ 0.5: {metrics['recall']:.6f}")
    print(f"F1 @ 0.5: {metrics['f1']:.6f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )


def _log_metrics(metrics):
    for metric_name, metric_value in metrics.items():
        if metric_name not in {"method", "model_name"}:
            mlflow.log_metric(metric_name, metric_value)



def select_optimal_threshold(y_true, y_proba):
    from sklearn.metrics import f1_score
    import numpy as np
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.5, 1.0, 0.01):
        preds = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return round(float(best_threshold), 2)

def train_logistic_baseline(X_train, y_train, X_val, y_val, method="class_weight"):
    """Train and evaluate a Logistic Regression fraud baseline."""
    _configure_mlflow()
    X_fit, y_fit, fit_info = _prepare_training_data(X_train, y_train, method)
    model = _build_logistic_regression(method)

    with mlflow.start_run(run_name=f"logistic_{method}"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("method", method)
        mlflow.log_param("threshold", 0.5)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("validation_rows", len(y_val))
        mlflow.log_param("resampled_train_rows", fit_info["resampled_train_rows"])

        model.fit(X_fit, y_fit)
        metrics = _predict_and_evaluate(model, X_val, y_val)
        metrics["method"] = method
        metrics["model_name"] = "Logistic Regression"

        _log_metrics(metrics)

        mlflow.sklearn.log_model(model, name="model")

    _print_results(method, metrics)
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val, class_weight="balanced"):
    """Train and evaluate a class-weighted Random Forest baseline."""
    _configure_mlflow()
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )

    with mlflow.start_run(run_name="random_forest_class_weight"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("validation_rows", len(y_val))

        model.fit(X_train, y_train)
        metrics = _predict_and_evaluate(model, X_val, y_val)
        metrics["method"] = "class_weight"
        metrics["model_name"] = "Random Forest"

        _log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

    _print_results("class_weight", metrics)
    return model, metrics


def _compute_scale_pos_weight(y_train):
    class_counts = y_train.value_counts()
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))
    if positive_count == 0:
        raise ValueError("Cannot compute scale_pos_weight because y_train has no Class=1 rows.")
    return negative_count / positive_count


def _build_xgboost(scale_pos_weight, **params):
    defaults = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
    }
    defaults.update(params)
    return XGBClassifier(
        **defaults,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight=None):
    """Train and evaluate an XGBoost baseline with imbalance weighting."""
    _configure_mlflow()
    if scale_pos_weight is None:
        scale_pos_weight = _compute_scale_pos_weight(y_train)
    model = _build_xgboost(scale_pos_weight)

    with mlflow.start_run(run_name="xgboost_scale_pos_weight"):
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("subsample", model.subsample)
        mlflow.log_param("colsample_bytree", model.colsample_bytree)
        mlflow.log_param("min_child_weight", model.min_child_weight)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("validation_rows", len(y_val))

        model.fit(X_train, y_train)
        metrics = _predict_and_evaluate(model, X_val, y_val)
        metrics["method"] = "scale_pos_weight"
        metrics["model_name"] = "XGBoost"

        _log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

    _print_results("scale_pos_weight", metrics)
    return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train and evaluate a LightGBM baseline with imbalance weighting."""
    _configure_mlflow()
    computed_scale_pos_weight = _compute_scale_pos_weight(y_train)
    model = LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        max_depth=6,
        learning_rate=0.04,
        n_estimators=600,
        num_leaves=31,
        min_child_weight=8,
        subsample=0.865,
        colsample_bytree=0.996,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    with mlflow.start_run(run_name="lightgbm_is_unbalance"):
        mlflow.log_param("model_type", "LGBMClassifier")
        mlflow.log_param("imbalance_method", "is_unbalance")
        mlflow.log_param("is_unbalance", True)
        mlflow.log_param("computed_scale_pos_weight_reference", computed_scale_pos_weight)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("num_leaves", model.num_leaves)
        mlflow.log_param("min_child_weight", model.min_child_weight)
        mlflow.log_param("subsample", model.subsample)
        mlflow.log_param("colsample_bytree", model.colsample_bytree)
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("validation_rows", len(y_val))

        model.fit(X_train, y_train)
        metrics = _predict_and_evaluate(model, X_val, y_val)
        metrics["method"] = "is_unbalance"
        metrics["model_name"] = "LightGBM"

        _log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    _print_results("is_unbalance", metrics)
    return model, metrics


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=30):
    """Tune XGBoost hyperparameters with Optuna, optimizing validation PR-AUC."""
    _configure_mlflow()
    scale_pos_weight = _compute_scale_pos_weight(y_train)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = _build_xgboost(scale_pos_weight, **params)
        model.fit(X_train, y_train)
        metrics = _predict_and_evaluate(model, X_val, y_val)

        with mlflow.start_run(run_name=f"xgboost_trial_{trial.number}", nested=True):
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            _log_metrics(metrics)

        return metrics["pr_auc"]

    with mlflow.start_run(run_name="xgboost_optuna_tuning"):
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("objective", "validation_pr_auc")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_pr_auc = study.best_value
        best_model = _build_xgboost(scale_pos_weight, **best_params)
        best_model.fit(X_train, y_train)
        best_metrics = _predict_and_evaluate(best_model, X_val, y_val)
        best_metrics["method"] = "optuna_tuned_scale_pos_weight"
        best_metrics["model_name"] = "XGBoost Tuned"

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        _log_metrics(best_metrics)
        mlflow.log_metric("best_pr_auc", best_pr_auc)
        mlflow.xgboost.log_model(best_model, name="model")

    _print_results("optuna_tuned_scale_pos_weight", best_metrics)
    return best_model, best_params, best_pr_auc


def compare_imbalance_methods(X_train, y_train, X_val, y_val):
    """Train both imbalance strategies and print a side-by-side comparison."""
    results = {}
    for method in ["class_weight", "smote"]:
        model, metrics = train_logistic_baseline(X_train, y_train, X_val, y_val, method=method)
        results[method] = {"model": model, "metrics": metrics}

    comparison = pd.DataFrame(
        {method: result["metrics"] for method, result in results.items()}
    ).T
    comparison = comparison[METRIC_COLUMNS]

    print("=== Imbalance Method Comparison ===")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))
    return comparison, results


def compare_all_models(X_train, y_train, X_val, y_val):
    """Train all baseline/tree models and compare validation metrics."""
    results = {}

    logistic_model, logistic_metrics = train_logistic_baseline(
        X_train, y_train, X_val, y_val, method="class_weight"
    )
    results["logistic_class_weight"] = {"model": logistic_model, "metrics": logistic_metrics}

    random_forest_model, random_forest_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    results["random_forest_class_weight"] = {
        "model": random_forest_model,
        "metrics": random_forest_metrics,
    }

    xgboost_model, xgboost_metrics = train_xgboost(X_train, y_train, X_val, y_val)
    results["xgboost_scale_pos_weight"] = {"model": xgboost_model, "metrics": xgboost_metrics}

    tuned_model, tuned_params, tuned_pr_auc = tune_xgboost(X_train, y_train, X_val, y_val)
    tuned_metrics = _predict_and_evaluate(tuned_model, X_val, y_val)
    tuned_metrics["method"] = "optuna_tuned_scale_pos_weight"
    tuned_metrics["model_name"] = "XGBoost Tuned"
    tuned_metrics["best_pr_auc"] = tuned_pr_auc
    results["xgboost_tuned"] = {
        "model": tuned_model,
        "metrics": tuned_metrics,
        "best_params": tuned_params,
    }

    lightgbm_model, lightgbm_metrics = train_lightgbm(X_train, y_train, X_val, y_val)
    results["lightgbm_is_unbalance"] = {
        "model": lightgbm_model,
        "metrics": lightgbm_metrics,
    }

    comparison = pd.DataFrame(
        {model_name: result["metrics"] for model_name, result in results.items()}
    ).T
    comparison = comparison[METRIC_COLUMNS].sort_values("pr_auc", ascending=False)

    print("=== All Model Comparison (sorted by PR-AUC) ===")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))
    return comparison, results


def compute_curve_data(model, X, y):
    """Return ROC and precision-recall curve data for a fitted classifier."""
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    return {
        "y_proba": y_proba,
        "roc": {"fpr": fpr, "tpr": tpr},
        "precision_recall": {"precision": precision, "recall": recall},
    }
