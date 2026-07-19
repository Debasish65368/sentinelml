from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from xgboost import XGBClassifier

from api.schemas import ExplainRequest, ExplainResponse, PredictResponse, TransactionInput
from src.explain import explain_single_prediction, generate_explanation
from src.train_models import _build_xgboost, _compute_scale_pos_weight, load_processed_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TUNED_MODEL_PATH = MODEL_DIR / "tuned_xgboost.json"
AUTOENCODER_SCALER_PATH = MODEL_DIR / "autoencoder_scaler.joblib"
MODEL_VERSION = "xgboost_tuned_experiment_03"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "Amount",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
]
TUNED_XGBOOST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.04087659627832751,
    "n_estimators": 600,
    "subsample": 0.8654555403181324,
    "colsample_bytree": 0.9958149399088472,
    "min_child_weight": 8,
}


def _request_to_frame(transaction: TransactionInput) -> pd.DataFrame:
    payload = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
    values = {feature: payload[feature] for feature in FEATURE_COLUMNS}
    row = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    if row.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {row.shape[1]}.")
    return row


def _predict_probability(model, row):
    return float(model.predict_proba(row)[:, 1][0])


def _label_from_probability(probability):
    return "fraud" if probability >= 0.5 else "normal"


def load_tuned_xgboost_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if TUNED_MODEL_PATH.exists():
        model = XGBClassifier()
        model.load_model(TUNED_MODEL_PATH)
        print(f"Loaded tuned XGBoost model from {TUNED_MODEL_PATH}")
        return model

    print("No cached tuned XGBoost model found; training Experiment 03 tuned model from processed data.")
    X_train, _, _, y_train, _, _ = load_processed_splits()
    scale_pos_weight = _compute_scale_pos_weight(y_train)
    model = _build_xgboost(scale_pos_weight, **TUNED_XGBOOST_PARAMS)
    model.fit(X_train[FEATURE_COLUMNS], y_train)
    model.save_model(TUNED_MODEL_PATH)
    print(f"Saved tuned XGBoost model to {TUNED_MODEL_PATH}")
    return model


def load_model_resources():
    model = load_tuned_xgboost_model()
    shap_explainer = shap.TreeExplainer(model)
    scaler = joblib.load(AUTOENCODER_SCALER_PATH) if AUTOENCODER_SCALER_PATH.exists() else None
    return {
        "model": model,
        "shap_explainer": shap_explainer,
        "scaler": scaler,
        "model_version": MODEL_VERSION,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = load_model_resources()
    app.state.model = resources["model"]
    app.state.shap_explainer = resources["shap_explainer"]
    app.state.scaler = resources["scaler"]
    app.state.model_version = resources["model_version"]
    app.state.ready = True
    yield


app = FastAPI(title="SentinelML Fraud API", version="0.1.0", lifespan=lifespan)


def _require_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok" if getattr(request.app.state, "ready", False) else "starting",
        "model_version": getattr(request.app.state, "model_version", MODEL_VERSION),
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "shap_loaded": getattr(request.app.state, "shap_explainer", None) is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: TransactionInput, request: Request):
    model = _require_model(request)
    try:
        row = _request_to_frame(transaction)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = perf_counter()
    risk_score = _predict_probability(model, row)
    elapsed_ms = (perf_counter() - start_time) * 1000
    print(f"/predict latency_ms={elapsed_ms:.3f}")

    return PredictResponse(
        risk_score=risk_score,
        label=_label_from_probability(risk_score),
        transaction_id=transaction.transaction_id,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(transaction: ExplainRequest, request: Request):
    model = _require_model(request)
    shap_explainer = getattr(request.app.state, "shap_explainer", None)
    if shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer is not loaded.")

    try:
        row = _request_to_frame(transaction)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    risk_score = _predict_probability(model, row)
    explanation_result = explain_single_prediction(
        model,
        shap_explainer,
        row.iloc[0],
        FEATURE_COLUMNS,
    )
    breakdown = explanation_result["breakdown"]
    genai_explanation = generate_explanation(breakdown, risk_score, row.iloc[0])
    shap_contributions = [
        {
            "feature": record["feature"],
            "value": float(record["value"]),
            "contribution": float(record["shap_contribution"]),
        }
        for record in breakdown[["feature", "value", "shap_contribution"]].to_dict("records")
    ]

    return ExplainResponse(
        risk_score=risk_score,
        label=_label_from_probability(risk_score),
        shap_contributions=shap_contributions,
        explanation=genai_explanation,
    )

