from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import os
import secrets
import json

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from xgboost import XGBClassifier

from api.schemas import (
    ExplainRequest,
    ExplainResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictResponse,
    TransactionInput,
)
from src.explain import explain_single_prediction, generate_explanation, validate_explanation_accuracy, _fallback_explanation
from src.train_models import _build_xgboost, _compute_scale_pos_weight, load_processed_splits, select_optimal_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TUNED_MODEL_PATH = MODEL_DIR / "tuned_xgboost_final.json"
TUNED_THRESHOLD_PATH = MODEL_DIR / "tuned_xgboost_threshold.json"
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

def _requests_to_frame(transactions: list[TransactionInput]) -> pd.DataFrame:
    rows = []
    for transaction in transactions:
        payload = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
        rows.append({feature: payload[feature] for feature in FEATURE_COLUMNS})
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    if frame.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {frame.shape[1]}.")
    return frame

def _predict_probability(model, row):
    return float(model.predict_proba(row)[:, 1][0])

def _label_from_probability(probability, threshold):
    return "fraud" if probability >= threshold else "normal"

def load_tuned_xgboost_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if TUNED_MODEL_PATH.exists() and TUNED_THRESHOLD_PATH.exists():
        model = XGBClassifier()
        model.load_model(TUNED_MODEL_PATH)
        print(f"Loaded tuned XGBoost model from {TUNED_MODEL_PATH}")
        with open(TUNED_THRESHOLD_PATH, "r") as f:
            threshold_data = json.load(f)
        return model, threshold_data["threshold"]

    print("No cached tuned XGBoost model found; training final model on train+val data.")
    X_train, X_val, _, y_train, y_val, _ = load_processed_splits()

    scale_pos_weight_train = _compute_scale_pos_weight(y_train)
    threshold_model = _build_xgboost(scale_pos_weight_train, **TUNED_XGBOOST_PARAMS)
    threshold_model.fit(X_train[FEATURE_COLUMNS], y_train)
    y_val_proba = threshold_model.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]
    best_threshold = select_optimal_threshold(y_val, y_val_proba)

    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    scale_pos_weight = _compute_scale_pos_weight(y_full)
    model = _build_xgboost(scale_pos_weight, **TUNED_XGBOOST_PARAMS)
    model.fit(X_full[FEATURE_COLUMNS], y_full)
    model.save_model(TUNED_MODEL_PATH)

    with open(TUNED_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_threshold}, f)

    print(f"Saved final tuned XGBoost model to {TUNED_MODEL_PATH}")
    return model, best_threshold

def load_model_resources():
    model, threshold = load_tuned_xgboost_model()
    shap_explainer = shap.TreeExplainer(model)
    scaler = joblib.load(AUTOENCODER_SCALER_PATH) if AUTOENCODER_SCALER_PATH.exists() else None
    return {
        "model": model,
        "threshold": threshold,
        "shap_explainer": shap_explainer,
        "scaler": scaler,
        "model_version": MODEL_VERSION,
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = load_model_resources()
    app.state.model = resources["model"]
    app.state.threshold = resources["threshold"]
    app.state.shap_explainer = resources["shap_explainer"]
    app.state.scaler = resources["scaler"]
    app.state.model_version = resources["model_version"]
    app.state.ready = True
    yield

app = FastAPI(title="SentinelML Fraud API", version="0.1.0", lifespan=lifespan)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    origins = ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.getenv("SENTINEL_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="API key is not configured on the server.")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key")
    if not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

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
        label=_label_from_probability(risk_score, getattr(request.app.state, "threshold", 0.5)),
        transaction_id=transaction.transaction_id,
    )

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(batch: PredictBatchRequest, request: Request):
    model = _require_model(request)
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    try:
        frame = _requests_to_frame(batch.transactions)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = perf_counter()
    risk_scores = model.predict_proba(frame)[:, 1]
    elapsed_ms = (perf_counter() - start_time) * 1000
    print(f"/predict_batch rows={len(batch.transactions)} latency_ms={elapsed_ms:.3f}")

    threshold = getattr(request.app.state, "threshold", 0.5)
    predictions = [
        PredictResponse(
            risk_score=float(score),
            label=_label_from_probability(float(score), threshold),
            transaction_id=transaction.transaction_id,
        )
        for score, transaction in zip(risk_scores, batch.transactions)
    ]
    return PredictBatchResponse(predictions=predictions)

@app.post("/explain", response_model=ExplainResponse)
def explain(transaction: ExplainRequest, request: Request, api_key: str = Depends(verify_api_key)):
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

    try:
        is_valid = validate_explanation_accuracy(genai_explanation, breakdown)
        if not is_valid:
            genai_explanation = _fallback_explanation(breakdown, risk_score)
    except Exception as exc:
        print(f"WARNING: Explanation validation failed with exception: {exc}")
        genai_explanation = _fallback_explanation(breakdown, risk_score)

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
        label=_label_from_probability(risk_score, getattr(request.app.state, "threshold", 0.5)),
        shap_contributions=shap_contributions,
        explanation=genai_explanation,
    )
