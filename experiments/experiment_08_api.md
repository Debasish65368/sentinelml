# Experiment 08 — FastAPI Serving Layer

## What I did
Built two endpoints wiring together the full pipeline:
- POST /predict: tuned XGBoost only, fast path (no SHAP/Groq)
- POST /explain: XGBoost + SHAP + Groq, on-demand, slower path
- GET /health: status check
Model loaded once at startup from models/tuned_xgboost.json, not retrained per 
request. SHAP explainer also initialized once at startup.

## Why split endpoints
/predict needs to be fast for potential batch/bulk use (e.g. scoring an uploaded 
CSV of many transactions). /explain involves SHAP computation + an external Groq 
API call, both slower — only triggered on-demand when a user inspects one 
specific transaction, not run for every row by default.

## Results
- /predict latency: 6.7ms (target was <100ms)
- 21/21 tests passing (schema validation, error handling, endpoint responses)
- Verified end-to-end on a real validation transaction through both endpoints

## Sample output
[paste your normal-transaction /predict and /explain output here]

## Next
Phase 9: Streamlit demo UI.