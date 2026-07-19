# Experiment 09 — Streamlit Demo UI

## What I did
Built a single-page Streamlit app: upload CSV -> call /predict for each row -> 
display sorted risk table (color-coded fraud/normal) -> select a transaction -> 
call /explain -> show risk score, Groq explanation, and SHAP bar chart.

## Design choices
- Single page, no login/case management/multi-page nav — the model is the 
  product, not the UI, per the original architecture decision.
- API base URL configurable via environment variable, so the same code works 
  locally and once deployed (Render URL swapped in later).
- Explicit error handling for API-down and malformed-CSV cases, since a demo 
  that crashes silently is worse than no demo.

## Verification
- Manual end-to-end test against real X_val.csv rows: upload, score, explain all 
  worked correctly.
- Automated check via Streamlit's AppTest simulating the full user flow.

## Next
Phase 10: Deploy FastAPI to Render, Streamlit to Streamlit Community Cloud.