# sentinelml

Fraud detection ML system scaffold for classic ML models, deep learning, explainability, and a GenAI explanation layer.

## Setup

Create and activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Notes

- Add your real Groq API key to a local `.env` file.
- Place the Kaggle dataset under `data/raw/` when ready.
- Model code, data pipelines, and dataset download logic are intentionally not included yet.

## Local API

Run the FastAPI service from the project root:

```powershell
.\venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Local Streamlit Demo

Start the API first, then run Streamlit from a second terminal:

```powershell
$env:SENTINELML_API_BASE_URL="http://localhost:8000"
.\venv\Scripts\streamlit.exe run streamlit_app\app.py
```

## Deployment

Render FastAPI start command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Render sets `$PORT` automatically. The app does not hardcode a serving port.

Streamlit Community Cloud app entrypoint:

```text
streamlit_app/app.py
```

Set `SENTINELML_API_BASE_URL` in Streamlit Cloud secrets or environment configuration to the deployed Render API URL. Set `GROQ_API_KEY` in Render for `/explain`.

Deployment artifact policy:

- Commit `models/tuned_xgboost.json`; the API needs this model in a fresh deployment.
- Keep `data/raw/`, `data/processed/`, `mlruns/`, and local autoencoder artifacts ignored.
- `models/autoencoder_scaler.joblib` is optional for the current API startup path and is not required by `/predict` or `/explain`.
