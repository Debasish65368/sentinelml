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
