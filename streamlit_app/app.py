import os

import altair as alt
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("SENTINELML_API_BASE_URL", "http://localhost:8000").rstrip("/")
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "Amount",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
]
REQUEST_TIMEOUT_SECONDS = 30


def validate_columns(df):
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    extra = [column for column in df.columns if column not in FEATURE_COLUMNS]
    return missing, extra


def build_payload(row, transaction_id):
    payload = {column: float(row[column]) for column in FEATURE_COLUMNS}
    payload["transaction_id"] = str(transaction_id)
    return payload


def post_json(endpoint, payload):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to the SentinelML API at {API_BASE_URL}. "
            "Start FastAPI first with: uvicorn api.main:app --reload"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"The SentinelML API request to {endpoint} timed out.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = response.text if "response" in locals() else str(exc)
        raise RuntimeError(f"The SentinelML API returned an error for {endpoint}: {detail}") from exc
    return response.json()


def predict_transactions(df):
    rows = []
    progress = st.progress(0, text="Scoring transactions...")
    total_rows = len(df)

    for position, (index, row) in enumerate(df.iterrows(), start=1):
        prediction = post_json("/predict", build_payload(row, index))
        rows.append(
            {
                "transaction_index": index,
                "risk_score": prediction["risk_score"],
                "label": prediction["label"],
            }
        )
        progress.progress(position / total_rows, text=f"Scored {position} of {total_rows} transactions...")

    progress.empty()
    results = pd.DataFrame(rows).sort_values("risk_score", ascending=False, ignore_index=True)
    return results


def style_predictions(df):
    normal_style = "background-color: #eef7ee; color: #1a1a1a"
    fraud_style = "background-color: #ffe3e3; color: #1a1a1a"
    return df.style.apply(
        lambda row: [
            fraud_style if row["label"] == "fraud" else normal_style
            for _ in row
        ],
        axis=1,
    ).format({"risk_score": "{:.6f}"})


def top_shap_chart(contributions):
    shap_df = pd.DataFrame(contributions).head(5)
    shap_df["direction"] = shap_df["contribution"].apply(lambda value: "raises risk" if value > 0 else "lowers risk")
    chart = (
        alt.Chart(shap_df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Contribution"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"],
                    range=["#d62728", "#2ca02c"],
                ),
                title="Direction",
            ),
            tooltip=["feature", "value", "contribution", "direction"],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        shap_df[["feature", "value", "contribution", "direction"]],
        width="stretch",
        hide_index=True,
    )


st.set_page_config(page_title="SentinelML Fraud Demo", layout="wide")
st.title("SentinelML Fraud Detection Demo")

st.caption(f"API: {API_BASE_URL}")
uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV with V1-V28, Amount, hour_of_day, hour_sin, and hour_cos columns.")
    st.stop()

try:
    transactions = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

missing_columns, extra_columns = validate_columns(transactions)
if missing_columns:
    st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
    st.stop()

if extra_columns:
    st.warning(f"Ignoring extra columns not sent to the API: {', '.join(extra_columns)}")

if transactions.empty:
    st.error("CSV has no transactions to score.")
    st.stop()

transactions = transactions[FEATURE_COLUMNS].copy()

if st.button("Score Transactions", type="primary"):
    try:
        st.session_state.predictions = predict_transactions(transactions)
        st.session_state.transactions = transactions
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

if "predictions" not in st.session_state:
    st.stop()

predictions = st.session_state.predictions
st.subheader("Prediction Results")
st.dataframe(style_predictions(predictions), width="stretch", hide_index=True)

flagged = predictions[predictions["label"] == "fraud"]
selection_source = flagged if not flagged.empty else predictions
selection_label = "Flagged transaction" if not flagged.empty else "Transaction"

selected_index = st.selectbox(
    f"Select {selection_label.lower()} to explain",
    options=selection_source["transaction_index"].tolist(),
    format_func=lambda index: (
        f"index {index} | risk={selection_source.loc[selection_source['transaction_index'] == index, 'risk_score'].iloc[0]:.6f}"
    ),
)

if st.button("Explain Selected Transaction"):
    selected_row = st.session_state.transactions.loc[selected_index]
    try:
        explanation = post_json("/explain", build_payload(selected_row, selected_index))
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Explanation")
    metric_label = "Fraud" if explanation["label"] == "fraud" else "Normal"
    st.metric("Risk score", f"{explanation['risk_score']:.6f}", metric_label)
    st.write(explanation["explanation"])
    st.subheader("Top SHAP Contributions")
    top_shap_chart(explanation["shap_contributions"])
