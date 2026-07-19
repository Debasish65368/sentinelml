import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from groq import Groq


EDA_TOP_CORRELATED_FEATURES = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11", "V4", "V18"]
GROQ_MODEL = "llama-3.3-70b-versatile"


def compute_shap_values(model, X):
    """Compute SHAP values for a tree model on the provided feature matrix."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    return np.asarray(shap_values)


def get_global_feature_importance(shap_values, feature_names):
    """Rank features by mean absolute SHAP value."""
    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    print("=== Top 15 SHAP Global Feature Importance ===")
    print(importance.head(15).to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    return importance


def explain_single_prediction(model, shap_explainer, X_row, feature_names):
    """Return baseline, feature contributions, and final prediction for one row."""
    if isinstance(X_row, pd.Series):
        row_df = X_row.to_frame().T
        row_values = X_row.to_numpy()
    elif isinstance(X_row, pd.DataFrame):
        row_df = X_row.iloc[[0]]
        row_values = row_df.iloc[0].to_numpy()
    else:
        row_values = np.asarray(X_row).reshape(-1)
        row_df = pd.DataFrame([row_values], columns=feature_names)

    shap_values = shap_explainer.shap_values(row_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values).reshape(-1)

    expected_value = shap_explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]

    prediction = float(model.predict_proba(row_df)[:, 1][0])
    log_odds_prediction = float(expected_value + shap_values.sum())

    breakdown = pd.DataFrame(
        {
            "feature": list(feature_names),
            "value": row_values,
            "shap_contribution": shap_values,
            "abs_contribution": np.abs(shap_values),
        }
    ).sort_values("abs_contribution", ascending=False, ignore_index=True)

    return {
        "baseline_value": float(expected_value),
        "final_prediction_probability": prediction,
        "final_prediction_log_odds": log_odds_prediction,
        "breakdown": breakdown,
    }


def compare_shap_to_eda_correlations(shap_importance, eda_features=EDA_TOP_CORRELATED_FEATURES):
    """Compare SHAP top features to the EDA correlation ranking."""
    shap_top_15 = shap_importance.head(15)["feature"].tolist()
    overlap = [feature for feature in eda_features if feature in shap_top_15]
    shap_only = [feature for feature in shap_top_15 if feature not in eda_features]
    agreement = len(overlap) / len(eda_features)

    print("=== SHAP vs EDA Correlation Cross-Check ===")
    print(f"EDA top correlated features: {eda_features}")
    print(f"SHAP top 15 features: {shap_top_15}")
    print(f"Overlap count: {len(overlap)} / {len(eda_features)}")
    print(f"Overlapping features: {overlap}")
    print(f"SHAP-only top features: {shap_only}")
    if agreement >= 0.6:
        print("Verdict: SHAP mostly agrees with the EDA correlation ranking, with some tree-specific differences.")
    else:
        print("Verdict: SHAP reveals a different importance pattern, likely from nonlinear effects/interactions.")

    return {
        "eda_features": eda_features,
        "shap_top_15": shap_top_15,
        "overlap": overlap,
        "shap_only": shap_only,
        "agreement_fraction": agreement,
    }


def _format_top_contributions(shap_breakdown, top_n=5):
    top_features = shap_breakdown.head(top_n).copy()
    lines = []
    for _, row in top_features.iterrows():
        direction = "increases fraud risk" if row["shap_contribution"] > 0 else "decreases fraud risk"
        lines.append(
            f"- {row['feature']}: value={row['value']:.6f}, "
            f"contribution={row['shap_contribution']:.6f} ({direction})"
        )
    return "\n".join(lines)


def _fallback_explanation(shap_breakdown, prediction_prob):
    top_features = shap_breakdown.head(3)
    risk_features = top_features[top_features["shap_contribution"] > 0]["feature"].tolist()
    offset_features = top_features[top_features["shap_contribution"] < 0]["feature"].tolist()
    risk_text = ", ".join(risk_features) if risk_features else "the strongest inputs"
    offset_text = ", ".join(offset_features) if offset_features else "some offsetting inputs"
    return (
        f"The model estimated a {prediction_prob:.2%} fraud probability. "
        f"The main drivers were {risk_text}, while {offset_text} reduced the score; "
        "this fallback explanation was generated because the GenAI service was unavailable."
    )


def generate_explanation(shap_breakdown, prediction_prob, transaction_data):
    """Generate a plain-English analyst rationale from SHAP-style contributions."""
    load_dotenv()
    top_contributions = _format_top_contributions(shap_breakdown, top_n=5)
    transaction_snapshot = transaction_data.to_dict() if hasattr(transaction_data, "to_dict") else dict(transaction_data)

    prompt = f"""
You are helping a fraud analyst understand one model prediction.

Prediction probability: {prediction_prob:.6f}

Top model drivers:
{top_contributions}

Transaction values for context:
{transaction_snapshot}

Write a 2-3 sentence plain-English explanation an analyst could read.
Avoid technical jargon such as "SHAP value", "log odds", or "feature attribution".
Mention the most important feature names directly, and explain whether they raised or lowered concern.
Do not contradict yourself: features marked "increases fraud risk" should be described as raising concern,
and features marked "decreases fraud risk" should be described as lowering concern.
"""

    try:
        client = Groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You translate fraud model evidence into concise analyst-facing explanations.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
            timeout=20,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"WARNING: Groq explanation generation failed: {exc}")
        explanation = _fallback_explanation(shap_breakdown, prediction_prob)

    return explanation


def validate_explanation_accuracy(explanation_text, shap_breakdown):
    """Check that the explanation references at least 2 of the top 3 drivers."""
    top_3_features = shap_breakdown.head(3)["feature"].tolist()
    normalized_text = explanation_text.lower()
    mentioned = [feature for feature in top_3_features if feature.lower() in normalized_text]

    if len(mentioned) < 2:
        print(
            "WARNING: Explanation may be disconnected from SHAP input. "
            f"Expected at least 2 of {top_3_features}, found {mentioned}."
        )
        return False

    print(f"Explanation sanity check passed. Mentioned top drivers: {mentioned}")
    return True
