# Experiment 06 — SHAP Explainability

## What I did
Applied SHAP TreeExplainer to the tuned XGBoost model on the validation set 
(42,559 rows x 32 features). Generated global feature importance, a beeswarm 
summary plot, and waterfall plots for one true positive, one false positive, and 
one false negative.

## Top 15 SHAP features
V14, V4, V12, V10, V11, V26, V3, V8, Amount, V7, V19, V21, V15, V18, V20 
(ranked by mean |SHAP value|)

## SHAP vs EDA correlation
8 of the top 10 EDA-correlated features (V14, V12, V10, V3, V7, V11, V4, V18) 
also appear in SHAP's top 15 — strong general agreement. Notably, V17 (EDA's #1 
correlated feature) does not appear in SHAP's top 15, likely because it's 
correlated with V14/V12 (PCA components aren't fully independent), so the tree 
model relies on the more informative among a correlated group rather than using 
both. SHAP also surfaces features EDA's simple correlation missed (V26, V8, 
Amount, V19, V21) — these likely contribute through nonlinear interactions that 
linear correlation can't detect.

## Case studies
- **True positive** (prob 0.9998): V14, V10, V4, V16, V11 strongly push toward 
  fraud — model is highly confident and correct.
- **False positive** (prob 0.9862): same fraud-associated features (V14, V10, 
  V12, V4, V7) present at levels that overwhelm the normal signal, despite the 
  transaction being legitimate — a case where the transaction genuinely resembles 
  fraud patterns.
- **False negative** (prob 0.0023): V14, V11, V10, V4 push away from fraud 
  despite some opposing pressure from V8, V20 — the fraud looked statistically 
  "normal" on the features that usually matter most, causing it to be missed.

## Decision
SHAP confirms the model isn't a black box — its top features have real predictive 
grounding, and the FP/FN cases show interpretable, if imperfect, reasoning. Ready 
to use SHAP output as input to the GenAI explanation layer next.

## Next
Study GenAI basics, then Phase 7: Groq-based natural language explanation layer.