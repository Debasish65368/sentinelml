# Experiment 04 — LightGBM Evaluation (Excluded)

## What I did
Trained LightGBM once as a sanity comparison against tuned XGBoost, using similar 
hyperparameters (max_depth=6, learning_rate=0.04, n_estimators=600).

## Bug found and fixed
Initial run gave PR-AUC of 0.096 — drastically inconsistent with its own F1 (0.21). 
Investigation showed predicted probabilities were near-binary/saturated (only 20 
unique values), which breaks PR-AUC's ranking-based calculation. Fixed by using 
is_unbalance=True and adding min_child_weight=8 (matching XGBoost's tuned value), 
which restored a proper probability distribution (42,409 unique values).

## Corrected results (validation)
PR-AUC: 0.8599, ROC-AUC: 0.9566, Precision: 0.824, Recall: 0.859, 
Inference: 0.219s

## Decision
Excluded from final pipeline. Corrected PR-AUC (0.8599) does not beat tuned 
XGBoost (0.8779), and inference time (0.219s) is more than double XGBoost's 
(0.094s), failing the <100ms target. No dimension where it wins. Kept in the repo 
as a documented comparison, not used in the deployed pipeline.

## Note for DESIGN_DECISIONS.md
LightGBM evaluated and excluded — see this file for the saturated-probability bug 
found and fixed during evaluation, and the final PR-AUC comparison.