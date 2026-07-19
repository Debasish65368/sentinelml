# Experiment 02 — Baseline Logistic Regression: SMOTE vs Class Weights

## What I did
Trained Logistic Regression two ways on the same train split: class_weight='balanced' 
vs SMOTE oversampling (applied to train only, never val/test).

## Results (validation set)
| Method | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| class_weight | 0.9741 | 0.7214 | 0.057 | 0.901 | 0.107 |
| SMOTE | 0.9771 | 0.7096 | 0.075 | 0.887 | 0.138 |

## Decision
Chose class_weight over SMOTE. Reasoning: PR-AUC is the more meaningful metric for 
this level of imbalance (threshold-independent, unlike F1 which depends on the 
arbitrary 0.5 cutoff), and class_weight scored higher on it (0.7214 vs 0.7096). 
SMOTE's slightly better F1/ROC-AUC likely comes from synthetic minority samples 
that don't necessarily reflect real fraud patterns. class_weight is also simpler 
and avoids any risk of synthetic-data artifacts.

## Success criteria (locking in real numbers now)
- Baseline PR-AUC: 0.7214 (class_weight Logistic Regression)
- Target for tree models (Phase 4): meaningfully exceed 0.7214 PR-AUC
- Precision/recall at 0.5 threshold is not the deciding factor — will tune 
  threshold once XGBoost is trained, since 0.5 is not the right operating point 
  for this level of imbalance

## Next
Phase 4: Random Forest + XGBoost using class_weight, compare against this baseline.