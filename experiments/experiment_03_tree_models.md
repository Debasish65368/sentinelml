# Experiment 03 — Random Forest, XGBoost, Hyperparameter Tuning

## What I did
Trained Random Forest and XGBoost (default + Optuna-tuned, 30 trials optimizing 
PR-AUC) using class_weight/scale_pos_weight for imbalance handling, on the same 
splits as Experiment 02.

## Results (validation set)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Inference Time |
|---|---|---|---|---|---|---|
| XGBoost tuned | 0.9828 | 0.8779 | 0.922 | 0.831 | 0.874 | 0.094s |
| Random Forest | 0.9605 | 0.8588 | 0.908 | 0.831 | 0.868 | 0.116s |
| XGBoost default | 0.9800 | 0.8499 | 0.642 | 0.859 | 0.735 | 0.058s |
| Logistic Regression (baseline) | 0.9741 | 0.7214 | 0.057 | 0.901 | 0.107 | 0.004s |

## Decision
XGBoost (tuned) selected as primary model. PR-AUC improved from 0.7214 (baseline) 
to 0.8779 — a meaningful gain, not marginal. Also beats Random Forest on both 
PR-AUC and inference time (0.094s vs 0.116s, under my <100ms target). Untuned 
XGBoost underperforming Random Forest on PR-AUC shows tuning specifically drove 
the improvement, not just the algorithm choice.

Best hyperparameters: max_depth=6, learning_rate=0.041, n_estimators=600, 
subsample=0.865, colsample_bytree=0.996, min_child_weight=8.

## Next
Decide on LightGBM (train once, keep only if it meaningfully beats tuned XGBoost). 
Then Phase 5: PyTorch autoencoder for unsupervised comparison.