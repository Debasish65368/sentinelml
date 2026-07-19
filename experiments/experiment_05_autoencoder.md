# Experiment 05 — PyTorch Autoencoder (Unsupervised Anomaly Detection)

## What I did
Trained a feedforward autoencoder (32->16->8->16->32) on normal (Class=0) 
transactions only, 50 epochs, final training loss 0.484. Tested three threshold 
strategies on reconstruction error: 95th percentile, 99th percentile, and 
F1-optimal search.

## Threshold comparison
| Strategy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| normal_p95 | 0.029 | 0.887 | 0.056 | 0.240 |
| normal_p99 | 0.113 | 0.761 | 0.196 | 0.240 |
| f1_optimal | 0.488 | 0.296 | 0.368 | 0.240 |

Chose f1_optimal (threshold=44.54) as the best balance, though PR-AUC (0.240) is 
far below tuned XGBoost (0.878).

## Overlap with tuned XGBoost
[fill in corrected numbers once re-run against tuned model]

## Decision / Interpretation
The autoencoder adds no incremental fraud detection value over supervised XGBoost 
on this dataset. This is expected: autoencoders are most useful when labeled 
fraud data is scarce or absent, since they learn only "normal" patterns without 
ever seeing confirmed fraud. Here, XGBoost has direct access to 473 labeled fraud 
cases during training, giving it a much stronger, more targeted signal than 
reconstruction-error-based anomaly detection can provide. 

This isn't a failed experiment — it's a legitimate finding demonstrating that 
supervised learning is the right choice when labels exist, and that unsupervised 
anomaly detection's value is conditional on label scarcity, not universal.

## Next
Phase 6: SHAP explainability on tuned XGBoost.