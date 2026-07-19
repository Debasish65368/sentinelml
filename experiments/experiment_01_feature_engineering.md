# Experiment 01 — Feature Engineering & Split

## What I did
- Removed 1,081 duplicate rows (284,807 -> 283,726), per the leakage-prevention 
  decision from Experiment 00.
- Dropped raw Time (not generalizable — just seconds since dataset start).
- Engineered hour_of_day from Time, then converted it to cyclical hour_sin/hour_cos 
  so the model treats hour 23 and hour 0 as close, not far apart.
- Final feature set: V1-V28, Amount, hour_of_day, hour_sin, hour_cos (32 features).
- Split stratified on Class (not time-based, per Experiment 00 reasoning) into 
  train/val/test at roughly 70/15/15.

## Result
- Fraud count after dedup: 473 (492 - 19 duplicate fraud rows removed).
- Split: Train 331 fraud / 198,608 rows, Val 71 fraud / 42,559 rows, 
  Test 71 fraud / 42,559 rows. Fraud ratio held consistent (~0.167%) across all 
  three splits — stratification worked correctly.
- 6/6 pytest checks passed (no NaNs, expected columns present, hour_sin/cos bounded 
  correctly, splits sum to original count, fraud ratio preserved, dedup actually 
  reduces row count).

## Next
Phase 3: compare SMOTE vs class weights on this split, train Logistic Regression 
baseline, lock success criteria numbers.