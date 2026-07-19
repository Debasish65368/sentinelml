# Experiment 00 — EDA Findings

## What I found
- Dataset: 284,807 transactions, 492 fraud cases (0.173% fraud rate, ~578:1 
  normal-to-fraud ratio). Severe class imbalance, more extreme than typical 
  tutorial datasets.
- Top 5 features correlated with fraud: V17, V14, V12, V10, V16. These are 
  PCA-anonymized components, so I can't interpret what they physically represent, 
  but they carry the strongest linear signal pre-modeling.
- Found 1,081 duplicate rows in the raw data. Of these, 1,062 are normal 
  transactions (Class=0) and 19 are fraud (Class=1).
- Amount alone shows a weak standalone separation between fraud/non-fraud 
  (standardized mean gap below threshold) — not a strong univariate predictor on 
  its own, though tree models may still find nonlinear interactions using it.
- Time shows a usable pattern by the same test — worth investigating further as 
  a cyclical (hour-of-day) feature rather than raw elapsed seconds.
- No missing values, no negative Amount values, all expected columns present.
- Data spans only ~2 days (Time ranges from 0 to 172,792 seconds).

## Decisions
1. **Drop duplicate rows before splitting.** Reason: a duplicated row landing in 
   both train and test would let the model "see" a test row during training, 
   inflating evaluation metrics artificially (data leakage). This matters more 
   than the small loss in row count.
2. **Use a stratified random split, not a time-ordered split.** With only ~2 days 
   of data, there isn't enough time range for a meaningful chronological holdout — 
   a time-based split here would mostly just be arbitrary, not a real simulation 
   of "future" transactions.
3. **Engineer Time into an hour-of-day cyclical feature** instead of using raw 
   seconds-since-start, since raw Time doesn't generalize (it's just an artifact 
   of when this particular dataset was collected) but time-of-day patterns might 
   be real signal.
4. **Compare SMOTE vs class weights explicitly in Phase 3**, rather than assuming 
   either works better upfront, given how extreme the imbalance is here.

## Going into Phase 2
Final feature set: V1–V28 + Amount + derived hour_of_day. Drop duplicates first, 
then stratified train/val/test split.