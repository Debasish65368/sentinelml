# Experiment 07 — GenAI Explanation Layer (Groq)

## What I did
Built generate_explanation() which takes SHAP's top 5 feature contributions plus 
the model's predicted probability, and prompts Groq (llama-3.3-70b-versatile) to 
produce a 2-3 sentence plain-English rationale for a fraud analyst — grounded in 
the actual SHAP values, not generic LLM knowledge about fraud.

## Why Groq specifically
Used Groq for low-latency inference, since this sits behind an on-demand 
/explain endpoint rather than a background batch job — response time matters 
here. Also consistent with the GenAI tooling already used in BrewCo CRM.

## Why SHAP-grounded prompting
Passing actual SHAP contributions into the prompt (rather than asking the LLM to 
"explain why this might be fraud" generically) ties the explanation to the 
specific model's real reasoning for this specific prediction. This reduces 
hallucination risk — the LLM is summarizing given evidence, not generating 
fraud-detection knowledge from scratch.

## Validation
Added validate_explanation_accuracy() as an automated hallucination guard — 
checks that the generated text references at least 2 of the top 3 SHAP-driving 
features for that transaction. All 3 test cases (TP/FP/FN) passed.

Beyond the automated check, I manually cross-referenced each explanation against 
its SHAP waterfall breakdown from Experiment 06 to confirm both the feature 
names AND the direction of contribution (raises vs lowers concern) matched — not 
just that the right words appeared. All three were consistent.

## Sample output

**True Positive** (prob 0.999821):
"This transaction has a high predicted probability of being fraudulent, with a 
likelihood of 0.999821. The model is particularly concerned about the values of 
V14 and V10, which are unusually low and raise significant concern about the 
transaction's legitimacy, while the value of V2 is somewhat higher than expected 
and lowers the concern. Additionally, V4 raises some concern, but to a lesser 
extent, whereas V8 has a mildly reassuring effect, slightly lowering the overall 
fraud risk."

**False Positive** (prob 0.9862):
"The model predicts a high likelihood of fraud with a probability of 98.62%. 
This is largely due to concerns raised by features V14, V10, and V12, which have 
unusually low values and are contributing to the increased fraud risk. On the 
other hand, feature V8 has a relatively high value, which is lowering the 
concern for fraud, but its impact is outweighed by the other factors, 
particularly V14."

**False Negative** (prob 0.0023):
"The model has predicted a low fraud probability of 0.23% for this transaction. 
The presence of high values in features such as V14 and low values in features 
like V11 and V10 have lowered concern for fraud, suggesting that these factors 
contribute to the transaction's legitimacy. On the other hand, the values of V8 
and V20 have raised some concern, as they are associated with an increased risk 
of fraud, although their impact is outweighed by the more significant lowering 
effect of the other factors."

## Note on feature differences vs global SHAP ranking
Individual explanations reference each transaction's own top 5 SHAP contributors 
(e.g. V2, V16 appear here), which can differ from the global top-15 feature list 
in Experiment 06. This is expected: the global ranking reflects average 
importance across all transactions, while each explanation is local to that 
specific row's actual feature values.

## Decision
GenAI layer works as intended: fast, grounded in real SHAP output, passes both 
automated and manual consistency checks. Kept as a secondary feature behind 
/explain, not the headline of the project — core value still comes from the ML/
DL/XAI pipeline.

## Next
Phase 8: FastAPI serving layer (/predict, /explain endpoints).