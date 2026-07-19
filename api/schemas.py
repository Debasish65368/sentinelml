from typing import Optional

from pydantic import BaseModel


class TransactionInput(BaseModel):
    transaction_id: Optional[str] = None
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    hour_of_day: float
    hour_sin: float
    hour_cos: float


class PredictResponse(BaseModel):
    risk_score: float
    label: str
    transaction_id: Optional[str] = None


class ExplainRequest(TransactionInput):
    pass


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class ExplainResponse(BaseModel):
    risk_score: float
    label: str
    shap_contributions: list[ShapContribution]
    explanation: str

