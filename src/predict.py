"""Load the saved model and score one transaction."""
from pathlib import Path

import joblib

from . import config, features


class ModelNotLoaded(RuntimeError):
    pass


def risk_indicators(row):
    """Plain-language reasons that apply to this transaction.

    These are simple rules over the inputs, not the model's own attribution. SHAP is the
    natural upgrade if you want per-prediction explanations from the model itself.
    """
    reasons = []
    if row["is_new_payee"]:
        reasons.append("Payment to a new payee")
    if row["is_new_device"]:
        reasons.append("Unrecognised device")
    if row["is_new_city"]:
        reasons.append("Unusual location")
    if row["amount_ratio"] >= 4:
        reasons.append(f"Amount is {row['amount_ratio']:.0f}x the user's usual payment")
    if row["is_night"]:
        reasons.append("Late-night transaction (00:00 to 05:59)")
    if row["txns_last_hour"] >= 3:
        reasons.append(f"{int(row['txns_last_hour'])} payments in the last hour")
    if row["type_collect_request"] and row["is_new_payee"]:
        reasons.append("Collect request from an unknown payee")
    if row["type_qr_scan"] and row["is_new_payee"]:
        reasons.append("QR payment to a new merchant")
    return reasons


class Scorer:
    def __init__(self, model_path=None):
        self.path = Path(model_path or config.MODEL_PATH)
        self.bundle = None
        self.load()

    def load(self):
        # Only load model files you trained yourself: joblib files can run code when opened.
        self.bundle = joblib.load(self.path) if self.path.exists() else None
        return self.ready

    @property
    def ready(self):
        return self.bundle is not None

    def info(self):
        if not self.ready:
            return {"model_loaded": False}
        b = self.bundle
        return {"model_loaded": True, "model_name": b["model_name"], "strategy": b["strategy"],
                "threshold": round(b["threshold"], 4), "trained_at": b["trained_at"]}

    def score(self, payload):
        if not self.ready:
            raise ModelNotLoaded("No trained model found. Run: python -m src.train")
        X = features.features_from_request(payload)[self.bundle["feature_columns"]]
        probability = float(self.bundle["model"].predict_proba(X)[0, 1])
        threshold = self.bundle["threshold"]
        flagged = probability >= threshold
        if flagged:
            level = "high"
        elif probability >= threshold / 2:
            level = "medium"
        else:
            level = "low"
        row = X.iloc[0]
        return {
            "fraud_probability": round(probability, 4),
            "flagged": bool(flagged),
            "risk_level": level,
            "threshold": round(threshold, 4),
            "indicators": risk_indicators(row),
            "features": {name: round(float(value), 3) for name, value in row.items()},
        }
