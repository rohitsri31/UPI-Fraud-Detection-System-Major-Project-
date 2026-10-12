"""Feature engineering shared by training and the live API.

Training builds history features from a whole dataset, using past transactions only
so nothing leaks from the future. When scoring one live transaction the caller supplies
the same signals. Both paths end in `finalize_features`, so the model always receives
exactly `config.FEATURE_COLUMNS`.
"""
import numpy as np
import pandas as pd

from . import config

PRIOR_WEIGHT = 3  # until a user has history, their "usual amount" leans on the dataset median


def add_history_features(df):
    """Sort chronologically and add per-sender history columns (past transactions only)."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    by_sender = df.groupby("sender_id", sort=False)

    prior_n = by_sender.cumcount()
    prior_sum = by_sender["amount"].cumsum() - df["amount"]
    median_amount = df["amount"].median()
    df["typical_amount"] = (prior_sum + PRIOR_WEIGHT * median_amount) / (prior_n + PRIOR_WEIGHT)

    has_history = prior_n > 0   # a sender's very first payment has nothing to compare with
    for column, flag in (("receiver_id", "is_new_payee"), ("device_id", "is_new_device"), ("city", "is_new_city")):
        first_seen = df.groupby(["sender_id", column], sort=False).cumcount() == 0
        df[flag] = (first_seen & has_history).astype(int)

    df["hour"] = df["timestamp"].dt.hour

    seconds = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    counts = np.zeros(len(df), dtype=int)
    for idx in by_sender.indices.values():       # rows of one sender, already in time order
        t = seconds[idx]
        counts[idx] = np.arange(len(idx)) - np.searchsorted(t, t - 3600, side="left")
    df["txns_last_hour"] = counts
    return df


def finalize_features(frame):
    """Turn history columns into the model's input. Used by training and by the API."""
    out = pd.DataFrame(index=frame.index)
    amount = frame["amount"].astype(float)
    out["amount_log"] = np.log1p(amount)
    out["amount_ratio"] = (amount / frame["typical_amount"].astype(float).clip(lower=1.0)).clip(upper=100.0)
    out["hour"] = frame["hour"].astype(int)
    out["is_night"] = out["hour"].between(0, 5).astype(int)
    for flag in ("is_new_payee", "is_new_device", "is_new_city"):
        out[flag] = frame[flag].astype(int)
    out["txns_last_hour"] = frame["txns_last_hour"].astype(int).clip(upper=20)
    for txn_type in config.TXN_TYPES:
        out[f"type_{txn_type}"] = (frame["txn_type"] == txn_type).astype(int)
    return out[config.FEATURE_COLUMNS].astype(float)


def build_features(df):
    """Whole dataset -> (X, y) in chronological order."""
    history = add_history_features(df)
    return finalize_features(history), history["is_fraud"].astype(int)


# --- One live transaction (used by the API) --------------------------------------------

def _number(payload, key, default, low, high, as_int=False):
    raw = payload.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return int(value) if as_int else value


def _flag(payload, key):
    raw = payload.get(key, 0)
    if isinstance(raw, str):
        return int(raw.strip().lower() in ("1", "true", "yes", "on"))
    return int(bool(raw))


def features_from_request(payload):
    """Validate a JSON payload and return a one-row feature frame.

    Fields: amount (required), typical_amount (the user's usual payment; defaults to amount),
    hour (0-23, default 12), txn_type, is_new_payee, is_new_device, is_new_city, txns_last_hour.
    """
    amount = _number(payload, "amount", None, 1, 1_000_000)
    if amount is None:
        raise ValueError("amount is required")
    txn_type = str(payload.get("txn_type") or "p2p")
    if txn_type not in config.TXN_TYPES:
        raise ValueError(f"txn_type must be one of {config.TXN_TYPES}")
    row = {
        "amount": amount,
        "typical_amount": _number(payload, "typical_amount", amount, 1, 1_000_000),
        "hour": _number(payload, "hour", 12, 0, 23, as_int=True),
        "txn_type": txn_type,
        "is_new_payee": _flag(payload, "is_new_payee"),
        "is_new_device": _flag(payload, "is_new_device"),
        "is_new_city": _flag(payload, "is_new_city"),
        "txns_last_hour": _number(payload, "txns_last_hour", 0, 0, 1000, as_int=True),
    }
    return finalize_features(pd.DataFrame([row]))
