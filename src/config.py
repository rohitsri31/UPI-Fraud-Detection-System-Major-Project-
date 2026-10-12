"""Project-wide settings. Edit values here, or override the folders with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("UPI_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("UPI_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.environ.get("UPI_REPORT_DIR", ROOT / "reports"))

DATA_PATH = DATA_DIR / "transactions.csv"   # raw transactions (CSV)
MODEL_PATH = MODEL_DIR / "model.joblib"     # saved model + threshold
DB_PATH = DATA_DIR / "app.db"               # SQLite file used by the dashboard

SEED = 42

# --- Synthetic data generator (src/generate_data.py) -----------------------------------
N_USERS = 2000
N_TRANSACTIONS = 60_000
FRAUD_RATE = 0.015      # share of fraudulent transactions
N_DAYS = 90

TXN_TYPES = ["p2p", "merchant", "collect_request", "qr_scan"]

# Columns a dataset needs (see README, "Using your own dataset").
RAW_COLUMNS = ["timestamp", "sender_id", "receiver_id", "amount",
               "txn_type", "device_id", "city", "is_fraud"]

# --- Model inputs -----------------------------------------------------------------------
# Order matters: training and the live API both build exactly these columns.
FEATURE_COLUMNS = [
    "amount_log", "amount_ratio", "hour", "is_night",
    "is_new_payee", "is_new_device", "is_new_city", "txns_last_hour",
] + [f"type_{t}" for t in TXN_TYPES]

# --- Training and model selection (src/train.py) ----------------------------------------
TRAIN_FRACTION = 0.6    # oldest 60% of transactions
VAL_FRACTION = 0.2      # next 20%; the newest 20% is the untouched test set
F_BETA = 2.0            # F2 weighs recall higher than precision (missed fraud costs more)
MODELS = ["logreg", "random_forest", "gradient_boosting", "xgboost"]
STRATEGIES = ["none", "class_weight", "smote", "undersample"]
