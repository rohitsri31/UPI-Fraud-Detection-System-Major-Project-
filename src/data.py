"""Load the transaction dataset, creating the synthetic one on first use."""
from pathlib import Path

import pandas as pd

from . import config, generate_data


def load_transactions(path=None, create_if_missing=True):
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        if not create_if_missing:
            raise FileNotFoundError(path)
        print(f"No dataset at {path}; generating a synthetic one.")
        generate_data.generate_and_save(path)
    df = pd.read_csv(path)
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Expected at least: {config.RAW_COLUMNS}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
