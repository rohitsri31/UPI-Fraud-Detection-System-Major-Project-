"""Ways to deal with rare fraud: none, class weights (in train.py), SMOTE, undersampling.

SMOTE is written out here (about 25 lines) so the project needs no extra package.
To use the imbalanced-learn version instead, replace `smote()` with
`imblearn.over_sampling.SMOTE(sampling_strategy=0.2).fit_resample(X, y)`.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config

SMOTE_RATIO = 0.2        # after oversampling: 1 fraud row for every 5 genuine rows
UNDERSAMPLE_RATIO = 0.1  # after undersampling: 1 fraud row for every 10 genuine rows
WHOLE_NUMBER_COLUMNS = ("hour", "txns_last_hour")


def smote(X, y, ratio=SMOTE_RATIO, k=5, seed=config.SEED):
    """Add synthetic fraud rows between a fraud row and one of its nearest fraud neighbours."""
    rng = np.random.default_rng(seed)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    fraud = X[y == 1].to_numpy(dtype=float)
    n_new = int(ratio * int((y == 0).sum())) - len(fraud)
    if n_new <= 0 or len(fraud) < 2:
        return X, y

    k = min(k, len(fraud) - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(fraud).kneighbors(fraud, return_distance=False)[:, 1:]
    base = rng.integers(0, len(fraud), n_new)
    other = neighbours[base, rng.integers(0, k, n_new)]
    synthetic = fraud[base] + rng.random((n_new, 1)) * (fraud[other] - fraud[base])

    # keep 0/1 flags and whole-number columns valid after interpolating
    columns = list(X.columns)
    binary = [i for i, c in enumerate(columns) if set(X[c].unique()) <= {0, 1}]
    whole = [columns.index(c) for c in WHOLE_NUMBER_COLUMNS if c in columns]
    for i in binary + whole:
        synthetic[:, i] = np.round(synthetic[:, i])

    X_new = pd.concat([X, pd.DataFrame(synthetic, columns=columns)], ignore_index=True)
    y_new = pd.concat([y, pd.Series(np.ones(n_new, dtype=int))], ignore_index=True)
    return X_new, y_new


def undersample(X, y, ratio=UNDERSAMPLE_RATIO, seed=config.SEED):
    """Keep every fraud row and a random subset of genuine rows."""
    rng = np.random.default_rng(seed)
    labels = y.to_numpy()
    fraud_idx = np.flatnonzero(labels == 1)
    genuine_idx = np.flatnonzero(labels == 0)
    n_keep = min(len(genuine_idx), int(len(fraud_idx) / ratio))
    keep = np.sort(np.concatenate([fraud_idx, rng.choice(genuine_idx, n_keep, replace=False)]))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def resample(strategy, X, y):
    """'none' and 'class_weight' leave the data alone (class_weight is applied inside the model)."""
    if strategy == "smote":
        return smote(X, y)
    if strategy == "undersample":
        return undersample(X, y)
    if strategy in ("none", "class_weight"):
        return X, y
    raise ValueError(f"unknown imbalance strategy {strategy!r}")
