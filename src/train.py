"""Train, compare and save the fraud model.

Run:  python -m src.train

Every model is trained with every imbalance strategy on the oldest 60% of the data. The best
combination is chosen by F2 on the next 20% (validation), where its decision threshold is also
tuned. It is then scored once on the newest 20% (test), next to a simple rule-based baseline.

Outputs: models/model.joblib, models/comparison.csv, models/metrics.json, charts in reports/.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, features, imbalance

try:  # optional: pip install xgboost (macOS also needs: brew install libomp)
    from xgboost import XGBClassifier
except Exception:  # xgboost raises errors other than ImportError when its native library is missing
    XGBClassifier = None

MODEL_NAMES = ("logreg", "random_forest", "gradient_boosting", "xgboost")


def make_model(name, strategy, y_train):
    weighted = strategy == "class_weight"
    class_weight = "balanced" if weighted else None
    n_genuine, n_fraud = int((y_train == 0).sum()), int((y_train == 1).sum())
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=class_weight,
                                      n_jobs=-1, random_state=config.SEED)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=6,
                                              early_stopping=False, class_weight=class_weight,
                                              random_state=config.SEED)
    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed (pip install xgboost)")
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="aucpr", n_jobs=-1,
                             scale_pos_weight=n_genuine / max(n_fraud, 1) if weighted else 1.0,
                             random_state=config.SEED)
    raise ValueError(f"unknown model {name!r}")


def time_split(X, y):
    """Oldest rows train, next rows validate, newest rows test. Rows must be in time order."""
    n = len(X)
    i = int(n * config.TRAIN_FRACTION)
    j = int(n * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    return (X.iloc[:i], y.iloc[:i]), (X.iloc[i:j], y.iloc[i:j]), (X.iloc[j:], y.iloc[j:])


def best_threshold(y, proba):
    """Probability cut-off that maximises F-beta (beta=config.F_BETA) on the given data."""
    if int(y.sum()) == 0:
        raise ValueError("no fraud rows to tune the threshold on")
    precision, recall, thresholds = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    beta2 = config.F_BETA ** 2
    f_beta = (1 + beta2) * precision * recall / np.maximum(beta2 * precision + recall, 1e-12)
    k = int(np.argmax(f_beta))
    return float(thresholds[k]), float(f_beta[k])


def evaluate_labels(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=config.F_BETA, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate(y, proba, threshold):
    out = evaluate_labels(y, (proba >= threshold).astype(int))
    out.update(threshold=float(threshold), roc_auc=float(roc_auc_score(y, proba)),
               pr_auc=float(average_precision_score(y, proba)))
    return out


def rule_baseline(X):
    """The kind of fixed rules a simple system might use. The ML model has to beat this."""
    rules = (((X["is_new_payee"] == 1) & (X["amount_ratio"] >= 4))
             | ((X["is_new_device"] == 1) & (X["is_new_city"] == 1))
             | (X["txns_last_hour"] >= 4))
    return rules.astype(int)


def feature_importance(model, columns):
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:   # e.g. HistGradientBoosting has no built-in importances
        return {}
    total = values.sum() or 1.0
    return dict(sorted(zip(columns, (values / total).tolist()), key=lambda kv: -kv[1]))


def _save_plots(report_dir, label, comparison, y_test, p_test, test, baseline, importance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(fpr, tpr, color="#1d4ed8", label=f"{label} (AUC {test['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    base_fpr = baseline["fp"] / max(baseline["fp"] + baseline["tn"], 1)
    ax.scatter([base_fpr], [baseline["recall"]], color="#dc2626", marker="x", s=60, zorder=3,
               label="Rule-based baseline")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve, test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    cm = np.array([[test["tn"], test["fp"]], [test["fn"], test["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Genuine", "Fraud"])
    ax.set_yticks([0, 1], ["Genuine", "Fraud"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix, test set")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    ranked = comparison.sort_values("val_f2")
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.32 * len(ranked) + 1)))
    ax.barh([f"{m} + {s}" for m, s in zip(ranked["model"], ranked["strategy"])], ranked["val_f2"],
            color="#1d4ed8")
    ax.set(xlabel="F2 on validation data", title="Model and imbalance-strategy comparison")
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    if importance:
        names = list(importance)[:10][::-1]
        fig, ax = plt.subplots(figsize=(5.6, 4))
        ax.barh(names, [importance[n] for n in names], color="#1d4ed8")
        ax.set(xlabel="Share of total importance", title="Top features")
        fig.tight_layout()
        fig.savefig(report_dir / "feature_importance.png", dpi=150)
        plt.close(fig)


def train_and_save(df, models=None, strategies=None, model_path=None, report_dir=None, verbose=True):
    log = print if verbose else (lambda *args, **kwargs: None)
    models = list(models or config.MODELS)
    strategies = list(strategies or config.STRATEGIES)
    model_path = Path(model_path or config.MODEL_PATH)
    report_dir = Path(report_dir or config.REPORT_DIR)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = features.build_features(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y)
    log(f"Rows: train {len(X_train):,} | validation {len(X_val):,} | test {len(X_test):,} "
        f"| fraud share {y.mean():.2%}")

    results, fitted = [], {}
    for name in models:
        for strategy in strategies:
            started = time.time()
            try:
                model = make_model(name, strategy, y_train)
            except RuntimeError as err:
                log(f"  skipped {name} + {strategy}: {err}")
                continue
            X_fit, y_fit = imbalance.resample(strategy, X_train, y_train)
            model.fit(X_fit, y_fit)
            p_val = model.predict_proba(X_val)[:, 1]
            threshold, _ = best_threshold(y_val, p_val)
            val = evaluate(y_val, p_val, threshold)
            results.append({"model": name, "strategy": strategy, "threshold": threshold,
                            **{f"val_{k}": val[k] for k in ("precision", "recall", "f1", "f2", "roc_auc", "pr_auc")},
                            "fit_seconds": round(time.time() - started, 1)})
            fitted[(name, strategy)] = (model, threshold)
            log(f"  {name:<18} + {strategy:<13} F2 {val['f2']:.3f}  recall {val['recall']:.3f}  "
                f"precision {val['precision']:.3f}")
    if not results:
        raise RuntimeError("No model could be trained. Check the models and strategies you asked for.")

    comparison = (pd.DataFrame(results).sort_values(["val_f2", "val_pr_auc"], ascending=False)
                  .reset_index(drop=True))
    best = comparison.iloc[0]
    name, strategy = str(best["model"]), str(best["strategy"])
    model, threshold = fitted[(name, strategy)]

    p_test = model.predict_proba(X_test)[:, 1]
    test = evaluate(y_test, p_test, threshold)
    baseline = evaluate_labels(y_test, rule_baseline(X_test))
    importance = feature_importance(model, list(X.columns))

    log(f"\nBest on validation: {name} + {strategy} (threshold {threshold:.3f})")
    log(f"Test set  -> precision {test['precision']:.3f}, recall {test['recall']:.3f}, "
        f"F1 {test['f1']:.3f}, ROC-AUC {test['roc_auc']:.3f}")
    log(f"Rule-based baseline -> precision {baseline['precision']:.3f}, recall {baseline['recall']:.3f}, "
        f"F1 {baseline['f1']:.3f}")

    joblib.dump({"model": model, "threshold": threshold, "feature_columns": list(X.columns),
                 "model_name": name, "strategy": strategy,
                 "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, model_path)
    comparison.round(4).to_csv(model_path.parent / "comparison.csv", index=False)
    summary = {"best_model": name, "best_strategy": strategy, "threshold": threshold,
               "rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
               "fraud_share": float(y.mean()), "test": test, "rule_baseline": baseline,
               "feature_importance": importance}
    (model_path.parent / "metrics.json").write_text(json.dumps(summary, indent=2))
    _save_plots(report_dir, f"{name} + {strategy}", comparison, y_test, p_test, test, baseline, importance)
    log(f"\nSaved the model to {model_path}, results to {model_path.parent}, charts to {report_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and compare fraud models.")
    parser.add_argument("--data", default=str(config.DATA_PATH), help="transactions CSV")
    parser.add_argument("--models", nargs="+", default=config.MODELS, choices=MODEL_NAMES)
    parser.add_argument("--strategies", nargs="+", default=config.STRATEGIES,
                        choices=["none", "class_weight", "smote", "undersample"])
    args = parser.parse_args()
    train_and_save(data.load_transactions(args.data), models=args.models, strategies=args.strategies)


if __name__ == "__main__":
    main()
