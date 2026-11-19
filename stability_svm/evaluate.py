"""Metrics, per-group breakdowns, feature importance and learning curve."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from .config import BUSES, CLASS_NAMES, GROUP, LABEL, N_SAMPLES, SEED, STABLE, UNSTABLE
from .data import get_xy
from .model import build_model, instability_score


def classification_metrics(y_true, y_pred, score=None):
    """Metrics with Unstable as the class of interest. `score` > 0 means Unstable."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "unstable_recall": recall_score(y_true, y_pred, pos_label=UNSTABLE),
        "unstable_precision": precision_score(y_true, y_pred, pos_label=UNSTABLE, zero_division=0),
        "unstable_f1": f1_score(y_true, y_pred, pos_label=UNSTABLE),
        "stable_recall": recall_score(y_true, y_pred, pos_label=STABLE),
    }
    if score is not None:
        is_unstable = np.asarray(y_true) == UNSTABLE
        m["roc_auc"] = roc_auc_score(is_unstable, score)
        m["average_precision"] = average_precision_score(is_unstable, score)
    return m


def evaluate(model, df):
    """Predictions, instability scores and metrics of `model` on dataframe `df`."""
    X, y = get_xy(df)
    pred = model.predict(X)
    score = instability_score(model, X)
    return pred, score, classification_metrics(y, pred, score)


def confusion(y_true, y_pred):
    """2x2 confusion matrix as a table, rows = actual, columns = predicted (Unstable first)."""
    order = [UNSTABLE, STABLE]
    cm = confusion_matrix(y_true, y_pred, labels=order)
    names = [CLASS_NAMES[c] for c in order]
    return pd.DataFrame(cm, index=[f"actual {n}" for n in names], columns=[f"predicted {n}" for n in names])


def breakdown(df, y_pred, by):
    """Accuracy, unstable cases found / missed and false alarms per value of column `by`."""
    d = pd.DataFrame({by: df[by].to_numpy(), "y": df[LABEL].to_numpy(), "pred": np.asarray(y_pred)})
    d["correct"] = d["y"] == d["pred"]
    d["unstable"] = d["y"] == UNSTABLE
    d["unstable_found"] = d["unstable"] & (d["pred"] == UNSTABLE)
    d["unstable_missed"] = d["unstable"] & (d["pred"] == STABLE)
    d["false_alarm"] = ~d["unstable"] & (d["pred"] == UNSTABLE)
    t = d.groupby(by).agg(cases=("correct", "size"), accuracy=("correct", "mean"),
                          unstable=("unstable", "sum"), unstable_found=("unstable_found", "sum"),
                          unstable_missed=("unstable_missed", "sum"), false_alarms=("false_alarm", "sum"))
    t["accuracy"] = t["accuracy"].round(4)
    return t


def misclassified(df, y_pred, score):
    """Rows the model got wrong, with their scenario and score."""
    cols = [c for c in ["Case_ID", "Operating_Condition", "Contingency_Type", GROUP] if c in df.columns]
    out = df[cols].copy()
    out["actual"] = df[LABEL].map(CLASS_NAMES)
    out["predicted"] = pd.Series(y_pred, index=df.index).map(CLASS_NAMES)
    out["instability_score"] = np.round(score, 3)
    return out[df[LABEL].to_numpy() != np.asarray(y_pred)]


def grouped_permutation_importance(model, df, n_repeats=10, seed=SEED):
    """Drop in balanced accuracy when one bus's voltage (or frequency) readings are shuffled.

    The 4 samples of a bus/quantity are shuffled together, so the score shows how much the
    model relies on that measurement as a whole.
    """
    X, y = get_xy(df)
    base = balanced_accuracy_score(y, model.predict(X))
    rng = np.random.default_rng(seed)
    rows = []
    for bus in BUSES:
        for qty, tag in (("voltage", "Vmag"), ("frequency", "FreqHz")):
            cols = [f"{bus}_{tag}_{k}" for k in range(1, N_SAMPLES + 1)]
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].to_numpy()[perm]
                drops.append(base - balanced_accuracy_score(y, model.predict(Xp)))
            rows.append({"bus": bus, "quantity": qty, "feature_group": f"{bus} {qty}",
                         "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def learning_curve_on_validation(train_df, val_df, params, fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0), seed=SEED,
                                 make_model=None):
    """Train on growing shares of the training contingencies, score on the validation set.

    make_model: optional function returning a fresh unfitted model; default is the SVM with `params`.
    """
    rng = np.random.default_rng(seed)
    contingencies = rng.permutation(np.asarray(train_df[GROUP].unique(), dtype=object))
    X_val, y_val = get_xy(val_df)
    rows = []
    for f in fractions:
        keep = set(contingencies[: max(2, int(round(f * len(contingencies))))])
        sub = train_df[train_df[GROUP].isin(keep)]
        X_s, y_s = get_xy(sub)
        if y_s.nunique() < 2:
            continue
        m = (make_model() if make_model else build_model(**params)).fit(X_s, y_s)
        rows.append({"fraction": f, "training_rows": len(sub),
                     "train_bal_acc": balanced_accuracy_score(y_s, m.predict(X_s)),
                     "val_bal_acc": balanced_accuracy_score(y_val, m.predict(X_val))})
    return pd.DataFrame(rows)
