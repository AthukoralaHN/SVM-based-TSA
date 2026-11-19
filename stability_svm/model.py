"""SVM pipeline, tuning on the validation set, saving and loading."""
from datetime import datetime
from itertools import product

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

from .config import BUSES, CLIP_LIMIT, FEATURES, MODEL_FILE, N_SAMPLES, PARAM_GRID, SEED, STABLE, UNSTABLE


def build_model(scaler="robust_clip", C=10.0, gamma="scale"):
    """Scaler -> RBF-kernel SVM with class weights balanced for the ~18% unstable cases.

    scaler = "standard"    : subtract mean, divide by standard deviation
    scaler = "robust_clip" : subtract median, divide by interquartile range, then limit
                             to +/-CLIP_LIMIT (keeps impossible readings from dominating)
    """
    if scaler == "standard":
        steps = [("scale", StandardScaler())]
    elif scaler == "robust_clip":
        steps = [("scale", RobustScaler()),
                 ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -CLIP_LIMIT, "a_max": CLIP_LIMIT}))]
    else:
        raise ValueError(f"unknown scaler {scaler!r}")
    steps.append(("svm", SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")))
    return Pipeline(steps)


def instability_score(model, X):
    """Signed distance from the SVM boundary, oriented so that > 0 means Unstable.

    The further from 0, the more clearly the case sits on that side.
    """
    d = model.decision_function(X)
    # decision_function > 0 points to model.classes_[1]
    return -d if model.classes_[1] == STABLE else d


def _score_combo(scaler, C, gamma, X_train, y_train, X_val, y_val):
    m = build_model(scaler, C, gamma).fit(X_train, y_train)
    p_tr, p_val = m.predict(X_train), m.predict(X_val)
    return {
        "scaler": scaler, "C": C, "gamma": gamma,
        "train_bal_acc": balanced_accuracy_score(y_train, p_tr),
        "val_bal_acc": balanced_accuracy_score(y_val, p_val),
        "val_unstable_recall": recall_score(y_val, p_val, pos_label=UNSTABLE),
        "val_stable_recall": recall_score(y_val, p_val, pos_label=STABLE),
        "n_support_vectors": int(m["svm"].n_support_.sum()),
    }


def tune_with_cv(X_train, y_train, groups, grid=PARAM_GRID, n_splits=5, seed=SEED, n_jobs=-1):
    """Choose scaler, C and gamma by grouped k-fold cross-validation inside the training set.

    Each fold holds out about 1/n_splits of the training contingencies (all their load
    levels together). Averaging over several held-out groups is far less noisy than scoring
    on one small validation set.
    Returns (results table sorted best-first, best parameter dict).
    """
    rows = []
    for scaler in grid["scaler"]:
        search = GridSearchCV(build_model(scaler=scaler), {"svm__C": grid["C"], "svm__gamma": grid["gamma"]},
                              cv=StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed),
                              scoring="balanced_accuracy", n_jobs=n_jobs)
        search.fit(X_train, y_train, groups=groups)
        r = search.cv_results_
        for i, p in enumerate(r["params"]):
            rows.append({"scaler": scaler, "C": p["svm__C"], "gamma": p["svm__gamma"],
                         "cv_bal_acc": r["mean_test_score"][i], "cv_std": r["std_test_score"][i]})
    res = pd.DataFrame(rows)
    res["gamma_label"] = res["gamma"].astype(str)
    res = res.sort_values(["cv_bal_acc", "cv_std"], ascending=[False, True]).reset_index(drop=True)
    best = res.iloc[0]
    return res, {"scaler": best["scaler"], "C": float(best["C"]), "gamma": best["gamma"]}


def tune_on_validation(X_train, y_train, X_val, y_val, grid=PARAM_GRID, n_jobs=-1):
    """Fit every grid combination on the training set and score it on the validation set.

    Used for comparison only: with ~100 validation contingencies this choice is noisy.
    Best = highest validation balanced accuracy; ties go to higher unstable recall, then to
    fewer support vectors (a simpler boundary).
    Returns (results table sorted best-first, best parameter dict).
    """
    combos = list(product(grid["scaler"], grid["C"], grid["gamma"]))
    rows = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_score_combo)(s, c, g, X_train, y_train, X_val, y_val) for s, c, g in combos)
    res = pd.DataFrame(rows)
    res["gamma_label"] = res["gamma"].astype(str)
    res = res.sort_values(["val_bal_acc", "val_unstable_recall", "n_support_vectors"],
                          ascending=[False, False, True]).reset_index(drop=True)
    best = res.iloc[0]
    return res, {"scaler": best["scaler"], "C": float(best["C"]), "gamma": best["gamma"]}


def save_model(model, params, metrics, train_df, path=MODEL_FILE, notes=""):
    """Save the fitted pipeline with what predict.py needs to use and sanity-check it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "params": params,
        "features": FEATURES,
        "buses": BUSES,
        "n_samples": N_SAMPLES,
        "class_names": {STABLE: "Stable", UNSTABLE: "Unstable"},
        "metrics": metrics,
        "train_min": train_df[FEATURES].min().to_dict(),
        "train_max": train_df[FEATURES].max().to_dict(),
        "n_training_rows": len(train_df),
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    }, path)
    return path


def load_model(path=MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run train_model.py first.")
    return joblib.load(path)
