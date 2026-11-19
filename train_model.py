"""Train, evaluate and save the transient-stability SVM (command-line version of the notebook).

Steps
  1. Load all 3,715 cases (first try: nothing removed, impossible readings kept).
  2. Split by contingency into train / validation / test (about 70 / 15 / 15).
  3. Choose scaler, C and gamma by grouped 5-fold cross-validation inside the training set.
  4. Check the chosen model on the validation set.
  5. Refit on train + validation and evaluate once on the test set.
  6. Save the model, a text report, test predictions, example inputs and all figures.

Usage
    python train_model.py                      # dataset next to this script
    python train_model.py path/to/dataset.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stability_svm import data, evaluate, model, plots
from stability_svm.config import (BUSES, DATA_FILE, EXAMPLE_DIR, GROUP, LABEL, MODEL_FILE, N_SAMPLES,
                                  REPORT_DIR, STABLE, UNSTABLE)


def write_example(row, path):
    """Save one case in the per-generator layout (10 rows: Bus,V1..V4,F1..F4)."""
    table = pd.DataFrame({"Bus": BUSES})
    for k in range(1, N_SAMPLES + 1):
        table[f"V{k}"] = [row[f"{b}_Vmag_{k}"] for b in BUSES]
    for k in range(1, N_SAMPLES + 1):
        table[f"F{k}"] = [row[f"{b}_FreqHz_{k}"] for b in BUSES]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_FILE
    plots.apply_style()
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    raw = data.load_dataset(data_path)
    log(f"Dataset: {data_path.name}")
    log(data.quality_report(raw).to_string())
    df = data.add_scenario_details(raw)

    train, val, test = data.split_train_val_test(df)
    summary = data.split_summary(train, val, test)
    log("\n== Split (by contingency) ==")
    log(summary.to_string())
    log("Rows with impossible readings (train / validation / test): "
        f"{[int(data.extreme_rows(s).sum()) for s in (train, val, test)]}")

    X_tr, y_tr = data.get_xy(train)
    X_val, y_val = data.get_xy(val)

    log("\n== Choosing settings: grouped 5-fold cross-validation inside the training set ==")
    cv_results, best = model.tune_with_cv(X_tr, y_tr, train[GROUP])
    log(cv_results.drop(columns="gamma").head(10).round(4).to_string(index=False))
    log(f"Chosen: {best}")

    val_results, val_best = model.tune_on_validation(X_tr, y_tr, X_val, y_val)
    vb = cv_results[(cv_results["scaler"] == val_best["scaler"]) & (cv_results["C"] == val_best["C"])
                    & (cv_results["gamma_label"] == str(val_best["gamma"]))].iloc[0]
    log(f"\nFor comparison, choosing on the validation set alone would pick {val_best} "
        f"(validation {val_results.iloc[0]['val_bal_acc']:.4f}), but that setting scores only "
        f"{vb['cv_bal_acc']:.4f} in cross-validation - its validation win is mostly luck.")

    selection_model = model.build_model(**best).fit(X_tr, y_tr)
    _, _, m_train = evaluate.evaluate(selection_model, train)
    val_pred, val_score, m_val = evaluate.evaluate(selection_model, val)

    trainval = pd.concat([train, val])
    final = model.build_model(**best).fit(*data.get_xy(trainval))
    test_pred, test_score, m_test = evaluate.evaluate(final, test)

    m_cv = {"balanced_accuracy": float(cv_results.iloc[0]["cv_bal_acc"]), "std": float(cv_results.iloc[0]["cv_std"])}
    table = pd.DataFrame({"Train": m_train, "Validation": m_val, "Test": m_test}).T
    log("\n== Results ==")
    log(f"Cross-validation on train (chosen setting): balanced accuracy "
        f"{cv_results.iloc[0]['cv_bal_acc']:.4f} +/- {cv_results.iloc[0]['cv_std']:.4f}")
    log("Train and Validation rows: model fitted on the training set.")
    log("Test row: final model refitted on train + validation, scored once on the untouched test set.")
    log(table.round(4).to_string())
    log("\nValidation confusion matrix:")
    log(evaluate.confusion(val[LABEL], val_pred).to_string())
    log("\nTest confusion matrix:")
    log(evaluate.confusion(test[LABEL], test_pred).to_string())
    log("\nTest results by contingency type:")
    log(evaluate.breakdown(test, test_pred, "Contingency_Type").to_string())
    log("\nTest results by load change (%):")
    log(evaluate.breakdown(test, test_pred, "Load_Change_pct").to_string())
    for name, d, p, s in (("validation", val, val_pred, val_score), ("test", test, test_pred, test_score)):
        wrong = evaluate.misclassified(d, p, s)
        log(f"\nMisclassified {name} cases ({len(wrong)}):")
        log(wrong.to_string(index=False) if len(wrong) else "none")

    imp = evaluate.grouped_permutation_importance(final, test)
    lc = evaluate.learning_curve_on_validation(train, val, best)

    model.save_model(final, best, {"cv": m_cv, "train": m_train, "validation": m_val, "test": m_test}, trainval,
                     notes="First try: all rows kept, outliers not removed. Fitted on train + validation.")
    log(f"\nModel saved   -> {MODEL_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = test[["Case_ID", "Operating_Condition", "Contingency_Type", "Contingency_ID", LABEL]].copy()
    out["Predicted"] = test_pred
    out["Instability_score"] = test_score.round(4)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    cv_results.drop(columns="gamma").to_csv(REPORT_DIR / "cv_grid.csv", index=False)
    val_results.drop(columns="gamma").to_csv(REPORT_DIR / "validation_grid.csv", index=False)

    # Example inputs for predict.py. They demonstrate the input format, so they are taken from test
    # cases (unseen by the final model) that the model classifies correctly.
    for cls, name in ((UNSTABLE, "unstable"), (STABLE, "stable")):
        row = test[(test[LABEL].to_numpy() == cls) & (test_pred == cls)].iloc[0]
        write_example(row, EXAMPLE_DIR / f"example_case_{name}.csv")
        log(f"Example input -> {EXAMPLE_DIR / f'example_case_{name}.csv'}  (actual: {name}, {row['Case_ID']})")

    figures = {
        "01_class_balance": plots.plot_class_balance(df),
        "02_unstable_by_contingency_type": plots.plot_unstable_rate(
            df, "Contingency_Type", title="Unstable cases by contingency type"),
        "03_unstable_by_fault_and_time": plots.plot_fault_time_heatmap(df),
        "04_unstable_vs_load": plots.plot_rate_vs_load(df),
        "05_voltage_traces": plots.plot_sample_traces(df, "voltage"),
        "06_frequency_traces": plots.plot_sample_traces(df, "frequency"),
        "07_impossible_readings": plots.plot_extreme_values(df),
        "08_correlation": plots.plot_correlation(df),
        "09_pca": plots.plot_pca(df),
        "10_split": plots.plot_split_composition(summary),
        "11_cv_grid": plots.plot_grid_heatmaps(cv_results, best),
        "12_validation_only_grid": plots.plot_grid_heatmaps(
            val_results, val_best, value="val_bal_acc", value_label="validation balanced accuracy",
            title="For comparison: the same grid scored on the validation set only (black box = its pick)"),
        "13_learning_curve": plots.plot_learning_curve(lc),
        "14_metrics_by_set": plots.plot_metric_dots(table),
        "15_confusion_matrices": plots.plot_confusion_matrices({
            "Validation (model fitted on train)": evaluate.confusion(val[LABEL], val_pred),
            "Test (final model)": evaluate.confusion(test[LABEL], test_pred)}),
        "16_roc_pr_test": plots.plot_roc_pr(test[LABEL], test_score),
        "17_score_distribution_test": plots.plot_score_distribution(test[LABEL], test_score),
        "18_test_by_contingency_type": plots.plot_breakdown_dots(
            evaluate.breakdown(test, test_pred, "Contingency_Type"), "Test accuracy by contingency type"),
        "19_feature_importance": plots.plot_importance(imp),
    }
    for name, fig in figures.items():
        plots.save(fig, name)
        plt.close(fig)
    log(f"Figures saved -> {plots.FIGURE_DIR}")
    (REPORT_DIR / "training_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved  -> {REPORT_DIR / 'training_report.txt'}")


if __name__ == "__main__":
    main()
