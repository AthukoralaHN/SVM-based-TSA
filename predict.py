"""Predict Stable / Unstable for new cases with the trained SVM (models/stability_svm.joblib).

Usage
    python predict.py examples/example_case_unstable.csv     one case, 10-row per-generator layout
    python predict.py my_cases.csv                           many cases, one row each (80 columns)

Results are printed and written to <input name>_predictions.csv next to the input file.
See stability_svm/predict.py for the accepted layouts and for predict_case() to use from Python.
"""
import sys
from pathlib import Path

from stability_svm.config import LABEL
from stability_svm.predict import predict_dataframe, read_cases


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    out = predict_dataframe(read_cases(path))
    show = [c for c in ("Case_ID", "Prediction", "Instability_score", "Outside_training_range") if c in out.columns]
    print(out[show].head(30).to_string(index=False))
    if len(out) > 30:
        print(f"... {len(out) - 30} more rows")
    print("\nPredicted:", out["Prediction"].value_counts().to_dict())
    if LABEL in out.columns:
        actual = out[LABEL].map({1: "Stable", 0: "Unstable"})
        print(f"Agreement with the {LABEL} column: {(actual == out['Prediction']).mean():.2%} "
              "(optimistic if these rows were used in training)")
    dest = path.with_name(path.stem + "_predictions.csv")
    out.to_csv(dest, index=False)
    print(f"Saved -> {dest}")


if __name__ == "__main__":
    main()
