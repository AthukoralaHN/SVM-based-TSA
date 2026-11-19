# Transient stability prediction - IEEE 39-bus system (SVM)

Predicts whether the IEEE 39-bus system stays **stable** after a contingency from 4 voltage and 4 frequency samples
at each of the 10 generator buses (B30-B39), i.e. 80 inputs. Label `Overall_Stability` comes from the TSI
(1 = stable, 0 = unstable).

**First try:** all 3,715 cases, outliers kept.

## Contents

| Path | What it is |
|---|---|
| `transient_stability_svm.ipynb` | Full analysis with all graphs - start here |
| `stability_svm/` | Python package used by the notebook and scripts (`data`, `model`, `evaluate`, `plots`, `predict`, `config`) |
| `train_model.py` | Runs the whole procedure from the command line and saves model, figures and reports |
| `predict.py` | Predicts new cases from a CSV |
| `models/stability_svm.joblib` | Trained model |
| `figures/` | All graphs (PNG) |
| `reports/` | Text report, tuning grids, test-set predictions |
| `examples/` | Example input files for `predict.py` |
| `report/` | Full LaTeX project report (`transient_stability_svm_report.tex` + figures) and the compiled PDF |
| `paper/` | IEEE conference paper (max. 6 pages) with its own column-width figures and the compiled PDF |

## Procedure

1. Split by `Contingency_ID` into train / validation / test (about 70 / 15 / 15) - all 5 load levels of a
   contingency stay in one set.
2. Choose scaler, `C` and `gamma` of an RBF SVM (`class_weight="balanced"`) by grouped 5-fold cross-validation
   inside the training set.
3. Check the chosen model on the validation set.
4. Refit on train + validation, evaluate once on the test set.

## Run

```
py -m pip install -r requirements.txt
py train_model.py
py predict.py examples\example_case_unstable.csv
```

`predict.py` accepts either one row per case with the 80 training column names, or one case as a 10-row table:

```
Bus,V1,V2,V3,V4,F1,F2,F3,F4
B30,1.0008,0.9995,0.9988,0.9985,49.9762,49.9146,49.8779,49.8554
...
B39,...
```

The saved model must be loaded with the scikit-learn version it was trained with (1.9.1, see `requirements.txt`).
