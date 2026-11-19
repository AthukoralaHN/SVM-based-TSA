"""Project-wide settings: file locations, feature names, label coding, split sizes, tuning grid."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "IEEE39_voltage_frequency_4sample_SVM_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "stability_svm.joblib"
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"
EXAMPLE_DIR = PROJECT_DIR / "examples"

# 10 generator buses of the IEEE 39-bus system, 4 time samples of each quantity.
BUSES = [f"B{n}" for n in range(30, 40)]
N_SAMPLES = 4
VOLT_FEATURES = [f"{b}_Vmag_{k}" for b in BUSES for k in range(1, N_SAMPLES + 1)]
FREQ_FEATURES = [f"{b}_FreqHz_{k}" for b in BUSES for k in range(1, N_SAMPLES + 1)]
FEATURES = VOLT_FEATURES + FREQ_FEATURES          # 80 model inputs, in this order

LABEL = "Overall_Stability"      # from the TSI value
GROUP = "Contingency_ID"         # one contingency = 5 rows (one per load level)
STABLE, UNSTABLE = 1, 0
CLASS_NAMES = {STABLE: "Stable", UNSTABLE: "Unstable"}

SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

# "robust_clip" scales each feature by median / interquartile range and then limits the
# scaled value to +/-CLIP_LIMIT. No rows are removed; impossible readings stay in the data
# but cannot push the RBF kernel so far that the model stops comparing them with anything.
CLIP_LIMIT = 20.0

PARAM_GRID = {
    "scaler": ["standard", "robust_clip"],
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1],
}
