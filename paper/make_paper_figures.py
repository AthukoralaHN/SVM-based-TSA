"""Figures for the IEEE conference paper: single-column width (3.45 in), vector PDF.

Uses the saved SVM model and the same data split. The validation-check model (fitted on the
training set with the chosen settings) is refitted here; the final model is loaded as saved.
Run from any folder:  py paper/make_paper_figures.py
"""
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAPER_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from stability_svm import data, evaluate, model, plots  # noqa: E402
from stability_svm.config import LABEL, MODEL_FILE, STABLE, UNSTABLE  # noqa: E402

FIG_DIR = PAPER_DIR / "figures"
COL = 3.45                                   # IEEE two-column layout: column width in inches
TEST_C, VAL_C = "#6250d6", "#1baf7a"         # test / validation series (validated pair)
ST_C, UN_C = plots.STABLE_COLOR, plots.UNSTABLE_COLOR


def style():
    plots.apply_style()
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 8, "axes.titlelocation": "center", "axes.titleweight": "bold",
        "axes.titlepad": 4, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def no_frame(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_fault_time(df):
    d = df.dropna(subset=["Fault_Time_s"])
    rows = ["SLG", "LLG", "3PHG", "Line trip"]
    pv = d.pivot_table(index="Fault_Kind", columns="Fault_Time_s", values=LABEL,
                       aggfunc=lambda s: 100 * (s == UNSTABLE).mean()).reindex(rows)
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    im = ax.imshow(pv.values, cmap=plots.ORANGE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [f"{t:g} s" for t in pv.columns])
    ax.set_yticks(range(pv.shape[0]), rows)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else plots.INK)
    ax.set_xlabel("fault clearing time")
    no_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("unstable cases (%)", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "fig_fault_time")


def fig_traces(df):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.55), sharex=True)
    x = np.arange(1, 5)
    for c, bus in enumerate(("B31", "B34")):
        for r, (tag, unit) in enumerate((("Vmag", "voltage (p.u.)"), ("FreqHz", "frequency (Hz)"))):
            ax = axes[r, c]
            cols = [f"{bus}_{tag}_{k}" for k in x]
            for cls, color in ((STABLE, ST_C), (UNSTABLE, UN_C)):
                v = df.loc[df[LABEL] == cls, cols]
                ax.fill_between(x, v.quantile(0.25), v.quantile(0.75), color=color, alpha=0.2, linewidth=0)
                ax.plot(x, v.median(), color=color, marker="o", markeredgecolor="white", markeredgewidth=0.8)
            ax.set_title(f"{bus} {unit.split(' ')[0]}")
            if c == 0:
                ax.set_ylabel(unit)
            if r == 1:
                ax.set_xlabel("sample")
            ax.set_xticks(x)
    fig.legend(handles=[Patch(color=ST_C, label="Stable"), Patch(color=UN_C, label="Unstable")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_traces")


def fig_cv_grid(cv):
    sub = cv[cv["scaler"] == "robust_clip"]
    Cs = [0.1, 1, 10, 100, 1000]
    gammas = ["scale", "0.001", "0.003", "0.01", "0.03", "0.1"]
    pv = sub.pivot_table(index="C", columns="gamma_label", values="cv_bal_acc").reindex(index=Cs, columns=gammas)
    best = sub.iloc[0]
    vmin, vmax = cv["cv_bal_acc"].min(), cv["cv_bal_acc"].max()
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    im = ax.imshow(pv.values, cmap=plots.BLUE_RAMP, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(gammas)), gammas)
    ax.set_yticks(range(len(Cs)), [f"{c:g}" for c in Cs])
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("C")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.55 else plots.INK)
    i, j = Cs.index(float(best["C"])), gammas.index(str(best["gamma_label"]))
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=plots.INK, linewidth=1.6))
    no_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("CV balanced accuracy", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "fig_cv_grid")


def fig_confusion(y, preds, pairs=None):
    """Confusion matrices side by side. `pairs`: dict title -> (y_true, y_pred); otherwise one y for all preds."""
    pairs = pairs or {name: (y, p) for name, p in preds.items()}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
    for ax, (name, (yt, p)) in zip(axes, pairs.items()):
        cm = evaluate.confusion(yt, p).to_numpy()
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=plots.BLUE_RAMP, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else plots.INK)
        ax.set_xticks([0, 1], ["Unstable", "Stable"])
        ax.set_yticks([0, 1], ["Unstable", "Stable"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("actual")
        ax.set_title(name)
        no_frame(ax)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_confusion")


def fig_roc(curves):
    """`curves`: dict name -> (y_true, score). Zoomed to the top-left corner."""
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.plot([0, 1], [0, 1], color=plots.AXIS, linewidth=0.8)
    for name, (y, s) in curves.items():
        is_u = np.asarray(y) == UNSTABLE
        fpr, tpr, _ = roc_curve(is_u, s)
        ax.plot(fpr, tpr, color=TEST_C if name == "Test" else VAL_C,
                label=f"{name} (AUC = {roc_auc_score(is_u, s):.3f})")
    ax.set_xlim(-0.005, 0.25)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("false positive rate (stable cases flagged)")
    ax.set_ylabel("true positive rate (unstable found)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_roc")


def fig_importance(imp, top=10):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    y = np.arange(len(t))
    ax.barh(y, t["importance"], height=0.62, xerr=t["std"], color=TEST_C, edgecolor="white", linewidth=0.6,
            error_kw={"ecolor": plots.INK_2, "elinewidth": 0.6, "capsize": 0})
    ax.set_yticks(y, t["feature_group"])
    ax.axvline(0, color=plots.AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("drop in test balanced accuracy when shuffled")
    save(fig, "fig_importance")


def main():
    style()
    df = data.add_scenario_details(data.load_dataset())
    train, val, test = data.split_train_val_test(df)
    bundle = model.load_model(MODEL_FILE)
    final = bundle["model"]
    selection = model.build_model(**bundle["params"]).fit(*data.get_xy(train))   # validation-check model
    val_pred, val_score, m_val = evaluate.evaluate(selection, val)
    test_pred, test_score, m_test = evaluate.evaluate(final, test)
    cv = pd.read_csv(PROJECT_DIR / "reports" / "cv_grid.csv")

    fig_fault_time(df)
    fig_traces(df)
    fig_cv_grid(cv)
    fig_confusion(None, None, pairs={"Validation": (val[LABEL], val_pred), "Test": (test[LABEL], test_pred)})
    fig_roc({"Validation": (val[LABEL], val_score), "Test": (test[LABEL], test_score)})
    imp = evaluate.grouped_permutation_importance(final, test)
    fig_importance(imp)
    print("figures written to", FIG_DIR)
    print(f"validation BA {m_val['balanced_accuracy']:.4f}  test BA {m_test['balanced_accuracy']:.4f}")
    print(imp.head(6).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
