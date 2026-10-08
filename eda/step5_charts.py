"""Step 5 chart: forward-fold error per model."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; CLEAN, OUT = ROOT / "cleaned-datasets", ROOT / "eda" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})
res = pd.read_csv(CLEAN / "step5_model_comparison.csv")
keep = {"linear": ("Linear (log price)", BLUE), "gbm (lr .05, 600 it, 31 leaves)": ("Gradient-boosted trees", ORANGE), "linear+gbm (lr .05, 400 it, 15 leaves)": ("Linear + trees on residuals", AQUA)}
res = res[res.model.isin(keep)]; folds = list(dict.fromkeys(res.fold))
fig, ax = plt.subplots(figsize=(10.5, 4.6)); x = np.arange(len(folds)); w = 0.26
for i, (k, (lab, col)) in enumerate(keep.items()):
    v = [res[(res.model == k) & (res.fold == f)].mape.iloc[0] for f in folds]
    bars = ax.bar(x + (i - 1) * w, v, w, color=col, label=f"{lab}  (avg {np.mean(v):.2f}%)")
    for b, val in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", fontsize=8.5, color=INK2)
ax.set(xticks=x, xticklabels=[f.replace(">", " → ") for f in folds], ylabel="Mean absolute % error", ylim=(0, 2.6), title="Forward-in-time folds: the hybrid is best on every fold, pure trees are worst")
ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "17_step5_model_comparison.png", bbox_inches="tight"); print("wrote 17")
