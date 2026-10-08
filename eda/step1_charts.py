"""Charts for step 1: weight before/after the sign flip, and where the price outliers are (not removed)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN, OUT = ROOT / "data", ROOT / "cleaned-datasets", ROOT / "eda" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})

raw = pd.read_csv(RAW / "train_test.csv")
s1 = pd.read_csv(CLEAN / "step1_train.csv")

# ---- 07: weight before / after sign flip
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
bins = np.arange(-50000, 50001, 2500)
neg, pos = raw.weight[raw.weight < 0], raw.weight[raw.weight > 0]
axes[0].hist(pos, bins=bins, color=BLUE, alpha=0.85, label=f"positive ({len(pos):,})")
axes[0].hist(neg, bins=bins, color=ORANGE, label=f"negative ({len(neg):,})")
axes[0].axvline(0, color=INK2, lw=1, ls="--"); axes[0].legend(frameon=False)
axes[0].set(title="Before: raw weight", xlabel="Weight (lb)", ylabel="Number of loads")
flipped = s1.weight[raw.weight < 0]
axes[1].hist(s1.weight[raw.weight > 0], bins=bins, color=BLUE, alpha=0.85, label=f"unchanged ({len(pos):,})")
axes[1].hist(flipped, bins=bins, color=ORANGE, label=f"sign flipped ({len(flipped)})")
axes[1].axvline(0, color=INK2, lw=1, ls="--"); axes[1].legend(frameon=False)
axes[1].set(title="After step 1: sign flipped, blanks still blank", xlabel="Weight (lb)")
q_neg = flipped.quantile([.25, .5, .75]).round(0).astype(int).tolist(); q_pos = pos.quantile([.25, .5, .75]).round(0).astype(int).tolist()
fig.text(0.01, -0.04, f"Check: quartiles of flipped weights {q_neg} vs normal weights {q_pos}. Same shape, so the minus sign was a recording error.", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "07_step1_weight_sign_flip.png", bbox_inches="tight"); plt.close(fig); print("wrote 07")

# ---- 08: weight outliers after the flip? box plot per equipment
fig, ax = plt.subplots(figsize=(8.5, 3.8))
order = ["Dry Van", "Flatbed", "Reefer"]
bp = ax.boxplot([s1.loc[s1.equipment == e, "weight"].dropna() for e in order], tick_labels=order, vert=False, widths=0.55, patch_artist=True,
                medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                flierprops=dict(marker=".", markersize=4, markerfacecolor=ORANGE, markeredgecolor="none"))
for patch, c in zip(bp["boxes"], [BLUE, ORANGE, AQUA]): patch.set(facecolor=c, alpha=0.55, edgecolor=c)
ax.set(xlabel="Weight (lb)", title=f"Weight after step 1: range {int(s1.weight.min()):,} to {int(s1.weight.max()):,} lb, no outliers")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "08_step1_weight_outliers.png", bbox_inches="tight"); plt.close(fig); print("wrote 08")

# ---- 09: price outliers, flagged only (nothing removed)
t = s1.copy(); t["rpm"] = t.posted_rate / t.distance
t["dbin"] = pd.cut(t.distance, [0,200,400,600,800,1000,1500,2000,2500,4000])
core = t[(t.rpm > 0.8) & (t.rpm < 5)]
exp = core.groupby(["equipment", "dbin"], observed=True).rpm.median().rename("exp_rpm")
t = t.join(exp, on=["equipment", "dbin"]); t["ratio"] = t.posted_rate / (t.exp_rpm * t.distance)
hi, lo, ok = t[t.ratio > 2], t[t.ratio < 0.5], t[(t.ratio >= 0.5) & (t.ratio <= 2)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ax.scatter(ok.distance, ok.posted_rate, s=3, alpha=0.25, color=BLUE, lw=0, label=f"normal ({len(ok):,})")
ax.scatter(hi.distance, hi.posted_rate, s=8, color=ORANGE, lw=0, label=f"outlier: too high ({len(hi)})")
ax.scatter(lo.distance, lo.posted_rate, s=8, color=AQUA, lw=0, label=f"outlier: too low ({len(lo)})")
ax.set(xscale="log", yscale="log", xlabel="Distance (miles, log)", ylabel="Posted rate ($, log)", title="Price outliers: two bands far from the main cloud (still in the data)")
ax.legend(frameon=False, loc="upper left", markerscale=2.5)
ax = axes[1]
lb = np.logspace(np.log10(0.12), np.log10(6), 70)
ax.hist(ok.ratio, bins=lb, color=BLUE, alpha=0.85, label="normal")
ax.hist(pd.concat([hi.ratio, lo.ratio]), bins=lb, color=ORANGE, label="outlier")
for x, lab in [(0.5, " 0.5x"), (2.0, " 2x")]:
    ax.axvline(x, color=INK, ls="--", lw=1); ax.text(x, 1.5e4, lab, color=INK, fontsize=9)
ax.set(xscale="log", yscale="log", xlabel="Actual price / expected price (log)", ylabel="Number of loads (log)", title="Tight peak at 1x, two detached tails")
ax.set_xticks([0.2, 0.33, 0.5, 1, 2, 3, 5]); ax.set_xticklabels(["0.2x", "0.33x", "0.5x", "1x", "2x", "3x", "5x"])
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "09_price_outliers_flagged.png", bbox_inches="tight"); plt.close(fig); print("wrote 09")

print(f"\nflagged outliers: {len(hi)+len(lo)} of {len(t):,} ({(len(hi)+len(lo))/len(t):.1%})  too high {len(hi)}  too low {len(lo)}")
print("weight after step 1: min", s1.weight.min(), "max", s1.weight.max(), " blanks", s1.weight.isna().sum())
print("\nsample outlier rows:")
print(pd.concat([hi.head(3), lo.head(3)])[["load_id","pickup","delivery","distance","equipment","posted_rate","ratio"]].round(2).to_string(index=False))
