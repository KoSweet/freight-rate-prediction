"""Step 2 check: how accurate are the fills, and where do the filled values land?"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D, OUT = ROOT / "cleaned-datasets", ROOT / "eda" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})

s1 = pd.read_csv(D / "step1_train.csv", parse_dates=["date"])
s2 = pd.read_csv(D / "step2_train.csv", parse_dates=["date"])

# ---- accuracy test: hide 2,000 KNOWN values, fill them the same way, measure the miss
rng = np.random.default_rng(0)
known = s1[s1.market_index.notna()]
idx = rng.choice(known.index, 2000, replace=False)
t = s1.copy(); truth = t.loc[idx, "market_index"].copy(); t.loc[idx, "market_index"] = np.nan
filled = t.groupby("date").market_index.transform("mean").loc[idx]
mi_err = (filled - truth).abs()
w_known = s1[s1.weight.notna()]
w_idx = rng.choice(w_known.index, 2000, replace=False)
w_err = (s1.loc[w_idx, "weight"] - s1.weight.median()).abs()
print("ACCURACY TEST (hide known values, fill, compare)")
print(f"  market_index same-day mean: typical miss {mi_err.median():.3f}, 90% within {mi_err.quantile(.9):.3f}; "
      f"the index itself spans {s1.market_index.min():.2f}-{s1.market_index.max():.2f} and moves {s1.groupby('date').market_index.mean().std():.3f} day to day")
print(f"  weight median fill:         typical miss {w_err.median():,.0f} lb, 90% within {w_err.quantile(.9):,.0f} lb (weight spread is {s1.weight.std():,.0f} lb)")
# does the weight miss matter for price?  slope ~0.3% per 1000 lb -> typical price error from fill
print(f"  -> a {w_err.median():,.0f} lb weight miss moves the price by about {w_err.median()/1000*0.3:.1f}%, below the row-to-row noise of ~6.5%")

# ---- 10: market_index over time with filled points
daily = s2.groupby("date").market_index.mean()
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.scatter(s2.date[s2.market_index_was_missing == 0], s2.market_index[s2.market_index_was_missing == 0], s=2, alpha=0.15, color=BLUE, lw=0, label="recorded values (one dot per load)")
ax.plot(daily.index, daily.values, color=INK, lw=1.2, label="daily mean")
f = s2[s2.market_index_was_missing == 1]
ax.scatter(f.date, f.market_index, s=14, color=ORANGE, lw=0, zorder=3, label=f"filled with same-day mean ({len(f)})")
ax.set(xlabel="Date (2025)", ylabel="market_index", title="market_index is a daily market level; filled values sit on that day's line")
ax.legend(frameon=False, loc="upper right", markerscale=3)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
fig.tight_layout(); fig.savefig(OUT / "10_step2_market_index_fill.png", bbox_inches="tight"); plt.close(fig); print("wrote 10")

# ---- 11: weight fill + accuracy-test histograms
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.arange(5000, 47501, 1500)
axes[0].hist(s2.weight[s2.weight_was_missing == 0], bins=bins, color=BLUE, alpha=0.85, label="recorded")
axes[0].hist(s2.weight[s2.weight_was_missing == 1], bins=bins, color=AQUA, label=f"filled with median 31,496 lb ({int(s2.weight_was_missing.sum())})")
axes[0].set(xlabel="Weight (lb)", ylabel="Number of loads", title="Blank weights filled at the median")
axes[0].legend(frameon=False)
axes[1].hist(mi_err, bins=40, color=BLUE, alpha=0.85)
axes[1].axvline(mi_err.median(), color=INK, ls="--", lw=1)
axes[1].text(mi_err.median(), axes[1].get_ylim()[1]*0.9, f"  typical miss {mi_err.median():.3f}", fontsize=9.5)
axes[1].set(xlabel="|filled value - true value| for 2,000 hidden market_index values", ylabel="Count", title="Accuracy test: same-day mean recovers hidden values closely")
fig.tight_layout(); fig.savefig(OUT / "11_step2_fill_accuracy.png", bbox_inches="tight"); plt.close(fig); print("wrote 11")

# ---- are blank rows priced like others?  (so the blank itself carries no hidden meaning)
s2["rpm"] = s2.posted_rate / s2.distance
for col in ["weight_was_missing", "market_index_was_missing"]:
    g = s2.groupby(col).rpm.median()
    print(f"median rate/mile when {col}=1: {g[1]:.3f}  vs 0: {g[0]:.3f}")
