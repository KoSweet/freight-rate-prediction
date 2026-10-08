"""Step 3 evidence: does training with or without the bad-price rows predict better?"""
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
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})

s2 = pd.read_csv(D / "step2_train.csv", parse_dates=["date"])
s3 = pd.read_csv(D / "step3_train.csv", parse_dates=["date"])
removed = pd.read_csv(D / "step3_train_removed.csv", parse_dates=["date"])

def X(d):
    ld = np.log(d.distance.values)
    return np.column_stack([np.ones(len(d)), ld, ld**2, d.weight.values/1e4, (d.equipment=="Reefer").values, (d.equipment=="Flatbed").values, d.market_index.values])
def fit(d):
    b, *_ = np.linalg.lstsq(X(d), np.log(d.posted_rate.values), rcond=None); return b
def errors(b, d):
    pred = np.exp(X(d) @ b); return np.abs(pred - d.posted_rate.values) / d.posted_rate.values * 100

m = s2.date.dt.month
train_with, train_without = s2[m <= 8], s3[s3.date.dt.month <= 8]
test_clean = s3[s3.date.dt.month >= 9]          # Sep-Oct genuine prices
test_all = s2[m >= 9]                            # Sep-Oct including the bad prices
b_with, b_without = fit(train_with), fit(train_without)
rows = []
for name, b in [("trained WITH bad prices", b_with), ("trained WITHOUT bad prices", b_without)]:
    ec, ea = errors(b, test_clean), errors(b, test_all)
    rows.append((name, np.mean(ec), np.median(ec), np.mean(ea)))
    print(f"{name:28s} Sep-Oct genuine rows: mean error {np.mean(ec):.2f}%  median {np.median(ec):.2f}%   | incl. bad rows: mean {np.mean(ea):.2f}%")
# robustness: least squares on log price is already fairly robust; show the bias the bad rows induce
bias_with = np.mean(np.exp(X(test_clean) @ b_with) / test_clean.posted_rate.values - 1) * 100
bias_without = np.mean(np.exp(X(test_clean) @ b_without) / test_clean.posted_rate.values - 1) * 100
print(f"average over/under-pricing on genuine rows: with {bias_with:+.2f}%   without {bias_without:+.2f}%")

# ---- leaf-average model in DOLLARS (what a default squared-error tree does inside each leaf)
def leaf_model(train_df, test_df):
    cells = ["equipment", pd.cut(train_df.distance, DIST), train_df.date.dt.month.rename("m")]
    tbl = train_df.groupby(cells, observed=True).apply(lambda g: (g.posted_rate / g.distance).mean()).rename("rpm_mean")
    tbl.index = tbl.index.set_names(["equipment", "dbin", "m"])
    key = pd.DataFrame({"equipment": test_df.equipment.values, "dbin": pd.cut(test_df.distance, DIST), "m": np.minimum(test_df.date.dt.month.values, 8)})
    key["m"] = 8  # test months are unseen; use the latest training month's leaf
    rpm = key.join(tbl, on=["equipment", "dbin", "m"]).rpm_mean.values
    return rpm * test_df.distance.values
DIST = [0,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2300,2600,3000,4000]
leaf_rows = []
for name, tr in [("with", train_with), ("without", train_without)]:
    pred = leaf_model(tr, test_clean)
    e = np.abs(pred - test_clean.posted_rate.values) / test_clean.posted_rate.values * 100
    leaf_rows.append((name, np.nanmean(e)))
    print(f"dollar leaf-average model trained {name:8s} bad prices: Sep-Oct genuine mean error {np.nanmean(e):.2f}%")

# ---- chart 13: before/after + impact
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
ax = axes[0]
ax.scatter(s2.distance, s2.posted_rate, s=3, alpha=0.25, color=BLUE, lw=0)
ax.scatter(removed.distance, removed.posted_rate, s=7, color=ORANGE, lw=0)
ax.set(xscale="log", yscale="log", xlabel="Distance (miles, log)", ylabel="Posted rate ($, log)", title=f"Before step 3: {len(s2):,} rows, 677 bad prices in orange")
ax = axes[1]
ax.scatter(s3.distance, s3.posted_rate, s=3, alpha=0.25, color=BLUE, lw=0)
ax.set(xscale="log", yscale="log", xlabel="Distance (miles, log)", title=f"After step 3: {len(s3):,} rows, one clean band")
ax.set_ylim(axes[0].get_ylim())
ax = axes[2]
groups = ["log-price\nmodel", "dollar-average\nmodel"]
with_v = [rows[0][1], leaf_rows[0][1]]; without_v = [rows[1][1], leaf_rows[1][1]]
x = np.arange(2); w = 0.38
b1 = ax.bar(x - w/2, with_v, w, color=ORANGE, label="trained with bad prices")
b2 = ax.bar(x + w/2, without_v, w, color=BLUE, label="trained without")
for bar, v in list(zip(b1, with_v)) + list(zip(b2, without_v)):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.08, f"{v:.2f}%", ha="center", fontsize=9.5, color=INK)
ax.set(xticks=x, xticklabels=groups, ylabel="Mean error on Sep-Oct genuine prices (%)", title="Trained Jan-Aug, tested Sep-Oct", ylim=(0, max(with_v + without_v) * 1.3))
ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(w_pad=2); fig.savefig(OUT / "13_step3_before_after.png", bbox_inches="tight"); plt.close(fig); print("wrote 13")

# ---- are the removed rows random?  share by month / equipment / distance bucket vs kept
print("\nremoved share by month:", (removed.date.dt.month.value_counts().sort_index() / s2.date.dt.month.value_counts().sort_index()).round(3).to_dict())
print("removed share by equipment:", (removed.equipment.value_counts() / s2.equipment.value_counts()).round(3).to_dict())
db = pd.cut(s2.distance, [0,400,800,1500,2500,4000]); dbr = pd.cut(removed.distance, [0,400,800,1500,2500,4000])
print("removed share by distance:", (dbr.value_counts().sort_index() / db.value_counts().sort_index()).round(3).to_dict())
print("removed share by pickup city: min %.3f max %.3f" % tuple((removed.pickup.value_counts() / s2.pickup.value_counts()).fillna(0).agg(["min","max"])))
