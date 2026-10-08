"""Six exploratory charts for the freight-rate assessment.

Run:  python eda/eda_charts.py
Writes PNGs to eda/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (validated categorical slots + text/grid tokens)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#9a9891", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- load
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
train["rpm"] = train.posted_rate / train.distance
LO, HI = 0.8, 5.0  # rate-per-mile band outside which we call a label an outlier
train["outlier"] = (train.rpm < LO) | (train.rpm > HI)

# ---------------------------------------------------------------- 1. rate vs distance (log-log)
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ok, bad = train[~train.outlier], train[train.outlier]
ax.scatter(ok.distance, ok.posted_rate, s=3, alpha=0.25, color=BLUE, lw=0, label=f"normal loads (n={len(ok):,})")
ax.scatter(bad.distance, bad.posted_rate, s=7, alpha=0.8, color=ORANGE, lw=0, label=f"rate/mile outside {LO}-{HI} $/mi (n={len(bad):,})")
xs = np.array([70, 3500])
med = train.rpm.median()
ax.plot(xs, xs * med, color=INK, lw=1.2, ls="--", label=f"median {med:.2f} $/mi")
ax.set(xscale="log", yscale="log", xlabel="Distance (miles, log scale)", ylabel="Posted rate ($, log scale)",
       title=f"Rate is almost a straight line in distance; {train.outlier.mean():.1%} of labels sit far off the line")
ax.legend(frameon=False, loc="upper left", markerscale=3)
save(fig, "01_rate_vs_distance.png")

# ---------------------------------------------------------------- 2. rate per mile by equipment
order = ["Dry Van", "Flatbed", "Reefer"]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
data = [train.loc[train.equipment == e, "rpm"].values for e in order]
bp = ax.boxplot(data, tick_labels=order, showfliers=True, widths=0.5, patch_artist=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.35, markerfacecolor=INK2, markeredgecolor="none"),
                medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for patch, c in zip(bp["boxes"], [BLUE, ORANGE, AQUA]):
    patch.set(facecolor=c, alpha=0.55, edgecolor=c)
for i, e in enumerate(order, 1):
    m = train.loc[train.equipment == e, "rpm"].median()
    ax.annotate(f"median {m:.2f}", (i + 0.27, m), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK2)
ax.set(ylim=(0, 6), ylabel="Rate per mile ($/mi)", title="Reefer and Flatbed earn more per mile than Dry Van")
ax.text(0, -0.14, f"y-axis clipped at 6 $/mi; {int((train.rpm > 6).sum())} loads above are not shown", transform=ax.transAxes, fontsize=9, color=INK2)
save(fig, "02_rpm_by_equipment.png")

# ---------------------------------------------------------------- 3. seasonality by month
clean = train[~train.outlier]
monthly = clean.groupby(clean.date.dt.to_period("M")).rpm.quantile([0.25, 0.5, 0.75]).unstack()
x = monthly.index.to_timestamp()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.fill_between(x, monthly[0.25], monthly[0.75], color=BLUE, alpha=0.15, lw=0, label="interquartile range")
ax.plot(x, monthly[0.5], color=BLUE, lw=2, marker="o", ms=6, label="median rate per mile")
for xi, yi in zip(x, monthly[0.5]):
    ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=ORANGE, alpha=0.12, lw=0)
ax.text(pd.Timestamp("2025-11-03"), monthly[0.75].max(), "validation\nperiod\n(no labels)", fontsize=9, color=ORANGE, va="top")
ax.set(ylabel="Rate per mile ($/mi)", xlabel="Pickup month (2025)", xlim=(pd.Timestamp("2024-12-20"), pd.Timestamp("2026-01-05")),
       title="Prices rise from January to June, then plateau; Nov-Dec is unseen")
ax.legend(frameon=False, loc="lower right")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
save(fig, "03_rpm_by_month.png")

# ---------------------------------------------------------------- 4. rate per mile vs distance (binned)
bins = [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3500]
clean = clean.assign(dbin=pd.cut(clean.distance, bins))
g = clean.groupby("dbin", observed=True).rpm.quantile([0.25, 0.5, 0.75]).unstack()
centers = [iv.mid for iv in g.index]
counts = clean.groupby("dbin", observed=True).size()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.fill_between(centers, g[0.25], g[0.75], color=BLUE, alpha=0.15, lw=0, label="interquartile range")
ax.plot(centers, g[0.5], color=BLUE, lw=2, marker="o", ms=6, label="median rate per mile")
for c, m, n in zip(centers, g[0.5], counts):
    ax.annotate(f"{m:.2f}", (c, m), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set(xlabel="Distance (miles, bin midpoint)", ylabel="Rate per mile ($/mi)",
       title="Short hauls cost far more per mile, so price is not simply rate x distance")
ax.legend(frameon=False)
save(fig, "04_rpm_vs_distance_binned.png")

# ---------------------------------------------------------------- 5. data-quality issues
def issues(df, has_label):
    d = {
        "Missing weight": df.weight.isna().sum(),
        "Missing market_index": df.market_index.isna().sum(),
        "Negative weight": (df.weight < 0).sum(),
    }
    if has_label:
        d[f"Rate/mile outside {LO}-{HI}"] = df.outlier.sum()
    else:
        d[f"Rate/mile outside {LO}-{HI}"] = np.nan
    return pd.Series(d)
q = pd.DataFrame({"train (48,000 rows)": issues(train, True), "validation (12,000 rows)": issues(val, False)})
fig, ax = plt.subplots(figsize=(8.5, 4.2))
y = np.arange(len(q))
h = 0.38
ax.barh(y + h / 2, q.iloc[:, 0], height=h - 0.03, color=BLUE, label=q.columns[0])
ax.barh(y - h / 2, q.iloc[:, 1].fillna(0), height=h - 0.03, color=ORANGE, label=q.columns[1])
for yi, (a, b) in zip(y, q.values):
    ax.text(a + 8, yi + h / 2, f"{int(a):,}", va="center", fontsize=9.5, color=INK2)
    ax.text((0 if np.isnan(b) else b) + 8, yi - h / 2, "n/a (no label)" if np.isnan(b) else f"{int(b):,}", va="center", fontsize=9.5, color=INK2)
ax.set(yticks=y, yticklabels=q.index, xlabel="Number of rows", title="Data-quality issues are small in count but present in both files")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper right")
save(fig, "05_data_quality.png")

# ---------------------------------------------------------------- 6. city map: seen vs unseen
def cities(df):
    p = df[["pickup", "pickup_lat", "pickup_lon"]].rename(columns=lambda c: c.replace("pickup_", "").replace("pickup", "city"))
    d = df[["delivery", "delivery_lat", "delivery_lon"]].rename(columns=lambda c: c.replace("delivery_", "").replace("delivery", "city"))
    return pd.concat([p, d]).groupby("city")[["lat", "lon"]].first()
ct, cv = cities(train), cities(val)
unseen = cv[~cv.index.isin(ct.index)]
val_rows_unseen = ((~val.pickup.isin(ct.index)) | (~val.delivery.isin(ct.index))).sum()
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.scatter(ct.lon, ct.lat, s=36, color=BLUE, alpha=0.85, lw=0, label=f"cities in training data ({len(ct)})")
ax.scatter(unseen.lon, unseen.lat, s=90, color=ORANGE, lw=0, zorder=3, label=f"validation-only cities ({len(unseen)})")
for name, r in unseen.iterrows():
    ax.annotate(name, (r.lon, r.lat), xytext=(7, 4), textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold")
ax.set(xlabel="Longitude", ylabel="Latitude", aspect=1.25,
       title=f"8 validation cities never appear in training ({val_rows_unseen:,} of 12,000 rows touch one)")
ax.legend(frameon=False, loc="lower left")
ax.text(0, -0.16, "Coordinates come from the dataset and are offset from true city locations; they are consistent per city.",
        transform=ax.transAxes, fontsize=9, color=INK2)
save(fig, "06_city_coverage.png")
