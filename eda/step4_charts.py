"""Step 4 charts: city-effect map with fallbacks, and how each feature relates to price per mile."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
CLEAN, ART, OUT = ROOT / "cleaned-datasets", ROOT / "cleaned-datasets" / "artifacts", ROOT / "eda" / "figures"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 150})

eff = pd.read_csv(ART / "city_effects.csv", index_col=0).effect
ct = pd.read_csv(ART / "city_table.csv", index_col=0)
Xv = pd.read_csv(CLEAN / "step4_validation_features.csv"); val = pd.read_csv(CLEAN / "step3_validation.csv")
unseen = sorted((set(val.pickup) | set(val.delivery)) - set(eff.index))
fb_vals = {c: Xv.loc[val.pickup == c, "pickup_city_effect"].iloc[0] if (val.pickup == c).any() else Xv.loc[val.delivery == c, "delivery_city_effect"].iloc[0] for c in unseen}

# ---- 14: city effect map
fig, ax = plt.subplots(figsize=(10.5, 6.2))
known = ct.loc[eff.index]
v = eff.values * 100; lim = np.abs(v).max()
sc = ax.scatter(known.lon, known.lat, c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=120, edgecolor=INK2, lw=0.5, zorder=3)
for c in unseen:
    ax.scatter(ct.loc[c, "lon"], ct.loc[c, "lat"], marker="D", s=110, c=[[fb_vals[c] * 100]], cmap="RdBu_r", vmin=-lim, vmax=lim, edgecolor=INK, lw=1.6, zorder=4)
    ax.annotate(f"{c}\n{fb_vals[c]*100:+.1f}% (fallback)", (ct.loc[c, "lon"], ct.loc[c, "lat"]), xytext=(8, -4), textcoords="offset points", fontsize=8.5, fontweight="bold")
for c in eff.abs().nlargest(6).index:
    ax.annotate(f"{c} {eff[c]*100:+.1f}%", (ct.loc[c, "lon"], ct.loc[c, "lat"]), xytext=(7, 5), textcoords="offset points", fontsize=8.5, color=INK2)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02); cb.set_label("City price effect (%), applied at each end of a lane")
ax.set(xlabel="Longitude", ylabel="Latitude", aspect=1.25, title="City effects learned from training (circles) and nearest-3 fallbacks for unseen cities (diamonds)")
fig.tight_layout(); fig.savefig(OUT / "14_step4_city_effects_map.png", bbox_inches="tight"); plt.close(fig); print("wrote 14")

# ---- 15: how each feature relates to price per mile (training)
Xt = pd.read_csv(CLEAN / "step4_train_features.csv")
log_rpm = np.log(Xt.posted_rate / Xt.distance)
feats = [c for c in Xt.columns if c not in ("load_id", "posted_rate", "date")]
corr = pd.Series({c: np.corrcoef(Xt[c], log_rpm)[0, 1] for c in feats if Xt[c].std() > 0}).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
colors = [BLUE if x > 0 else ORANGE for x in corr.values]
ax.barh(corr.index, corr.values, color=colors)
for i, (n, x) in enumerate(corr.items()):
    ax.text(x + (0.01 if x >= 0 else -0.01), i, f"{x:+.2f}", va="center", ha="left" if x >= 0 else "right", fontsize=8.5, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set(xlabel="Correlation with log(rate per mile), training rows", title="Which features move the price per mile, and in which direction")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "15_step4_feature_correlations.png", bbox_inches="tight"); plt.close(fig); print("wrote 15")

# ---- 16: December feature inputs (what will drive the chart)
Xd = pd.read_csv(CLEAN / "step4_december_features.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(Xd.date, Xd.mi_short, color=BLUE, lw=2, marker="o", ms=5)
for d, m, wk in zip(Xd.date, Xd.mi_short, Xd.date.dt.dayofweek):
    if wk in (5, 6): ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=GRID, lw=0)
ax.axhline(0, color=INK2, lw=1, ls="--")
ax.set(ylabel="mi_short", xlabel="December 2025 (weekends shaded)", title="The only December inputs that change day to day: short-term market signal and weekday")
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d"))
fig.tight_layout(); fig.savefig(OUT / "16_step4_december_inputs.png", bbox_inches="tight"); plt.close(fig); print("wrote 16")
