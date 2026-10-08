"""The three regimes of quote_signal: clean, mirrored, random. Run: python eda/quote_signal_chart.py"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; CLEAN, OUT = ROOT / "cleaned-datasets", ROOT / "eda" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})
d = pd.read_csv(CLEAN / "step3_train.csv", parse_dates=["date"]); v = pd.read_csv(CLEAN / "step3_validation.csv", parse_dates=["date"])
d["m"] = d.date.dt.month; v["m"] = v.date.dt.month; d["rpm"] = d.posted_rate / d.distance

fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
examples = [(1, "January: clean", "quote_signal equals the rate per mile", BLUE), (4, "April: mirrored", "quote_signal equals 4.15 minus the rate per mile", ORANGE), (8, "August: random", "no relation to the rate per mile", INK2)]
for i, (m, title, sub, col) in enumerate(examples):
    ax = fig.add_subplot(gs[0, i]); s = d[d.m == m]
    ax.scatter(s.rpm, s.quote_signal, s=4, alpha=0.35, color=col, lw=0)
    ax.set(xlabel="Actual rate per mile ($)", ylabel="quote_signal" if i == 0 else "", title=title, xlim=(1.2, 3.6), ylim=(0.6, 3.6))
    ax.text(0.03, 0.04, f"{sub}\ncorrelation {np.corrcoef(s.rpm, s.quote_signal)[0,1]:+.2f}", transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")

ax = fig.add_subplot(gs[1, :])
months = list(range(1, 13)); corr = []; cols = []
for m in months:
    s = d[d.m == m] if m <= 10 else v[v.m == m]
    c = np.corrcoef(s.quote_signal, np.log(s.distance))[0, 1]; corr.append(c)
    cols.append(BLUE if c < -0.5 else ORANGE if c > 0.5 else INK2)
bars = ax.bar([pd.Timestamp(2025, m, 1).strftime("%b") for m in months], corr, color=cols, width=0.65)
for b, c, m in zip(bars, corr, months):
    ax.text(b.get_x() + b.get_width() / 2, c + (0.04 if c >= 0 else -0.04), f"{c:+.2f}", ha="center", va="bottom" if c >= 0 else "top", fontsize=9, color=INK2)
    if m >= 11: ax.text(b.get_x() + b.get_width() / 2, 0.42, "validation\n(no prices)", ha="center", fontsize=8, color=INK2)
ax.axhline(0, color=INK2, lw=1); ax.set_ylim(-1.05, 1.05); ax.grid(axis="x", visible=False)
ax.set(ylabel="Correlation of quote_signal with log distance", title="The regime can be read without prices: clean months near -0.8, mirrored near +0.8, random near 0. November and December are random.")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="clean: quote = rate per mile"), Patch(color=ORANGE, label="mirrored: quote = 4.15 - rate per mile"), Patch(color=INK2, label="random")], frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(h_pad=2.5); fig.savefig(OUT / "18_quote_signal_regimes.png", bbox_inches="tight"); print("wrote 18")
