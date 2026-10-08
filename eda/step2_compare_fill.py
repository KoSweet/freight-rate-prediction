"""Compare two ways to fill a blank market_index: same-day average vs same-weekday week-before/after average."""
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

s1 = pd.read_csv(D / "step1_train.csv", parse_dates=["date"])
rng = np.random.default_rng(1)
k = s1.dropna(subset=["market_index"])
idx = rng.choice(k.index, 3000, replace=False)
h = s1.copy(); truth = h.loc[idx, "market_index"].copy(); h.loc[idx, "market_index"] = np.nan
daily = h.groupby("date").market_index.mean()
dates = h.loc[idx, "date"]
same_day = h.groupby("date").market_index.transform("mean").loc[idx]
monday = pd.Series(np.nanmean(np.c_[daily.reindex(dates - pd.Timedelta(days=7)).values,
                                    daily.reindex(dates + pd.Timedelta(days=7)).values], axis=1), index=idx)
e_day, e_mon = (same_day - truth).abs(), (monday - truth).abs()

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
axes = [None, fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]

def example(ax, target, label):
    win = (s1.date >= target - pd.Timedelta(days=10)) & (s1.date <= target + pd.Timedelta(days=10))
    w = s1[win].dropna(subset=["market_index"])
    dm = w.groupby("date").market_index.mean()
    ax.scatter(w.date, w.market_index, s=5, alpha=0.2, color=BLUE, lw=0, label="individual loads")
    ax.plot(dm.index, dm.values, color=INK, lw=1.2, label="daily average")
    tgt = dm[target]; pv, nv = dm[target - pd.Timedelta(days=7)], dm[target + pd.Timedelta(days=7)]
    mon = (pv + nv) / 2
    ax.scatter([target], [tgt], s=140, color=BLUE, zorder=5, edgecolor=SURFACE, lw=1.5, label=f"same-day average = {tgt:.3f}")
    ax.scatter([target - pd.Timedelta(days=7), target + pd.Timedelta(days=7)], [pv, nv], s=90, color=ORANGE, zorder=5, edgecolor=SURFACE, lw=1.5, label="last Monday & next Monday")
    ax.plot([target - pd.Timedelta(days=7), target + pd.Timedelta(days=7)], [pv, nv], color=ORANGE, lw=1.2, ls="--")
    ax.scatter([target], [mon], s=140, marker="D", color=ORANGE, zorder=6, edgecolor=SURFACE, lw=1.5, label=f"Monday average = {mon:.3f}")
    ax.annotate("", xy=(target, mon), xytext=(target, tgt), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(target + pd.Timedelta(hours=16), (tgt + mon) / 2, f"gap {abs(tgt - mon):.3f}", fontsize=9, color=INK2, va="center")
    ax.set(title=f"{label}: blank on Monday {target:%-d %B}", ylabel="market_index")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b")); ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))

example(ax_a, pd.Timestamp("2025-06-09"), "Calm week")
example(ax_b, pd.Timestamp("2025-07-28"), "Market shift")
ax_b.set_xlabel("Date (2025)")

# --- panel 2: distribution of misses
ax = axes[1]
bins = np.linspace(0, 0.12, 49)
ax.hist(e_day, bins=bins, color=BLUE, alpha=0.75, label=f"same-day average (typical {e_day.median():.3f})")
ax.hist(e_mon, bins=bins, color=ORANGE, alpha=0.6, label=f"Monday average (typical {e_mon.median():.3f})")
ax.axvline(e_day.median(), color=BLUE, lw=1.4, ls="--"); ax.axvline(e_mon.median(), color=ORANGE, lw=1.4, ls="--")
ax.set(title="Miss on 3,000 hidden values", xlabel="|filled value - true value|", ylabel="Count")
ax.legend(frameon=False, fontsize=8.5)

# --- panel 3: typical miss by weekday
ax = axes[2]
dow = dates.dt.dayofweek
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
md = [e_day[dow == i].median() for i in range(7)]; mm = [e_mon[dow == i].median() for i in range(7)]
x = np.arange(7); wdt = 0.38
ax.bar(x - wdt / 2, md, wdt, color=BLUE, label="same-day average")
ax.bar(x + wdt / 2, mm, wdt, color=ORANGE, label="same weekday, ±1 week")
ax.set(xticks=x, xticklabels=names, title="Typical miss by weekday", ylabel="Typical miss")
ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8.5)

fig.tight_layout(w_pad=2.5); fig.savefig(OUT / "12_step2_fill_method_comparison.png", bbox_inches="tight"); plt.close(fig)
print("wrote 12")
print(f"same-day better on {int((e_day < e_mon).sum())} of 3000 hidden values; Monday method better on {int((e_mon < e_day).sum())}")
