"""Train the final model with and without quote_signal and test one month at a time. Run: python eda/quote_signal_with_without.py"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4, step5_model as m5
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#9a9891", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.dpi": 150})

d = pd.read_csv(f4.CLEAN / "step3_train.csv", parse_dates=["date"]); v = pd.read_csv(f4.CLEAN / "step3_validation.csv", parse_dates=["date"]); d["m"] = d.date.dt.month
ct = f4.city_coordinates([d, v]); mk = f4.daily_market_index([d, v])
err = lambda p, a: np.mean(np.abs(p - a) / a) * 100
BASE_LIN, BASE_GBM = list(m5.LIN_COLS), list(f4.FEATURE_COLUMNS)
tests = [("Jul", range(1, 7), 7, "mirror"), ("Aug", range(1, 8), 8, "random"), ("Sep", range(1, 9), 9, "copy"), ("Oct", range(1, 10), 10, "mirror")]
rows = []
for name, trm, tem, regime in tests:
    tr, te = d[d.m.isin(trm)], d[d.m == tem]
    fb = f4.FeatureBuilder(ct, mk).fit(tr); Xtr, _ = fb.transform(tr); Xte, _ = fb.transform(te)
    lt, le = tr.pickup + ">" + tr.delivery, te.pickup + ">" + te.delivery; y = np.log(tr.posted_rate.values); a = te.posted_rate.values
    m5.LIN_COLS[:] = BASE_LIN; m5.GBM_COLS = list(BASE_GBM); without = err(np.exp(m5.FinalModel().fit(Xtr, y, lt).predict(Xte, le)), a)
    Xtr["quote_signal"], Xte["quote_signal"] = tr.quote_signal.values, te.quote_signal.values
    m5.LIN_COLS[:] = BASE_LIN + ["quote_signal"]; m5.GBM_COLS = BASE_GBM + ["quote_signal"]; with_ = err(np.exp(m5.FinalModel().fit(Xtr, y, lt).predict(Xte, le)), a)
    rows.append((name, regime, without, with_)); print(f"  test {name} ({regime:6s}): without {without:.2f}%  with {with_:.2f}%")
m5.LIN_COLS[:] = BASE_LIN

fig, ax = plt.subplots(figsize=(10, 4.8)); x = np.arange(len(rows)); w = 0.36
b1 = ax.bar(x - w / 2, [r[2] for r in rows], w, color=BLUE, label="trained without quote_signal")
b2 = ax.bar(x + w / 2, [r[3] for r in rows], w, color=ORANGE, label="trained with quote_signal")
for b, val in list(zip(b1, [r[2] for r in rows])) + list(zip(b2, [r[3] for r in rows])):
    ax.text(b.get_x() + b.get_width() / 2, val + 0.06, f"{val:.2f}%", ha="center", fontsize=9.5, color=INK)
ax.set(xticks=x, xticklabels=[f"test {r[0]}\n({r[1]} month)" for r in rows], ylabel="Error on that month (mean absolute %)", ylim=(0, max(r[3] for r in rows) * 1.3),
       title="Same model, trained with and without quote_signal. On the one random month, August, the error more than doubles.")
ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="upper right")
i = [r[0] for r in rows].index("Aug"); ax.annotate("November and December\nare random months like August", xy=(i + w / 2, rows[i][3]), xytext=(i + 0.9, rows[i][3] * 1.08),
    fontsize=9.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
fig.tight_layout(); fig.savefig(ROOT / "eda" / "figures" / "19_quote_signal_with_without.png", bbox_inches="tight"); print("wrote 19")
