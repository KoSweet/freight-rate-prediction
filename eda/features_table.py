"""Feature table as an image, for the README and the walkthrough. Run: python eda/features_table.py"""
from pathlib import Path
import textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
SECTIONS = [
  ("Reshaped from the raw columns", BLUE, [
    ("distance, log distance, log distance squared", "distance", "Price is a straight line in log distance. The square lets the line bend for short hauls."),
    ("equipment, 3 yes/no columns", "equipment", "Reefer costs about 12% more per mile than Dry Van, Flatbed about 8% more."),
    ("weight", "weight", "Small effect, about 0.3% per extra 1,000 lb."),
    ("weekday, 7 yes/no columns", "date", "Midweek is slightly dearer than weekends."),
    ("pickup and delivery latitude, longitude", "coordinates, one row per city", "Lets the model place a city it has never seen next to the ones it knows."),
  ]),
  ("Built for this model", AQUA, [
    ("short-term market signal", "market_index minus its 28-day average", "The raw index hurt out of time. The short-term part is stable: +9 to +13% price per unit in every period."),
    ("pickup city effect, delivery city effect", "learned from training rows", "Location is worth about one point of error. A city costs the same as origin or destination, so one effect per city."),
    ("nearest-3-cities fallback", "city effects plus coordinates", "The 8 cities that appear only in validation get the average effect of their 3 nearest known cities."),
    ("two was-filled flags", "the step 2 fills", "Marks a filled weight or market_index so the model can discount it. In practice the effect is zero."),
  ]),
  ("Left out on purpose", ORANGE, [
    ("month, day of year, any trend", "", "Extrapolate badly into unseen months: +4 to +8% bias on forward folds."),
    ("raw market_index", "", "Its effect decays from 18% to 8% per unit over the year and it under-priced the autumn by 4%."),
    ("quote_signal", "", "A copy of the answer in seven training months, random in Aug, Nov and Dec. Doubles the error when used on a random month."),
    ("lane names as categories", "", "736 validation lanes never appear in training. Coordinates and city effects cover them instead."),
  ]),
]
COLS = [("Feature", 0.02, 0.27), ("Built from", 0.30, 0.22), ("Why", 0.53, 0.45)]
fig = plt.figure(figsize=(13, 9.4), facecolor=SURFACE); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
y = 0.965
ax.text(0.02, y, "The 23 features the model sees", fontsize=17, fontweight="bold", color=INK, va="top"); y -= 0.035
ax.text(0.02, y, "Everything learned (city effects, fallbacks, fill values) comes from training rows only and is refitted inside every validation fold.", fontsize=10.5, color=INK2, va="top"); y -= 0.045
for name, x, w in COLS: ax.text(x, y, name, fontsize=11, fontweight="bold", color=INK2, va="top")
y -= 0.012; ax.plot([0.02, 0.98], [y, y], color=INK2, lw=1); y -= 0.012
for title, color, rows in SECTIONS:
    ax.add_patch(plt.Rectangle((0.02, y - 0.028), 0.96, 0.028, color=color, alpha=0.12, lw=0)); ax.plot([0.02, 0.02], [y - 0.028, y], color=color, lw=4)
    ax.text(0.03, y - 0.014, title, fontsize=12, fontweight="bold", color=INK, va="center"); y -= 0.04
    for feat, src, why in rows:
        lines = [textwrap.wrap(feat, 34), textwrap.wrap(src, 30) if src else [""], textwrap.wrap(why, 68)]
        h = max(len(l) for l in lines) * 0.0185 + 0.012
        for (name, x, w), txt in zip(COLS, lines):
            ax.text(x, y, "\n".join(txt), fontsize=10.5, color=INK if name == "Feature" else INK2, va="top", fontweight="bold" if name == "Feature" else "normal", linespacing=1.3)
        y -= h; ax.plot([0.02, 0.98], [y + 0.004, y + 0.004], color=LINE, lw=0.8)
    y -= 0.012
fig.savefig(ROOT / "eda" / "figures" / "20_features_table.png", dpi=150, bbox_inches="tight", facecolor=SURFACE); print("wrote 20")
