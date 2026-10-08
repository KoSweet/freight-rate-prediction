"""Step 3 of cleaning: remove training rows whose price is far from any plausible value.
Validation is passed through untouched (it has no price, and all 12,000 rows must be predicted).

Run:  python src/step3_remove_bad_prices.py
Reads  cleaned-datasets/step2_train.csv, step2_validation.csv
Writes cleaned-datasets/step3_train.csv           (kept rows)
       cleaned-datasets/step3_train_removed.csv   (removed rows, with expected price and ratio)
       cleaned-datasets/step3_validation.csv      (identical to step2_validation.csv)

Rule: expected rate per mile = median of the central cloud (0.8-5 $/mi) for the same equipment and
distance bucket. A row is removed if posted_rate / expected is above 2 or below 0.5. The two bands of
bad prices sit at 0.17-0.47x and 2.2-5.4x, with an empty gap either side of the cut, so no genuine
load is clipped.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "cleaned-datasets"
DIST_BINS = [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 4000]
HIGH, LOW = 2.0, 0.5

train = pd.read_csv(D / "step2_train.csv")
val = pd.read_csv(D / "step2_validation.csv")

t = train.assign(rpm=train.posted_rate / train.distance, dbin=pd.cut(train.distance, DIST_BINS))
core = t[(t.rpm > 0.8) & (t.rpm < 5.0)]
expected_rpm = core.groupby(["equipment", "dbin"], observed=True).rpm.median().rename("expected_rpm")
t = t.join(expected_rpm, on=["equipment", "dbin"])
t["expected_rate"] = (t.expected_rpm * t.distance).round(2)
t["rate_ratio"] = (t.posted_rate / t.expected_rate).round(3)
bad = (t.rate_ratio > HIGH) | (t.rate_ratio < LOW)

kept = train[~bad]
removed = t[bad].drop(columns=["rpm", "dbin", "expected_rpm"])
kept.to_csv(D / "step3_train.csv", index=False)
removed.to_csv(D / "step3_train_removed.csv", index=False)
val.to_csv(D / "step3_validation.csv", index=False)

print(f"train      rows in {len(train):,}  removed {int(bad.sum())} ({bad.mean():.1%})  "
      f"too high {int((t.rate_ratio > HIGH).sum())}  too low {int((t.rate_ratio < LOW).sum())}  kept {len(kept):,}")
print(f"validation rows in {len(val):,}  removed 0  kept {len(val):,}")
print(f"nearest kept row to the cut: ratio {t.rate_ratio[~bad].max():.3f} (cut 2.0) and {t.rate_ratio[~bad].min():.3f} (cut 0.5)")
print(f"nearest removed row to the cut: ratio {t.rate_ratio[bad & (t.rate_ratio > 1)].min():.3f} and {t.rate_ratio[bad & (t.rate_ratio < 1)].max():.3f}")
