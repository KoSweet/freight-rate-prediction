"""Step 4 check: refit the FeatureBuilder inside each forward fold and confirm the error; tune the mi_short window."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4

train = pd.read_csv(f4.CLEAN / "step3_train.csv", parse_dates=["date"]); val = pd.read_csv(f4.CLEAN / "step3_validation.csv", parse_dates=["date"])
train["m"] = train.date.dt.month
city_table = f4.city_coordinates([train, val])
folds = [((1, 8), (9, 10)), ((1, 6), (7, 8)), ((1, 7), (8, 10))]

def linear_fit_predict(Xtr, ytr, Xte):
    A = np.column_stack([np.ones(len(Xtr)), Xtr.values]); B = np.column_stack([np.ones(len(Xte)), Xte.values])
    b, *_ = np.linalg.lstsq(A, np.log(ytr), rcond=None); return np.exp(B @ b)

def run(window, cols=None, label=""):
    f4.MI_WINDOW_DAYS = window
    market = f4.daily_market_index([train, val])
    out = []
    for trm, tem in folds:
        tr = train[train.m.between(*trm)]; te = train[train.m.between(*tem)]
        fb = f4.FeatureBuilder(city_table, market).fit(tr)
        Xtr, _ = fb.transform(tr); Xte, fb_log = fb.transform(te)
        if cols: Xtr, Xte = Xtr[cols], Xte[cols]
        p = linear_fit_predict(Xtr, tr.posted_rate.values, Xte); a = te.posted_rate.values
        out.append((np.mean(np.abs(p - a) / a) * 100, np.mean(p / a - 1) * 100, len(fb_log)))
    print(f"  {label:44s}" + " | ".join(f"err {e:.2f}% bias {b:+.2f}%" for e, b, _ in out) + f"   avg {np.mean([o[0] for o in out]):.2f}%   cities needing fallback per fold {[o[2] for o in out]}")

print("Forward folds, FeatureBuilder fitted INSIDE each fold, linear model on log price")
print(f"  {'':44s}{'Jan-Aug > Sep-Oct':>24s} | {'Jan-Jun > Jul-Aug':>22s} | {'Jan-Jul > Aug-Oct':>22s}")
drop_lin = ["distance", "eq_dry_van", "dow_6"]  # collinear with intercept / other columns in a linear model
full = [c for c in f4.FEATURE_COLUMNS if c not in drop_lin]
run(28, full, "all 23 features (window 28d)")
run(28, [c for c in full if not c.startswith(("pickup_city", "delivery_city"))], "without city effects (coords only)")
run(28, [c for c in full if not c.endswith(("_lat", "_lon"))], "without coordinates (city effects only)")
run(28, [c for c in full if c != "mi_short"], "without mi_short")
run(21, full, "window 21d")
run(35, full, "window 35d")
run(42, full, "window 42d")
