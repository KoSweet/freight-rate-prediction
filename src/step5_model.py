"""Step 5: choose the model on forward-in-time folds.

Run:  python src/step5_model.py
Candidates (all predict log(posted_rate) from the step 4 features; FeatureBuilder refitted inside each fold):
  linear      ordinary least squares on log price (baseline, ~1.93% on folds)
  gbm         HistGradientBoostingRegressor on log price
  linear+gbm  OLS first, then a GBM fitted to the OLS residuals (keeps linear extrapolation, adds interactions)
  FINAL       linear+gbm plus a shrunk per-lane residual mean (0 for unseen lanes) -- the chosen model
Folds: Jan-Aug>Sep-Oct, Jan-Jun>Jul-Aug, Jan-Jul>Aug-Oct, Jan-Sep>Oct.
Writes cleaned-datasets/step5_model_comparison.csv and prints a table.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4

FOLDS = [("Jan-Aug>Sep-Oct", range(1, 9), [9, 10]), ("Jan-Jun>Jul-Aug", range(1, 7), [7, 8]),
         ("Jan-Jul>Aug-Oct", range(1, 8), [8, 9, 10]), ("Jan-Sep>Oct", range(1, 10), [10])]
LIN_COLS = [c for c in f4.FEATURE_COLUMNS if c not in ("distance", "eq_dry_van", "dow_6")]  # drop collinear columns for OLS
GBM_COLS = list(f4.FEATURE_COLUMNS)


class Linear:
    def fit(self, X, y, lanes=None):
        A = np.column_stack([np.ones(len(X)), X[LIN_COLS].values]); self.b, *_ = np.linalg.lstsq(A, y, rcond=None); return self
    def predict(self, X, lanes=None):
        return np.column_stack([np.ones(len(X)), X[LIN_COLS].values]) @ self.b


class GBM:
    def __init__(self, **kw):
        self.kw = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0,
                       early_stopping=False)  # sklearn turns early stopping on automatically above 10k rows and would stop at ~98 trees
        self.kw.update(kw)
    def fit(self, X, y, lanes=None):
        self.m = HistGradientBoostingRegressor(**self.kw).fit(X[GBM_COLS].values, y); return self
    def predict(self, X, lanes=None):
        return self.m.predict(X[GBM_COLS].values)


class LinearPlusGBM:
    def __init__(self, **kw):
        self.lin, self.gbm = Linear(), GBM(**kw)
    def fit(self, X, y, lanes=None):
        self.lin.fit(X, y); self.gbm.fit(X, y - self.lin.predict(X)); return self
    def predict(self, X, lanes=None):
        return self.lin.predict(X) + self.gbm.predict(X)


class FinalModel:
    """Chosen in the step 5 sweep: linear + GBM on residuals, then a smoothed per-lane residual mean
    learned from the training rows (0 for a lane not seen in training). k = shrinkage toward 0."""

    def __init__(self, k: int = 20, **gbm_kw):
        self.k = k
        self.base = LinearPlusGBM(**{**dict(max_iter=400, max_leaf_nodes=15), **gbm_kw})

    def fit(self, X, y, lanes=None):
        self.base.fit(X, y)
        r = y - self.base.predict(X)
        g = pd.DataFrame({"lane": np.asarray(lanes), "r": r}).groupby("lane").r.agg(["sum", "count"])
        self.lane_adj = g["sum"] / (g["count"] + self.k)
        return self

    def predict(self, X, lanes=None):
        adj = pd.Series(np.asarray(lanes)).map(self.lane_adj).fillna(0.0).values if lanes is not None else 0.0
        return self.base.predict(X) + adj


CANDIDATES = {
    "linear": lambda: Linear(),
    "gbm (lr .05, 600 it, 31 leaves)": lambda: GBM(),
    "gbm (lr .03, 1200 it, 15 leaves)": lambda: GBM(learning_rate=0.03, max_iter=1200, max_leaf_nodes=15),
    "gbm, absolute-error loss": lambda: GBM(loss="absolute_error"),
    "linear+gbm (lr .05, 400 it, 15 leaves)": lambda: LinearPlusGBM(max_iter=400, max_leaf_nodes=15),
    "linear+gbm (lr .03, 800 it, 15 leaves)": lambda: LinearPlusGBM(learning_rate=0.03, max_iter=800, max_leaf_nodes=15),
    "linear+gbm (lr .05, 400 it, 31 leaves)": lambda: LinearPlusGBM(max_iter=400, max_leaf_nodes=31),
    "FINAL: linear+gbm + lane adjustment (k=20)": lambda: FinalModel(k=20),
}


def main() -> None:
    train = pd.read_csv(f4.CLEAN / "step3_train.csv", parse_dates=["date"]); val = pd.read_csv(f4.CLEAN / "step3_validation.csv", parse_dates=["date"])
    train["m"] = train.date.dt.month
    city_table = f4.city_coordinates([train, val]); market = f4.daily_market_index([train, val])

    # build fold feature tables once (builder refitted per fold)
    folds = []
    for name, trm, tem in FOLDS:
        tr, te = train[train.m.isin(trm)], train[train.m.isin(tem)]
        fb = f4.FeatureBuilder(city_table, market).fit(tr)
        Xtr, _ = fb.transform(tr); Xte, _ = fb.transform(te)
        folds.append((name, Xtr, np.log(tr.posted_rate.values), Xte, te.posted_rate.values, tr.pickup + ">" + tr.delivery, te.pickup + ">" + te.delivery))

    rows = []
    for cname, make in CANDIDATES.items():
        t0 = time.time(); per = []
        for fname, Xtr, ytr, Xte, a, ltr, lte in folds:
            p = np.exp(make().fit(Xtr, ytr, ltr).predict(Xte, lte))
            per.append(dict(model=cname, fold=fname, mape=np.mean(np.abs(p - a) / a) * 100, mae=np.mean(np.abs(p - a)), bias=np.mean(p / a - 1) * 100))
        rows += per
        avg = np.mean([r["mape"] for r in per]); print(f"{cname:42s} avg err {avg:.3f}%   " + "  ".join(f"{r['mape']:.2f}" for r in per) + f"   ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows); res.to_csv(f4.CLEAN / "step5_model_comparison.csv", index=False)
    summary = res.groupby("model").agg(avg_mape=("mape", "mean"), worst_mape=("mape", "max"), avg_mae=("mae", "mean"), avg_bias=("bias", "mean")).sort_values("avg_mape")
    print("\n", summary.round(3).to_string())


if __name__ == "__main__":
    main()

