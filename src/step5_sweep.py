"""Step 5 addendum: broader model sweep (reproduces the table in MODELING_NOTES.md).

Run:  python src/step5_sweep.py          (~2 minutes)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4
import step5_model as m5

C = list(f4.FEATURE_COLUMNS)


class ResidualModel:
    """Linear model first, then any sklearn regressor on its residuals."""
    def __init__(self, reg, cols=C, scale=False): self.reg, self.cols, self.scale = reg, cols, scale
    def fit(self, X, y, lanes=None):
        self.lin = m5.Linear().fit(X, y); r = y - self.lin.predict(X); Z = X[self.cols].values
        if self.scale: self.sc = StandardScaler().fit(Z); Z = self.sc.transform(Z)
        self.reg.fit(Z, r); return self
    def predict(self, X, lanes=None):
        Z = X[self.cols].values
        if self.scale: Z = self.sc.transform(Z)
        return self.lin.predict(X) + self.reg.predict(Z)


SWEEP = {
    "hybrid: linear + GBM": lambda: m5.LinearPlusGBM(max_iter=400, max_leaf_nodes=15),
    "linear + random forest on residuals": lambda: ResidualModel(RandomForestRegressor(300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0)),
    "linear + extra-trees on residuals": lambda: ResidualModel(ExtraTreesRegressor(300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0)),
    "linear + k-NN on residuals (location+equipment)": lambda: ResidualModel(KNeighborsRegressor(50, weights="distance"), cols=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "eq_reefer", "eq_flatbed", "log_distance"], scale=True),
    "linear + neural net on residuals (2x64)": lambda: ResidualModel(MLPRegressor((64, 64), alpha=1e-3, max_iter=200, early_stopping=True, random_state=0), scale=True),
    "ridge with pairwise interactions": lambda: ResidualModel(make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(10.0)), cols=["log_distance", "log_distance_sq", "weight", "eq_reefer", "eq_flatbed", "mi_short", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "pickup_city_effect", "delivery_city_effect"], scale=True),
    "GBM alone, 2000 trees, 63 leaves": lambda: m5.GBM(learning_rate=0.02, max_iter=2000, max_leaf_nodes=63),
    "hybrid + lane adjustment k=5": lambda: m5.FinalModel(k=5),
    "hybrid + lane adjustment k=20 (FINAL)": lambda: m5.FinalModel(k=20),
    "hybrid + lane adjustment k=60": lambda: m5.FinalModel(k=60),
}


def main() -> None:
    train = pd.read_csv(f4.CLEAN / "step3_train.csv", parse_dates=["date"]); val = pd.read_csv(f4.CLEAN / "step3_validation.csv", parse_dates=["date"])
    train["m"] = train.date.dt.month
    city_table = f4.city_coordinates([train, val]); market = f4.daily_market_index([train, val])
    folds = []
    for name, trm, tem in m5.FOLDS:
        tr, te = train[train.m.isin(trm)], train[train.m.isin(tem)]
        fb = f4.FeatureBuilder(city_table, market).fit(tr); Xtr, _ = fb.transform(tr); Xte, _ = fb.transform(te)
        folds.append((name, Xtr, np.log(tr.posted_rate.values), Xte, te.posted_rate.values, tr.pickup + ">" + tr.delivery, te.pickup + ">" + te.delivery))
    rows = []
    for cname, make in SWEEP.items():
        t0 = time.time(); errs = []
        for fname, Xtr, ytr, Xte, a, ltr, lte in folds:
            p = np.exp(make().fit(Xtr, ytr, ltr).predict(Xte, lte)); errs.append(np.mean(np.abs(p - a) / a) * 100)
        rows.append(dict(model=cname, avg_mape=np.mean(errs), worst_mape=max(errs), **{f: e for (f, *_), e in zip(folds, errs)}))
        print(f"{cname:50s} avg {np.mean(errs):.3f}%  worst {max(errs):.2f}%  ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).sort_values("avg_mape").to_csv(f4.CLEAN / "step5_sweep_results.csv", index=False)


if __name__ == "__main__":
    main()
