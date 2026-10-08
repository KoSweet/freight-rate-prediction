"""Step 4: feature engineering.

Run:  python src/step4_features.py
Reads  cleaned-datasets/step3_train.csv, step3_validation.csv, assessment-data/december-chart-inputs.csv
Writes cleaned-datasets/step4_train_features.csv
       cleaned-datasets/step4_validation_features.csv
       cleaned-datasets/step4_december_features.csv
       cleaned-datasets/artifacts/{city_table.csv, city_effects.csv, daily_market_index.csv, feature_columns.txt}

Everything that is *learned* (city effects, residual model, fill values) is learned from the training
rows passed to FeatureBuilder.fit(). Coordinates and the daily market_index series are inputs, not
answers, so they may be assembled from every file that has them (train + validation).

Features (see MODELING_NOTES.md for the evidence):
  distance, log_distance, log_distance_sq
  eq_dry_van, eq_reefer, eq_flatbed
  weight, weight_was_missing
  dow_0..dow_6
  mi_short (market_index - centred 28-day mean of the daily index), market_index_was_missing
  pickup_lat, pickup_lon, delivery_lat, delivery_lon
  pickup_city_effect, delivery_city_effect   (symmetric per-city effect; unseen city -> mean of 3 nearest)
Excluded on evidence: raw market_index, month/day-of-year/trend, lane id, quote_signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "cleaned-datasets"
RAW = ROOT / "assessment-data"
ART = CLEAN / "artifacts"
MI_WINDOW_DAYS = 28
N_NEAREST = 3

FEATURE_COLUMNS = (
    ["distance", "log_distance", "log_distance_sq",
     "eq_dry_van", "eq_reefer", "eq_flatbed",
     "weight", "weight_was_missing"]
    + [f"dow_{k}" for k in range(7)]
    + ["mi_short", "market_index_was_missing",
       "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
       "pickup_city_effect", "delivery_city_effect"]
)


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def city_coordinates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One (lat, lon) per city, taken from every frame that carries coordinates."""
    parts = []
    for f in frames:
        if {"pickup_lat", "pickup_lon"} <= set(f.columns):
            parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
            parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].agg(lambda s: s.mode().iloc[0])


def daily_market_index(frames: list[pd.DataFrame]) -> pd.Series:
    """Daily mean market_index over a contiguous calendar, plus its centred rolling mean."""
    rows = pd.concat([f[["date", "market_index"]] for f in frames if "market_index" in f.columns])
    rows["date"] = pd.to_datetime(rows.date)
    daily = rows.groupby("date").market_index.mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D")).interpolate()
    level = daily.rolling(MI_WINDOW_DAYS, center=True, min_periods=MI_WINDOW_DAYS // 3).mean()
    return pd.DataFrame({"daily_mean": daily, "level": level})


class FeatureBuilder:
    """fit() on training rows, transform() any frame with the raw columns."""

    def __init__(self, city_table: pd.DataFrame, market: pd.DataFrame):
        self.city_table = city_table
        self.market = market
        self.city_effects: pd.Series | None = None
        self.weight_fill = None

    # ------------------------------------------------------------------ base columns
    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        date = pd.to_datetime(df.date)
        out["distance"] = df.distance.astype(float)
        out["log_distance"] = np.log(out.distance)
        out["log_distance_sq"] = out.log_distance ** 2
        for name, label in [("eq_dry_van", "Dry Van"), ("eq_reefer", "Reefer"), ("eq_flatbed", "Flatbed")]:
            out[name] = (df.equipment == label).astype(int)
        w = df.weight.astype(float)
        out["weight_was_missing"] = df["weight_was_missing"].astype(int) if "weight_was_missing" in df else w.isna().astype(int)
        out["weight"] = w.fillna(self.weight_fill)
        for k in range(7):
            out[f"dow_{k}"] = (date.dt.dayofweek == k).astype(int)
        # market: use the row's own market_index if present, else that day's mean (December rows)
        if "market_index" in df and df.market_index.notna().any():
            mi = df.market_index.astype(float)
            was_missing = df["market_index_was_missing"].astype(int) if "market_index_was_missing" in df else mi.isna().astype(int)
            mi = mi.fillna(pd.Series(self.market.daily_mean.reindex(date).values, index=df.index))
        else:
            mi = pd.Series(self.market.daily_mean.reindex(date).values, index=df.index)
            was_missing = pd.Series(1, index=df.index)
        out["mi_short"] = mi.values - self.market.level.reindex(date).values
        out["market_index_was_missing"] = was_missing.values
        for role in ("pickup", "delivery"):
            out[f"{role}_lat"] = df[role].map(self.city_table.lat).values
            out[f"{role}_lon"] = df[role].map(self.city_table.lon).values
        return out

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self.weight_fill = float(train.weight.median())
        base = self._base(train)
        y = np.log(train.posted_rate.values)
        # 1) residualise log price on everything except location
        X = self._design_without_cities(base)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        # 2) symmetric city effects: one coefficient per city, applied to both ends
        cities = sorted(set(train.pickup) | set(train.delivery))
        idx = {c: i for i, c in enumerate(cities)}
        C = np.zeros((len(train), len(cities)))
        C[np.arange(len(train)), train.pickup.map(idx).values] += 1
        C[np.arange(len(train)), train.delivery.map(idx).values] += 1
        eff, *_ = np.linalg.lstsq(C, resid, rcond=None)
        eff = eff - eff.mean()  # centre: the average city is 0
        self.city_effects = pd.Series(eff, index=cities, name="effect")
        return self

    @staticmethod
    def _design_without_cities(base: pd.DataFrame) -> np.ndarray:
        cols = ["log_distance", "log_distance_sq", "weight", "eq_reefer", "eq_flatbed", "mi_short"] + [f"dow_{k}" for k in range(6)]
        return np.column_stack([np.ones(len(base))] + [base[c].values for c in cols])

    # ------------------------------------------------------------------ city effect lookup with fallback
    def effect_for(self, cities: pd.Series) -> tuple[np.ndarray, dict]:
        known = self.city_effects
        out = cities.map(known)
        fallback_log = {}
        for c in cities[out.isna()].unique():
            if c not in self.city_table.index:
                raise KeyError(f"no coordinates for city {c!r}")
            lat, lon = self.city_table.loc[c, ["lat", "lon"]]
            kc = self.city_table.loc[known.index]
            dist = haversine_miles(lat, lon, kc.lat.values, kc.lon.values)
            nearest = known.index[np.argsort(dist)[:N_NEAREST]]
            val = float(known[nearest].mean())
            out[cities == c] = val
            fallback_log[c] = (list(nearest), [round(float(x), 0) for x in np.sort(dist)[:N_NEAREST]], val)
        return out.astype(float).values, fallback_log

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if self.city_effects is None:
            raise RuntimeError("call fit() first")
        out = self._base(df)
        out["pickup_city_effect"], log_p = self.effect_for(df.pickup)
        out["delivery_city_effect"], log_d = self.effect_for(df.delivery)
        out = out[FEATURE_COLUMNS]
        assert out.isna().sum().sum() == 0, out.isna().sum()[out.isna().sum() > 0]
        return out, {**log_p, **log_d}


def load_december() -> pd.DataFrame:
    dec = pd.read_csv(RAW / "december-chart-inputs.csv")
    return dec.drop(columns=["predicted_rate"])


def main() -> None:
    train = pd.read_csv(CLEAN / "step3_train.csv")
    val = pd.read_csv(CLEAN / "step3_validation.csv")
    dec = load_december()

    city_table = city_coordinates([train, val])
    market = daily_market_index([train, val])
    fb = FeatureBuilder(city_table, market).fit(train)

    Xtr, _ = fb.transform(train)
    Xva, fallback = fb.transform(val)
    Xde, _ = fb.transform(dec)

    Xtr.assign(load_id=train.load_id, posted_rate=train.posted_rate, date=train.date).to_csv(CLEAN / "step4_train_features.csv", index=False)
    Xva.assign(load_id=val.load_id, date=val.date).to_csv(CLEAN / "step4_validation_features.csv", index=False)
    Xde.assign(date=dec.date).to_csv(CLEAN / "step4_december_features.csv", index=False)

    ART.mkdir(exist_ok=True)
    city_table.to_csv(ART / "city_table.csv")
    fb.city_effects.to_csv(ART / "city_effects.csv")
    market.to_csv(ART / "daily_market_index.csv")
    (ART / "feature_columns.txt").write_text("\n".join(FEATURE_COLUMNS) + "\n")

    print(f"features: {len(FEATURE_COLUMNS)} columns")
    print(f"train {Xtr.shape}  validation {Xva.shape}  december {Xde.shape}  blanks: {int(Xtr.isna().sum().sum() + Xva.isna().sum().sum() + Xde.isna().sum().sum())}")
    print(f"city effects learned for {len(fb.city_effects)} cities, range {100*fb.city_effects.min():+.1f}% to {100*fb.city_effects.max():+.1f}%")
    print("unseen cities -> nearest-3 fallback:")
    for c, (near, dist, val_) in fallback.items():
        print(f"  {c:10s} {100*val_:+.2f}%   from {near}  at {[int(x) for x in dist]} mi")
    print(f"mi_short: train mean {Xtr.mi_short.mean():+.4f} std {Xtr.mi_short.std():.4f} | validation mean {Xva.mi_short.mean():+.4f} std {Xva.mi_short.std():.4f} | december range {Xde.mi_short.min():+.3f}..{Xde.mi_short.max():+.3f}")


if __name__ == "__main__":
    main()
