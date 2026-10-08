"""Small safety net for the feature pipeline. Run: python -m pytest -q"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4

train = pd.read_csv(f4.CLEAN / "step3_train.csv").head(6000)
val = pd.read_csv(f4.CLEAN / "step3_validation.csv")
city_table = f4.city_coordinates([train, val]); market = f4.daily_market_index([train, val])


def test_transform_has_no_blanks_and_expected_columns():
    fb = f4.FeatureBuilder(city_table, market).fit(train)
    X, _ = fb.transform(train)
    assert list(X.columns) == f4.FEATURE_COLUMNS and X.isna().sum().sum() == 0


def test_unseen_city_gets_mean_of_three_nearest():
    fb = f4.FeatureBuilder(city_table, market).fit(train)
    unseen = sorted((set(val.pickup) | set(val.delivery)) - set(fb.city_effects.index))
    assert unseen, "fixture should contain validation-only cities"
    rows = val[val.pickup.isin(unseen)].head(50)
    X, log = fb.transform(rows)
    for city, (nearest, dist, value) in log.items():
        assert len(nearest) == 3 and all(n in fb.city_effects.index for n in nearest)
        assert np.isclose(value, fb.city_effects[nearest].mean())
    assert X.pickup_city_effect.notna().all()


def test_fit_is_deterministic():
    a = f4.FeatureBuilder(city_table, market).fit(train); b = f4.FeatureBuilder(city_table, market).fit(train)
    assert np.allclose(a.city_effects.values, b.city_effects.values)
    Xa, _ = a.transform(val.head(200)); Xb, _ = b.transform(val.head(200))
    assert np.allclose(Xa.values, Xb.values)


def test_december_rows_transform_without_coordinates_or_market_index():
    fb = f4.FeatureBuilder(city_table, market).fit(train)
    dec = pd.read_csv(f4.RAW / "december_chart_inputs.csv").drop(columns=["predicted_rate"])
    X, _ = fb.transform(dec)
    assert len(X) == 31 and X.isna().sum().sum() == 0 and X.mi_short.abs().max() < 0.5
