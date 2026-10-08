"""Step 2 of cleaning: fill blank weight and blank market_index. Nothing is removed.

Run:  python src/step2_fill_blanks.py
Reads  cleaned-datasets/step1_train.csv, step1_validation.csv
Writes cleaned-datasets/step2_train.csv, step2_validation.csv

Rules (fill values come from TRAINING only, then applied to both files):
  weight blank        -> training median weight
  market_index blank  -> 1) mean of the other loads on the same day in the same file
                            (it is a daily market level; typical miss 0.016)
                         2) if the whole day is blank: mean of the same weekday one week
                            before and one week after (typical miss 0.020)
                         3) last resort: the training mean (typical miss 0.135)
Two flag columns record which rows were filled: weight_was_missing, market_index_was_missing.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "cleaned-datasets"

train = pd.read_csv(D / "step1_train.csv")
val = pd.read_csv(D / "step1_validation.csv")

WEIGHT_FILL = float(train.weight.median())
MI_FALLBACK = float(train.market_index.mean())


def fill(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    wm, mm = df.weight.isna(), df.market_index.isna()
    df["weight_was_missing"] = wm.astype(int)
    df["market_index_was_missing"] = mm.astype(int)
    df.loc[wm, "weight"] = WEIGHT_FILL
    day_mean = df.groupby("date").market_index.transform("mean")
    daily = df.groupby("date").market_index.mean()  # NaN for a fully blank day
    dates = pd.to_datetime(df.date)
    with warnings.catch_warnings():  # first/last week have no ±7-day neighbour; NaN is intended
        warnings.simplefilter("ignore", RuntimeWarning)
        week_mean = pd.Series(
            np.nanmean(np.c_[daily.reindex(dates - pd.Timedelta(days=7)).values,
                             daily.reindex(dates + pd.Timedelta(days=7)).values], axis=1),
            index=df.index)
    df["market_index"] = df.market_index.fillna(day_mean).fillna(week_mean).fillna(MI_FALLBACK)
    n_day = int((mm & day_mean.notna()).sum())
    n_week = int((mm & day_mean.isna() & week_mean.notna()).sum())
    print(f"{name:11s} weight filled {int(wm.sum()):>4} with {WEIGHT_FILL:,.0f} lb | "
          f"market_index filled {int(mm.sum()):>4}: same-day mean {n_day}, "
          f"same-weekday-adjacent-weeks {n_week}, training mean {int(mm.sum()) - n_day - n_week} | "
          f"blanks left {int(df.isna().sum().sum())}")
    return df


fill(train, "train").to_csv(D / "step2_train.csv", index=False)
fill(val, "validation").to_csv(D / "step2_validation.csv", index=False)
