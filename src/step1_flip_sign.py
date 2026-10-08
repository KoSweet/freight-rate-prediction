"""Step 1 of cleaning: flip negative weights to positive. Nothing else is changed.

Run:  python src/step1_flip_sign.py
Writes cleaned-datasets/step1_train.csv and cleaned-datasets/step1_validation.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "assessment-data", ROOT / "cleaned-datasets"
OUT.mkdir(exist_ok=True)

for src, dst in [("train-test.csv", "step1_train.csv"), ("validation.csv", "step1_validation.csv")]:
    df = pd.read_csv(RAW / src)
    neg = df.weight < 0
    df.loc[neg, "weight"] = df.loc[neg, "weight"].abs()
    df.to_csv(OUT / dst, index=False)
    print(f"{src:16s} rows {len(df):>6,}  negative weights flipped {int(neg.sum()):>4}  "
          f"negative left {int((df.weight < 0).sum())}  blanks left (untouched) {int(df.weight.isna().sum())}")
