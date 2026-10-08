"""Step 6: train the final model on all of Jan-Oct and produce the deliverables.

Run:  python src/step6_predict.py
Reads  cleaned-datasets/step3_train.csv, step3_validation.csv
       data/validation_predictions_template.csv, december_chart_inputs.csv
Writes validation_predictions.csv                  (load_id, predicted_rate; template order)
       data/december_chart_inputs.csv           (the 7 original columns, predicted_rate filled)
       outputs/final_model_summary.txt
       scorer_results/candidate_december.png       (via score.py)

Level anchor: the model has no time features, so its level is the Jan-Oct average. Nov-Dec inputs match
Sep-Oct (MODELING_NOTES.md), so predictions are shifted by the model's mean log residual on Sep-Oct rows.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import step4_features as f4
from step5_model import FinalModel

OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
ANCHOR_MONTHS = (9, 10)


def main() -> None:
    train = pd.read_csv(f4.CLEAN / "step3_train.csv", parse_dates=["date"])
    val = pd.read_csv(f4.CLEAN / "step3_validation.csv", parse_dates=["date"])
    template = pd.read_csv(f4.RAW / "validation_predictions_template.csv")
    dec_raw = pd.read_csv(f4.RAW / "december_chart_inputs.csv")
    dec = dec_raw.drop(columns=["predicted_rate"])

    # features: coordinates and the daily market series are inputs and may use both files; everything learned uses train only
    fb = f4.FeatureBuilder(f4.city_coordinates([train, val]), f4.daily_market_index([train, val])).fit(train)
    Xtr, _ = fb.transform(train); Xva, fallback = fb.transform(val); Xde, _ = fb.transform(dec)
    lane = lambda df: (df.pickup + ">" + df.delivery)

    model = FinalModel().fit(Xtr, np.log(train.posted_rate.values), lane(train))

    # level anchor from the model's own residuals on the last two training months
    resid = np.log(train.posted_rate.values) - model.predict(Xtr, lane(train))
    anchor = float(resid[train.date.dt.month.isin(ANCHOR_MONTHS)].mean())

    pred_val = np.exp(model.predict(Xva, lane(val)) + anchor)
    pred_dec = np.exp(model.predict(Xde, lane(dec)) + anchor)

    # validation_predictions.csv in template order
    sub = template[["load_id"]].merge(pd.DataFrame({"load_id": val.load_id, "predicted_rate": np.round(pred_val, 2)}), on="load_id", how="left")
    assert len(sub) == 12_000 and sub.predicted_rate.notna().all() and (sub.predicted_rate > 0).all()
    sub.to_csv(ROOT / "validation_predictions.csv", index=False)

    # December file: original seven columns, same order, predicted_rate filled
    dec_out = dec_raw.copy(); dec_out["predicted_rate"] = np.round(pred_dec, 2)
    dec_out.to_csv(f4.RAW / "december_chart_inputs.csv", index=False)  # filled in place, as the assessment README asks

    # summary
    lane_dv = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne") & (train.equipment == "Dry Van")].posted_rate
    rpm_tr, rpm_va = train.posted_rate / train.distance, pred_val / val.distance
    lines = [
        "Final model: linear (log price) + GBM(400 trees, 15 leaves, early stopping off) on residuals + per-lane residual mean (k=20)",
        f"trained on {len(train):,} rows (Jan-Oct 2025, bad prices removed); 23 features; {len(fb.city_effects)} city effects",
        f"level anchor (mean log residual on Sep-Oct training rows): {100*anchor:+.2f}%",
        f"validation: {len(sub):,} predictions, ${sub.predicted_rate.min():,.0f} to ${sub.predicted_rate.max():,.0f}, median ${sub.predicted_rate.median():,.0f}",
        f"rate per mile: training median {rpm_tr.median():.3f}, predicted validation median {rpm_va.median():.3f}",
        f"rows touching an unseen city: {(val.pickup.isin(fallback) | val.delivery.isin(fallback)).sum():,} (fallback effects: " + ", ".join(f"{c} {100*v[2]:+.1f}%" for c, v in fallback.items()) + ")",
        f"lanes with a learned lane adjustment: {lane(val).isin(model.lane_adj.index).mean():.0%} of validation rows",
        f"December chart: ${pred_dec.min():,.0f} to ${pred_dec.max():,.0f}, mean ${pred_dec.mean():,.0f}; training Lexington>Fort Wayne Dry Van median ${lane_dv.median():,.0f} (n={len(lane_dv)})",
    ]
    (OUT / "final_model_summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

    # Spotter's format checker + chart
    print("\n--- score.py ---")
    r = subprocess.run([sys.executable, "score.py", "--predictions", "validation_predictions.csv",
                        "--december-predictions", "data/december_chart_inputs.csv"], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip()) if r.stderr.strip() else None
    if r.returncode != 0: raise SystemExit("score.py failed")


if __name__ == "__main__":
    main()
