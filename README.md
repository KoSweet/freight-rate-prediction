# Freight Rate Prediction — Spotter ML assessment

Predicts the posted rate for 12,000 truck loads in Nov–Dec 2025 from 48,000 labelled loads in Jan–Oct 2025,
and produces the fixed December chart for Lexington → Fort Wayne.

**Deliverables in this repo**
- `validation_predictions.csv` — `load_id,predicted_rate` for all 12,000 loads (template order)
- `outputs/december-chart-inputs.csv` — the 31 December rows with `predicted_rate` filled
- `scorer_results/candidate_december.png` — produced by Spotter's `assessment-data/score.py`
- `cleaned-datasets/CLEANING_LOG.md` and `cleaned-datasets/MODELING_NOTES.md` — every decision with the evidence

## Run

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
./run_all.sh            # raw files -> cleaned data -> features -> model comparison -> predictions -> score.py
python -m pytest -q     # 4 pipeline tests
```

Raw inputs live in `assessment-data/` (train-test.csv, validation.csv, the template, december-chart-inputs.csv, score.py).
`run_all.sh` takes about two minutes. `python src/step5_sweep.py` reproduces the broader model sweep (~2 min more).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/step1_flip_sign.py` | 292 / 145 negative weights → positive (magnitudes are real weights) |
| 2 | `src/step2_fill_blanks.py` | blank weight → training median; blank market_index → same-day mean (fallback: same weekday ±1 week) |
| 3 | `src/step3_remove_bad_prices.py` | drop 677 training rows (1.4%) priced 2–5× or 0.2–0.5× the expected rate; validation untouched |
| 4 | `src/step4_features.py` | 23 features: log distance, equipment, weight, weekday, short-term market signal, coordinates, symmetric city effects with a nearest-3-cities fallback for 8 unseen cities |
| 5 | `src/step5_model.py` | model choice on four forward-in-time folds; `src/step5_sweep.py` = wider sweep |
| 6 | `src/step6_predict.py` | fit on all Jan–Oct, +0.6% level anchor, write both files, run `score.py` |

Charts for every step are in `eda/figures/`; `notebooks/01_eda.ipynb` is the exploratory walkthrough.

## Validation approach

The task is a forecast: train on Jan–Oct, predict Nov–Dec. So the model is selected on **forward-in-time
folds** (train Jan–Aug → test Sep–Oct, Jan–Jun → Jul–Aug, Jan–Jul → Aug–Oct, Jan–Sep → Oct), with every
learned component (fill values, city effects, lane adjustments) refitted inside each fold. A stratified
random split was tried and rejected: it reported 5.4% error for a model with seasonal features that scored
9.7% out of time. Folds overlap (Sep and Oct appear in three test sets), so differences under ~0.05 points
are not meaningful; the chosen model wins on every fold.

## Model

Linear model on log(price) → gradient-boosted trees (400, 15 leaves, early stopping off) on its residuals →
shrunk per-lane residual mean (0 for unseen lanes). Forward-fold error 1.70% mean absolute percentage
(≈ $40 per load), bias −0.6%, vs 1.87% linear alone and 1.92% trees alone.
Swapping the booster for CatBoost in the same structure gives 1.71%; CatBoost alone 1.90%. Random forest,
extra-trees, k-NN, a neural net and polynomial ridge were also tried (`src/step5_sweep.py`); none beat it.

Expected Nov–Dec error on genuine prices: **≈1.5–1.9% if the price level stays at the Sep–Oct level**
(every input says it does), **3–6% if the level shifts**. Spotter's measured error will be higher by the
share of bad prices in the answer key (~1.4% of rows at 2–5× off, undetectable without the price).

## Key findings

- Price is a near-perfect power law in distance (log–log correlation 0.97); rate per mile falls from $2.80 under
  200 mi to $1.87 over 2,500 mi. Reefer +12%, Flatbed +8% vs Dry Van. Location is worth ~1 point of error.
- **quote_signal is planted.** In Jan, Feb, Mar, Jun, Sep it *is* the rate per mile (0.8% MAPE on its own);
  in Apr, May, Jul, Oct it is 4.15 − rate per mile; in Aug, Nov, Dec it is random. Its correlation with
  log distance identifies the regime without prices (−0.80 / +0.81 / ≈0): **Nov 0.009, Dec 0.013 → random**,
  so it is excluded from the final model.
- Raw market_index is harmful out of time (its effect decays from +18% to +8% per unit over the year and
  it takes credit for a slow price drift); only its short-term deviation from a 28-day level is used.
  No month / day-of-year / trend features: they extrapolate badly (+4 to +8% bias).
- Remaining error ≈ 1% irreducible row noise (measured from the clean-quote months) plus a day-level drift
  that no input forecasts.

## Assumptions and caveats

- Nov–Dec follow the same rules as Jan–Oct. All inputs match Sep–Oct; no holiday effect exists in training.
- The market signal uses a centred 28-day window, so the model is batch-scored (validation ships all dates).
- City effects and the lane adjustment are fitted in-sample on the training rows (≈1,500 rows per city end;
  out-of-fold fitting of the lane term was checked and gains nothing).
- Step 2's weight median and step 3's expected prices are computed once on all Jan–Oct rows (negligible leak
  into the folds; the city/lane terms, which matter, are refitted per fold).
- Log target ≈ conditional median: optimal for MAE/MAPE, slightly low for RMSE. Spotter's metric is unknown.
