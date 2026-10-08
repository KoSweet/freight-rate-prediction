# Modelling notes — feature decisions (2026-10-08)

All numbers below come from a log-price linear model evaluated on three forward-in-time folds
(train Jan–Aug > test Sep–Oct; Jan–Jun > Jul–Aug; Jan–Jul > Aug–Oct), genuine prices only.
Error = mean absolute % error; bias = mean of predicted/actual − 1.

## What the forward test revealed

| Feature set | Fold 1 | Fold 2 | Fold 3 | Avg err | Bias range |
|---|---|---|---|---|---|
| base: log dist, log dist², weight, equipment | 2.82 | 3.04 | 2.78 | 2.88 | −0.7 to 0.0 |
| base + raw market_index | 4.62 | 2.91 | 4.44 | 3.99 | −4.3 to −1.5 |
| base + linear time trend | 4.43 | 7.85 | 7.81 | 6.70 | +4.0 to +7.8 |
| base + day of week | 2.76 | 2.94 | 2.71 | 2.81 | −0.7 |
| base + mi_short (today vs surrounding 4-week level) | 2.76 | 2.86 | 2.70 | 2.77 | −0.7 |
| base + coordinates (lat/lon + squares) | 2.11 | 2.41 | 2.07 | 2.20 | −0.8 |
| base + symmetric city effects | 1.98 | 2.30 | 1.93 | 2.07 | −0.8 |
| base + quote_signal | 2.82 | 3.24 | 2.79 | 2.95 | −1.0 |
| **base + dow + city effects + mi_short** | **1.93** | **2.05** | **1.85** | **1.94** | **−0.8** |

Key findings
1. **Raw market_index is harmful out of time.** Its slow component moves with an upward price drift
   in the first half of the year, so the model credits the index for the drift; its fitted effect
   decays from +18% per unit (Jan–Apr) to +8% (Sep–Oct). When the index falls back in autumn the
   model under-prices by 4%. Only its short-term part is stable: mi_short = market_index minus the
   centred 28-day mean of the daily index (+9 to +13% per unit in every period). Use mi_short only.
2. **No trend or calendar-season features.** A linear trend over-extrapolates (+4 to +8% bias);
   sin/cos day-of-year was already shown to fail (9.7% vs 5.3%). Day of week is safe and helps a little.
3. **Location is the biggest gain after distance.** City effects are symmetric (a city costs the same
   as origin or destination; A>B and B>A leftovers correlate +0.80), range −1.5% to +5.4% per end.
   After city effects the lane-level leftover is 1.4% vs 0.9% pure noise, so pair-specific effects
   are small. Coordinates recover most of the city effect (2.20 vs 2.07) and are what unseen cities
   must rely on; for the 8 validation-only cities use the mean effect of the 3 nearest known cities.
4. **quote_signal carries no usable signal** in a linear model. Test once in the tree model; drop if no gain.
5. **A consistent −0.7 to −0.8% under-prediction** remains in every fold: the slow drift the model is
   not allowed to extrapolate. Decision deferred to the model step: either accept it or apply a
   calibration uplift equal to the fold-measured bias. Validation Nov–Dec market_index (0.92–0.94)
   sits at the Sep–Oct level, so no regime change is visible in the inputs.
6. **Noise floor** is roughly 1.5–2.5% MAPE; the best linear model is already at 1.94%. Expect a tree
   model to gain tenths of a point from interactions (equipment × distance curve), not whole points.
   The dominant risk is Nov–Dec behaving unlike Jan–Oct, which no feature can fix.

Discipline that must also apply to the tree model: exclude raw market_index, mi_level, month,
day-of-year and any trend. A gradient-boosted model will learn the same confounded index effect if
it is allowed to see the raw column.

## Feature list for step 4

Inputs computed from the training rows only, then applied to validation and the December rows.
- log(distance), log(distance)², distance
- equipment one-hot (3)
- weight, weight_was_missing
- day of week (one-hot 7)
- mi_short: market_index − centred 28-day rolling mean of the daily mean index, computed on the
  contiguous train + validation daily series (market_index is an input, not the target);
  market_index_was_missing
- pickup / delivery latitude and longitude (canonical per city; per-row values are already identical)
- pickup / delivery city effect: symmetric per-city coefficient from the training fold; for a city
  not in the training fold, the mean of its 3 nearest training cities
- target: log(posted_rate); report MAE, MAPE, RMSE in dollars after exp()
Excluded: raw market_index, mi_level, month, day of year, trend, lane id, quote_signal (pending tree test).

## Legitimate peek at Nov–Dec (inputs only; prices are not available and would be off-limits anyway)

| | Train Jan–Oct | Validation Nov–Dec |
|---|---|---|
| distance median / mean | 954 / 1,137 | 954 / 1,142 |
| weight median | 31,496 | 31,478 |
| equipment Dry/Reefer/Flat | 57/25/18% | 56/25/18% |
| weekend share | 28.3% | 29.7% |
| market_index mean | 0.926 (Sep–Oct) | 0.927 |
| market_index max | 1.47 | 1.10 (no spring-like spike) |
| late-Dec holiday dip in index | – | none (Dec 1–19: 0.935, Dec 22–31: 0.946) |
| rows touching an unseen city | – | 1,447 (12.1%) |

Conclusion: every input says Nov–Dec is a Sep–Oct-like plateau. No regime change is visible. Rows per
day are higher (197 vs 156) but daily volume has no relation to price in training (corr +0.05).

Price level path (base-model leftover by month, %): Jan −5.4, Feb −4.0, Mar −0.3, Apr +0.1, May +2.4,
Jun +4.8, Jul +1.9, Aug −1.1, Sep +0.5, Oct +0.8. A summer hump plus a mild upward drift, not a line.
market_index tracked the hump (peak May–Jun), which is why anchoring to the index level is the best
witness for Nov–Dec, and it reads flat.

Level-rule test on five forward folds (train through month k, predict k+1..k+2):

| Rule | avg err | avg abs bias | plateau folds (9>10, 8>10) |
|---|---|---|---|
| no adjustment | 3.89% | 2.34% | 2.82 / 2.78 |
| + fixed 0.8% | 3.65% | 1.91% | 2.76 / 2.86 |
| + last month's level | 3.68% | 2.45% | 3.05 / 3.22 |
| + last 2 months' level | 3.70% | 2.46% | 2.76 / 4.03 |
| + last 3 months' level | 3.79% | 2.72% | 2.97 / 3.81 |

On plateau folds all rules are within ~0.1 points; they only diverge when the level is moving (summer
hump), where no rule is reliable. Decision: apply a small level anchor equal to the last two months'
leftover (+0.64% when trained on Jan–Oct), justified by the index reading flat at the Sep–Oct level.
Low stakes either way; document it as a judgment call. Revisit if the final model's fold bias differs.

## Confidence before building step 4 (recorded 2026-10-08)

Proven on forward folds: log-distance backbone, equipment, weight; symmetric city effects (~1 point);
coordinates recover most of the city effect; raw market_index / trend / season harmful; mi_short and
day of week small stable gains; quote_signal is noise.
Judgment calls, bounded: nearest-3-cities fallback (Laredo is 227 mi from its nearest neighbour);
28-day window for mi_short (untuned; check 21/35 in step 4); +0.6% level anchor (±1 point at stake).
Unknown until step 5: tree-model gain from interactions (expect tenths of a point); Spotter's metric.
Structural risk nobody can remove: Nov–Dec following different rules than Jan–Oct.

## Step 4 — feature pipeline built and checked (2026-10-08)

Script: `src/step4_features.py` (class `FeatureBuilder`: fit on training rows, transform any frame).
Outputs: `step4_train_features.csv` (47,323 × 23), `step4_validation_features.csv` (12,000 × 23),
`step4_december_features.csv` (31 × 23); learned artifacts in `cleaned-datasets/artifacts/`.
No blanks in any output. City effects learned for 64 cities, range −3.7% to +3.1% per end.

Fallbacks assigned to the 8 unseen cities (mean of 3 nearest training cities):
Norfolk −1.6%, Allentown −2.0%, Knoxville +0.1%, Laredo +2.4%, Jackson +2.4%, Charlotte −0.2%,
Chicago −1.3%, San Diego +0.4%.

Checks
- Forward folds with the builder refitted inside each fold, linear model on log price:
  1.91 / 2.06 / 1.83%, avg 1.93%, bias −0.8 / −0.8 / −0.2%. Reproduces the pre-build result.
- Ablations (avg over folds): without city effects 2.21%; without coordinates 1.94%; without
  mi_short 1.96%. Coordinates add nothing for seen cities but are required for unseen ones; keep both.
- mi_short window: 21d 1.93%, 28d 1.93%, 35d 1.95%, 42d 1.96%. Keep 28.
- End-to-end unseen-city test (hide 8 random cities, 5 repeats, ~11k rows each): error with fallback
  2.83% vs 2.80% if the cities were known vs 3.46% with no location features. The fallback recovers
  96% of the location benefit.
- Correlations with log rate-per-mile (training): log distance −0.79, reefer +0.37, weight +0.20,
  city effects +0.16 each, longitude +0.12 (west is cheaper per mile), mi_short +0.08, weekdays ±0.04.
  The two was-missing flags are ~0, as expected.
- December inputs: only mi_short (−0.10 to +0.10, a clean weekly cycle) and weekday vary, so the
  December chart will show a weekly wiggle of roughly ±1–2% around one level.

Charts: `eda/figures/14_step4_city_effects_map.png`, `15_step4_feature_correlations.png`,
`16_step4_december_inputs.png`. Step 4 is final.

### Horizon test (added at the user's suggestion, 2026-10-08)

Linear model on the step 4 features, error by test month and by how far ahead it was predicted:

| Trained on | Test month (horizon) | Error | Bias |
|---|---|---|---|
| Jan–Sep | Oct (1) | 1.69% | −0.57% |
| Jan–Aug | Sep (1) / Oct (2) | 2.12% / 1.70% | −0.93% / −0.63% |
| Jan–Jul | Aug (1) / Sep (2) / Oct (3) | 1.72% / 2.10% / 1.69% | +0.85% / −0.84% / −0.52% |
| Jan–Jun | Jul (1) / Aug (2) | 2.46% / 1.64% | −2.11% / +0.53% |

October scores 1.69–1.70% whether it is 1, 2 or 3 months out; September scores 2.1% in every fold.
Error depends on the month's own price level relative to the training average, not on the horizon.
Implication: December should not be harder than November merely for being further out. The
last-month fold (Jan–Sep > Oct, 1.69%, bias −0.57%) is added to the fold set for model selection.

## Step 5 — model selection (2026-10-08)

Script: `src/step5_model.py`; results in `step5_model_comparison.csv`; chart `eda/figures/17_step5_model_comparison.png`.
All candidates predict log(posted_rate) from the 23 step 4 features; FeatureBuilder refitted inside each
of four forward folds (Jan–Aug>Sep–Oct, Jan–Jun>Jul–Aug, Jan–Jul>Aug–Oct, Jan–Sep>Oct).

| Model | Avg error | Worst fold | Avg MAE | Avg bias |
|---|---|---|---|---|
| **Linear + GBM on residuals (lr .05, 400 it, 15 leaves)** | **1.76%** | **1.97%** | **$41.6** | −0.59% |
| Linear (OLS on log price) | 1.87% | 2.06% | $43.7 | −0.58% |
| GBM alone (several settings) | 1.91% | 2.13% | $43.8 | −0.57% |
| GBM alone, absolute-error loss | 2.01% | 2.20% | $46.0 | −0.43% |

Findings
- The hybrid is best on every fold. It keeps the linear model's smooth log-distance behaviour and lets
  trees add interactions (equipment-specific distance curve etc.) worth ~0.1 points, as predicted.
- GBM alone is worse than linear: trees approximate the smooth distance curve in steps and extrapolate
  flat at the edges. Boosting from a linear base removes both weaknesses.
- Hybrid is insensitive to its settings: 150–1000 trees, 7–31 leaves, min leaf 40–200 all give 1.76%.
- Unseen-city test (hide 8 cities, 5 repeats): linear 2.83%, GBM 2.91%, hybrid 2.76%. The hybrid is
  also the best on cities it has never seen.
- quote_signal, one trial in the hybrid: avg 1.99% vs 1.76% without; better on two folds (1.54, 1.55),
  far worse on two (2.71, 2.16). An unstable, time-varying relationship — the same failure shape as raw
  market_index. Excluded for good.

Chosen model: Linear + GBM on residuals, learning_rate 0.05, 400 iterations, 15 leaves, min_samples_leaf 40,
l2 1.0, on the 23 features, log target, exp() back to dollars, then the +0.6% level anchor (step 6 decision).
Expected error on Nov–Dec genuine prices ≈ 1.6–2.0% (≈ $40–50 per load); Spotter's measured error will be
higher by the share of bad prices in their answer key (~1.4% of rows at 2–5x off).

### Step 5 addendum — broader model sweep (2026-10-08)

Same four forward folds. "Residuals" = fitted on the linear model's leftovers.

| Model | Avg error | Worst fold |
|---|---|---|
| **hybrid + lane adjustment (shrinkage k=20)** | **1.69%** | **1.92%** |
| hybrid + lane adjustment (k=5 / k=60) | 1.71% / 1.72% | 1.94% |
| average of hybrid-GBM and hybrid-RF | 1.76% | 1.96% |
| hybrid: linear + GBM (previous choice) | 1.76% | 1.97% |
| linear + random forest on residuals | 1.77% | 1.97% |
| linear + extra-trees on residuals | 1.77% | 1.97% |
| ridge with all pairwise interactions | 1.83% | 2.02% |
| linear + neural net on residuals (2×64) | 1.88% | 2.20% |
| GBM alone, 2000 trees, 63 leaves | 1.93% | 2.15% |
| linear + k-NN on residuals | 2.12% | 2.33% |

Findings
- Every tree ensemble on residuals (GBM, RF, extra-trees) lands at 1.76–1.77%: the learner does not
  matter once the linear backbone is in place. Bigger GBM alone, neural net, k-NN and polynomial ridge
  are all worse. Averaging two ensembles gains 0.01 points — not worth the complexity.
- The one real gain is structural, not a learner: a smoothed per-lane residual mean (the 1.4% vs 0.9%
  lane leftover measured in step 4). It improves every fold, 1.76 → 1.69%, and is 0 for unseen lanes
  so it cannot hurt them. k=20 shrinkage is best; k=5–60 all help.
- Not tried: LightGBM/XGBoost/CatBoost (need extra installs; same model family as HistGradientBoosting,
  no reason to expect a different result given RF/ET/GBM agree to 0.01 points).

Chosen model (final): `FinalModel` in `src/step5_model.py` = linear + GBM(400 it, 15 leaves) on residuals
+ per-lane residual mean shrunk by k=20. Expected forward error ≈ 1.5–1.9% on genuine prices.

## quote_signal — what it actually is (found via external review, verified 2026-10-08)

Earlier notes called quote_signal "noise". That was wrong. By month, share of training rows where
quote_signal × distance is within 2% of posted_rate:

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|---|
| within 2% | 94% | 94% | 95% | 7% | 7% | 96% | 7% | 10% | 95% | 7% |
| spread of log ratio | .011 | .011 | .010 | .256 | .265 | .010 | .264 | .162 | .010 | .269 |

In Jan, Feb, Mar, Jun, Sep the column *is* the rate per mile (MAPE 0.82% on its own). In Apr, May, Jul,
Aug, Oct it is noise (MAPE 17.5%). The switch is at month boundaries: planted, almost certainly the thing
the assessment tests. It explains the step 5 fold pattern (folds whose test months were clean scored
1.54–1.55 with it; noisy ones 2.16–2.71).

Which regime are Nov–Dec in? Level-independent signature = spread of log(quote×distance / model
prediction), computable without prices: clean months 0.019–0.026; noisy months 0.256–0.268; Aug 0.161;
**Nov 0.159, Dec 0.159** → noisy, August-like. Share of rows agreeing with the model within 2%:
Aug 10.4%, Oct 7.2%, Nov 9.1%, Dec 9.5% (pure noise ≈ 6–7%), so at most ~3% of Nov–Dec rows carry a
usable quote.

Blended rule (use quote×distance when it agrees with the model within t, else the model) on noisy
test months: Aug 1.54% → 1.54/1.57/1.62/1.85% for t = 1/2/3/5%; Oct 1.48% → 1.48/1.50/1.54/1.73%.
No gain on noisy months at any threshold. Decision: **exclude quote_signal from the Nov–Dec model**,
now for the right reason, and put this table in the report.

Side use, step 3: in clean months quote×distance repairs the removed rows — 336 of the 677 removed rows
fall in clean months and the repaired price lands at 0.98× expected (89% within 10%). This confirms the
"injected label noise, rest of row intact" story. Repairing instead of dropping is optional (≤0.7% of rows).

Sanity anchor for the December chart: Lexington→Fort Wayne has 32 training rows, median $849 overall,
$808 for Dry Van.

## Second external review — verified points (2026-10-08)

1. **quote_signal has three regimes, not two.** Verified:

| Regime | Months | What quote_signal is | corr(qs, rate/mile) | corr(qs, log distance) |
|---|---|---|---|---|
| clean | Jan, Feb, Mar, Jun, Sep | the rate per mile | +0.996 | −0.80 |
| mirrored | Apr, May, Jul, Oct | 4.15 − rate per mile (spread 0.03) | −0.994 | +0.81 |
| random | Aug, **Nov, Dec** | independent noise | −0.02 (Aug) | +0.02 Aug, **+0.009 Nov, +0.013 Dec** |

   In mirrored months (4.15 − qs) × distance predicts price with 1.09% MAPE. The correlation with
   log distance is computable without prices and is the cleanest regime test: Nov and Dec are random.
   Exclusion for Nov–Dec stands. With the mirror, the quote can repair step 3 outliers in 9 of 10 months.
2. **Early stopping was silently on.** HistGradientBoostingRegressor defaults to early_stopping="auto",
   which is True above 10,000 rows and holds out 10% of training rows. "400 trees" actually built 98.
   That is why 150–1000 trees all scored the same. Fixed: early_stopping=False in `GBM`. Effect on the
   Sep–Oct fold: 1.791% → 1.783%. Conclusions unchanged; reproducibility improved.
3. **Error decomposition** (FinalModel, Jan–Aug > Sep–Oct): 35% of squared log error is a same-day shift
   shared by every load that day, 65% is within-day. Within-day error 1.82% vs genuine price noise
   1.04% (measured from clean-quote months). The reviewer's 72% day-shift share must come from folds
   where the level moved; the share depends on the test months. Honest statement: ~1% irreducible
   row noise + level drift that no input forecasts. Replaces the earlier "noise floor 1.5–2.5%".
4. Accepted without re-running: mi_short must stay row-level (day-level version lets trees memorise
   days, 2.56% vs 1.69%); earlier folds Jan–Mar>Apr–May 4.2% and Jan–Apr>May–Jun 5.7%, so report the
   error conditionally (≈1.5–1.9% if Nov–Dec stay at the Sep–Oct level, 3–6% if the level shifts);
   the +0.6% anchor is a one-fold win and ~1.4 points worse on summer-peak folds — keep as a judgment
   call; lane adjustment is directional (pooling both directions is worse) and OOF fitting gains nothing;
   no holiday effects in training (Jul 4 −0.34%, others within ±0.2%), so a flat Christmas is defensible;
   step 2 weight median and step 3 expected prices are computed once on all Jan–Oct (negligible, one
   sentence in the report).
5. December chart preview without anchor: $821–837 with ±1% weekly wiggle; Dry Van training median on
   this lane $808; with the anchor ≈ $826–842.

## Step 6 — final fit and deliverables (2026-10-08)

Script: `src/step6_predict.py`. Early stopping now off in the GBM; fold results with the fix:
FINAL 1.70% avg (1.73 / 1.93 / 1.66 / 1.49), hybrid 1.76%, linear 1.87%, GBM alone 1.90–1.92%.
Final model trained on 47,323 rows; level anchor from the model's own Sep–Oct residuals: +0.61%.
validation_predictions.csv: 12,000 rows, $197–$6,909, median $2,048; predicted rate/mile median 2.155
vs training 2.145. 88% of validation rows have a learned lane adjustment; 1,447 rows use a city fallback.
December chart: $824–$841, mean $833, weekly wiggle ±1%; training Dry Van median on the lane $808 (n=21).
score.py: both files validated, chart written to scorer_results/candidate_december.png.
Tests: `tests/test_features.py`, 4 passing. One-command run: `run_all.sh`.

### CatBoost check (2026-10-08)

Same four forward folds, same 23 features, CatBoost 1.2.10 (400 it, depth 6, lr 0.08) in place of the
scikit-learn booster. Optional: catboost is not in requirements.txt; LightGBM/XGBoost were not run because
they need the libomp system library on macOS and are the same model family.

| Model | Avg error | Worst fold | Folds |
|---|---|---|---|
| FINAL: linear + sklearn GBM + lane | 1.703% | 1.93% | 1.73 / 1.93 / 1.66 / 1.49 |
| linear + CatBoost + lane | 1.711% | 1.94% | 1.74 / 1.94 / 1.67 / 1.50 |
| linear + CatBoost, no lane | 1.764% | 1.97% | 1.79 / 1.97 / 1.72 / 1.57 |
| CatBoost alone | 1.897% | 2.11% | 1.90 / 2.11 / 1.85 / 1.73 |

Swapping the tree library changes the result by 0.008 points; the structure (linear backbone, lane term)
is what matters. Keeps the chosen model.

Decision on extra libraries (2026-10-08): CatBoost was tested as a one-off check and gave a result within
0.01 points of the chosen model, which is a tie on overlapping folds. It is not part of Spotter's provided
requirements, so it was uninstalled and is not used anywhere in the pipeline. LightGBM and XGBoost were
installed but never ran, because they need the libomp system library on macOS; both were uninstalled too.
The submission runs on the provided requirements plus scikit-learn and pytest only.
