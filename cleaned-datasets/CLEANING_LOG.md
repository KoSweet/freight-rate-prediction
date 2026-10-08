# Cleaning log

Each step is one script in `src/`, run in order. Each step reads the previous step's output
and writes `cleaned-datasets/stepN_train.csv` and `stepN_validation.csv`. The same rule is applied
to training and validation; rows are only ever removed from training.

## Step 1 - flip negative weights (final, confirmed 2026-10-08)

Script: `src/step1_flip_sign.py`. Input: raw files. Output: `step1_train.csv`, `step1_validation.csv`.

| | Train | Validation |
|---|---|---|
| Rows | 48,000 | 12,000 |
| Negative weights flipped to positive | 292 | 145 |
| Negative weights remaining | 0 | 0 |
| Blank weights (untouched in this step) | 300 | 165 |

Why flipping (not dropping) is correct - the flipped rows look like a random sample of all other rows:

| | Flipped rows (292) | All other rows |
|---|---|---|
| Months (Jan-Mar / Apr-Jun / Jul-Oct) | 35% / 33% / 33% | 30% / 30% / 40% |
| Median distance | 930 mi | 953 mi |
| Trailer mix (Dry Van / Reefer / Flatbed) | 53% / 23% / 24% | 57% / 25% / 18% |
| Weight quartiles after flip | 25,928 / 31,822 / 37,284 lb | 25,922 / 31,494 / 37,063 lb |
| Median market_index | 1.066 | 1.056 |
| Median rate per mile | 2.157 $/mi | 2.145 $/mi |
| Price / expected price | 1.000 | 1.000 |
| Price change per extra 1,000 lb | +0.25% | +0.30% |

The last row is decisive: heavier loads cost slightly more in both groups. Left negative, the flipped
rows would show the opposite relationship and teach the model that heavier means cheaper.

Charts: `eda/figures/07_step1_weight_sign_flip.png`, `08_step1_weight_outliers.png`.
Weight range after step 1: 5,000-47,500 lb, identical across trailer types, no weight outliers.

Known but not yet handled: 677 training rows (1.4%) priced 2-5x too high or 0.2-0.5x too low
(`eda/figures/09_price_outliers_flagged.png`). Still in the data after step 1.

## Step 2 - fill blank weight and blank market_index (done 2026-10-08)

Script: `src/step2_fill_blanks.py`. Input: step 1 files. Output: `step2_train.csv`, `step2_validation.csv`.
Two new flag columns: `weight_was_missing`, `market_index_was_missing`.

How each blank is filled:
- **weight** -> the training median, 31,496 lb. Weight is the same in every month and trailer type
  (~31,300 lb), so there is nothing better to condition on. Weight moves price only ~0.3% per 1,000 lb.
- **market_index** -> the mean of the other loads on the same day in the same file. market_index is a
  daily market level (within one day the spread is 0.025; day to day it moves 0.167), so the other
  loads that day pin it down. Fallback if a whole day were blank: the training mean 1.083 (never needed).

| | Train | Validation |
|---|---|---|
| Blank weights filled | 300 | 165 |
| Blank market_index filled | 374 | 249 |
| Fallback used | 0 | 0 |
| Blanks remaining | 0 | 0 |

Accuracy test (hide 2,000 known values, fill them the same way, compare):
- market_index: typical miss 0.016, 90% within 0.041, on a scale that spans 0.68-1.47.
- weight: typical miss 5,524 lb, which moves price by ~1.7%, below the ~6.5% row-to-row noise.

Blank rows are priced like everyone else (median rate/mile 2.12 vs 2.15), so the blank carries no
hidden meaning; the flags are kept only so the model can learn otherwise if it wants.

Side finding from chart 10: market_index has a strong weekly zigzag and two level shifts
(up mid-April, down late July). Day of week will matter as a feature.

Charts: `eda/figures/10_step2_market_index_fill.png`, `11_step2_fill_accuracy.png`.

### Step 2 addendum - alternatives considered for market_index (recorded 2026-10-08)

Test: hide 3,000 known market_index values, refill them each way, measure the miss.

| Method | Typical miss | 90% within | Cannot fill |
|---|---|---|---|
| **Same day, all loads (chosen)** | **0.016** | **0.041** | 0 |
| Same day, same equipment | 0.017 | 0.042 | 0 |
| Same weekday, week before and week after (national) | 0.020 | 0.061 | 0 |
| Same day, same pickup city | 0.021 | 0.049 | 307 |
| Same weekday, two weeks before and after | 0.023 | 0.065 | 0 |
| Same pickup city, same weekday, adjacent weeks | 0.023 | 0.066 | 59 |
| Average of day before and day after | 0.027 | 0.061 | 12 |
| Overall training mean | 0.135 | 0.268 | 0 |

Why same-day wins: market_index is a national daily level. A load differs from its own day's
average by 0.017 (pure noise, unrelated to city, equipment, distance or location), whereas the
market moves 0.017 from one week to the same weekday of the next, and 0.027 between adjacent days.
The other ~157 loads on the same day are therefore the closest neighbours available.

Why the weekday-trend idea is still adopted: it is the best method that does not need loads from the
same day, so it is now the fallback for a fully blank day (same weekday +/-7 days, then training mean).
No training or validation day was fully blank (min 114 non-blank loads per day), so the outputs are
unchanged. The December chart inputs have no market_index at all; same-day values from the
validation file cover those dates, and this fallback covers any gap.

Reverse-calculating weight from price was also considered and rejected: on 2,000 hidden weights it
missed by 10,433 lb vs 5,590 lb for the median, 25% of answers fell outside the possible range, and
it cannot be applied to validation, which has no price.

Comparison chart: `eda/figures/12_step2_fill_method_comparison.png`. Two real examples: on Monday
9 June (calm week) both methods agree within 0.002; on Monday 28 July (market dropped that week)
the same-weekday average gives 1.037 while the same-day loads show 0.954, a miss of 0.083. Head to
head on 3,000 hidden values: same-day closer on 1,689, same-weekday closer on 1,311; same-day wins on
every weekday. Conclusion: same-day reads the market directly; same-weekday assumes no shift between
weeks, which usually holds but not always. Step 2 is final.

## Step 3 - remove bad-price rows from training (done 2026-10-08)

Script: `src/step3_remove_bad_prices.py`. Input: step 2 files.
Output: `step3_train.csv` (47,323 rows), `step3_train_removed.csv` (677 rows, with expected price and
ratio), `step3_validation.csv` (12,000 rows, identical to step 2 - validation is never trimmed).

Rule: expected rate per mile = median of the central cloud for the same equipment and distance bucket.
Remove if posted_rate / expected > 2.0 or < 0.5.

| | Count | Share |
|---|---|---|
| Too high (2.16x-5.4x expected) | 340 | 0.7% |
| Too low (0.17x-0.47x expected) | 337 | 0.7% |
| Removed total | 677 | 1.4% |

The cut sits in empty space: the nearest kept row is at 1.23x / 0.79x, the nearest removed row at
2.16x / 0.47x. No genuine load is clipped.

Why they cannot be repaired: the error factor is random (not a clean x3 or x10), so the true price
cannot be recovered. The rest of each row is normal. They are spread evenly across months (1.3-1.6%),
equipment (1.4-1.5%), distance buckets (1.2-1.7%) and cities, consistent with injected label noise.

Impact test (same model trained Jan-Aug with and without the 677 rows, tested on Sep-Oct genuine prices):

| Model type | Trained with bad prices | Trained without |
|---|---|---|
| Log-price linear model | 4.61% | 4.62% |
| Dollar-space leaf-average model (how a default regression tree behaves) | 3.94% | 3.70% |

Honest reading: a log-space model is nearly immune because the high and low errors are symmetric and
cancel. A dollar-space model is not: the 2-5x rows pull leaf averages up and cost about 0.25 points
of error. Removing costs 1.4% of data and protects whichever model is used. A second benefit: the
holdout score is only meaningful on genuine prices (7.0% vs 4.6% mean error when bad rows are left in
the test set), so the removal also makes model comparison cleaner.

Note for the report: Spotter's hidden November-December answers almost certainly contain the same
~1.4% of bad prices. They cannot be detected without the price and will add roughly 2.4 points to
every candidate's measured error equally.

Chart: `eda/figures/13_step3_before_after.png`.
