#!/usr/bin/env bash
# Runs the whole pipeline from the raw files to the two deliverables and Spotter's chart.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY src/step1_flip_sign.py
$PY src/step2_fill_blanks.py
$PY src/step3_remove_bad_prices.py
$PY src/step4_features.py
$PY src/step5_model.py          # fold comparison (~30 s)
$PY src/step6_predict.py        # final fit, validation_predictions.csv, December file, score.py
$PY eda/eda_charts.py && $PY eda/step1_charts.py && $PY eda/step2_charts.py && $PY eda/step3_charts.py && $PY eda/step4_charts.py && $PY eda/step5_charts.py
echo "done: validation_predictions.csv, data/december_chart_inputs.csv, scorer_results/candidate_december.png"
