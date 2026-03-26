# Experiment Layout

All experiments should follow this layout:

```text
scripts/
  exp001/
    train.py
    feature_engineering.py
    outputs/
      validation_summary.json
      metrics.json
      fold_scores.csv
      oof_predictions.csv
      feature_importance.csv
      submission/
        submission.csv
        test_predictions.csv
```

Rules:

- `scripts/expXXX/` contains the code for one experiment
- `scripts/expXXX/outputs/` contains the outputs for that experiment
- save validation outputs inside that experiment's `outputs/` directory
- save per-fold validation scores in `outputs/fold_scores.csv`
- save the OOF dataframe in `outputs/oof_predictions.csv`
- save inference artifacts in `outputs/submission/`
- build `outputs/submission/submission.csv` against `data/hello_world_submission-1773575379610.csv`
- when an experiment is logged to `wandb`, point to files from that experiment's `outputs/` directory

# Current Modeling Flow

The current pipeline is a patient-level tabular training workflow.

## Targets

Each experiment trains two separate LightGBM classifiers, one per endpoint:

- `risk_hepatic_event` from raw column `evenements_hepatiques_majeurs`
- `risk_death` from raw column `death`

The supervised label `y` is binary for each endpoint:

- `1` means the event occurred
- `0` means the event did not occur

Rows with null target values are excluded for that endpoint before training.

## Feature Flow

The raw challenge table is in wide patient format, with repeated visit columns such as `Age_v1`, `Age_v2`, and similar patterns for labs and clinical variables.

For each experiment:

1. Load the raw train table and test table.
2. Build the cumulative engineered feature set with `build_cumulative_features(up_to_exp=N)`.
3. Select the feature columns for that experiment.
4. Build one training matrix per target.

Feature engineering remains tabular. The current code does not train a dedicated sequence model.

## Training Flow

For each target:

1. Build `X` from the engineered feature columns.
2. Build `y` from the binary event label for that endpoint.
3. Run 5-fold `StratifiedKFold` cross-validation.
4. Train `LGBMClassifier` on each fold.
5. Use `predict_proba(...)[:, 1]` as the risk score.
6. Collect out-of-fold predictions for validation.
7. Refit one full model on all available rows for test inference.

So the model is still a binary classifier, but its probability output is used as a ranking score.

## Model Output

For each patient and each endpoint, the model outputs a continuous risk score:

- `risk_hepatic_event`
- `risk_death`

These are produced from `predict_proba(...)[:, 1]` and written into:

- validation OOF tables under `outputs/oof_predictions.csv`
- test predictions under `outputs/submission/test_predictions.csv`
- final submission file under `outputs/submission/submission.csv`

## Evaluation Metric

The evaluation has been updated to match the challenge description.

Per endpoint, validation uses `concordance_index_censored` from `scikit-survival`.

### Time Definition

The C-index needs an event indicator and an event time.

For each endpoint:

- if the event occurred, the time is the corresponding event-age column
- if the event did not occur, the time is the patient's last observed age from the `Age_v*` columns

Current mappings:

- hepatic event time uses `evenements_hepatiques_age_occur`
- death time uses `death_age_occur`

This means the code evaluates whether higher predicted risk correctly ranks patients relative to observed event timing or censoring time.

### Final Combined Score

The official validation score is:

$$
\text{Score} = 0.3 \times \text{C-index}_{\text{death}} + 0.7 \times \text{C-index}_{\text{hepatic}}
$$

This weighted score is the current `combined_score` used for experiment comparison and baseline acceptance decisions.

## Secondary Metrics

The pipeline still records these secondary diagnostics for debugging and comparison:

- ROC AUC
- average precision

These are no longer the primary model-selection metric.

## Output Files

Key outputs produced by a standard experiment run:

- `validation_summary.json`: main validation summary, weighted combined score, and per-target metrics
- `metrics.json`: compact metric payload for logging
- `fold_scores.csv`: fold-level metrics, including per-fold C-index
- `oof_predictions.csv`: out-of-fold predictions on the training set
- `feature_importance.csv`: LightGBM feature importances
- `submission/test_predictions.csv`: raw test-set scores
- `submission/submission.csv`: final submission aligned to the template
