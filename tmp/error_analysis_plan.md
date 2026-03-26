# Error Analysis Plan

## Objective

Analyze failure patterns for the current best pre-blend target-specific models:

- Hepatic endpoint: CatBoost OOF scores from `scripts/exp039/outputs/oof_predictions.csv`
- Death endpoint: LightGBM OOF scores from `scripts/exp042/outputs/oof_predictions.csv`

The goal is to identify which patient groups are consistently misranked, what shared characteristics appear in those cases, and what hypotheses should guide the next feature/model iteration.

## Inputs

- Raw train table: `data/DB-1773398340961.csv`
- Hepatic OOF source: `scripts/exp039/outputs/oof_predictions.csv`
- Death OOF source: `scripts/exp042/outputs/oof_predictions.csv`
- Hepatic feature importance: `scripts/exp039/outputs/feature_importance.csv`
- Death feature importance: `scripts/exp042/outputs/feature_importance.csv`

## Join Strategy

- Join key: `patient_id_anon`
- Source of truth for raw patient attributes and event-age columns: raw train table
- OOF sources provide model scores and target copies for alignment checks

## Analysis Outputs

The script should write all outputs under `tmp/error_analysis_outputs/`.

### Data products

- `master_error_analysis_table.csv`
- `hepatic_decile_summary.csv`
- `death_decile_summary.csv`
- `hepatic_subgroup_summary.csv`
- `death_subgroup_summary.csv`
- `hepatic_hard_false_negatives.csv`
- `hepatic_hard_false_positives.csv`
- `death_hard_false_negatives.csv`
- `death_hard_false_positives.csv`
- `hepatic_feature_bucket_summary.csv`
- `death_feature_bucket_summary.csv`
- `hepatic_top_feature_importance.csv`
- `death_top_feature_importance.csv`
- `error_analysis_summary.md`

### Plots

- `hepatic_score_distribution.png`
- `death_score_distribution.png`
- `hepatic_decile_lift.png`
- `death_decile_lift.png`
- `hepatic_key_feature_boxplots.png`
- `death_key_feature_boxplots.png`
- `hepatic_subgroup_error_rates.png`
- `death_subgroup_error_rates.png`

## Core analysis questions

1. Which positive-event patients receive unexpectedly low scores?
2. Which non-event or censored patients receive unexpectedly high scores?
3. Do errors concentrate in specific subgroups such as sex, diabetes, hypertension, age bands, BMI bands, or sparse follow-up patterns?
4. Are errors associated with later event ages, shorter follow-up, or inconsistent longitudinal biomarker trajectories?
5. Which high-importance clinical features differ most between confused cases and correctly ranked cases?
6. What patterns are shared across hepatic and death models?

## Error bucket definitions

For each target:

- `hard_fn`: positive event case with score in the bottom quartile of positive-case scores
- `hard_fp`: negative/non-event case with score in the top decile of negative-case scores
- `well_ranked_positive`: positive event case with score in the top quartile of positive-case scores
- `well_ranked_negative`: negative/non-event case with score in the bottom quartile of negative-case scores
- `ambiguous`: everything else

Also compute a threshold-style view using a Youden-style ROC threshold for reference, but keep ranking-oriented slices as the primary lens because the official metric is weighted C-index.

## Feature context to engineer for interpretation

Create patient-level summary columns from visit-style longitudinal raw columns for:

- `Age`
- `BMI`
- `alt`
- `ast`
- `bilirubin`
- `chol`
- `ggt`
- `gluc_fast`
- `plt`
- `triglyc`
- `fibrotest_BM_2`
- `fibs_stiffness_med_BM_1`

For each, derive:

- first non-null
- last non-null
- mean
- min
- max
- std
- delta
- observation count
- missing fraction

## Validation checks

- `patient_id_anon` must remain unique after merging.
- Hepatic and death OOF rows must align with raw train rows.
- Row counts after target filtering must match the experiment summaries.
- Output tables and plots must exist for both targets.
- Summary markdown must be consistent with generated subgroup and case tables.

## Expected deliverable

A runnable standalone script in `tmp/run_error_analysis.py` that produces a first-pass evidence pack for model confusion analysis without retraining any model.