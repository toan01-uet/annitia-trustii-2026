# ANNITIA MASLD Risk Stratification Workspace

This repository has been reset from an old sports-modeling project into a clean workspace for the ANNITIA Data Challenge.

The current purpose of the repo is simple:

- keep the raw challenge data in `data/`
- keep prompt and checklist files that drive an autonomous research loop
- organize experiment code under `scripts/expXXX/`
- save each experiment's files under its own `scripts/expXXX/outputs/`
- document assumptions, validation decisions, and experiment history in markdown
- send each completed experiment's outputs, metrics, and scores to `wandb`

There is intentionally no retained legacy code, artifact, or submission pipeline from the previous project.

## Problem Framing

The task is MASLD risk stratification. We need to generate two risk scores for each patient:

- `risk_hepatic_event`
- `risk_death`

This should be treated as a tabular risk-scoring problem rather than a plain binary classification task. The model should rank higher-risk patients above lower-risk patients, and any probability-like interpretation must be justified by the challenge metric and target definition.

## Local Data Notes

The repo currently contains these local files:

- `data/DB-1773398340961.csv`
- `data/dictionary-1773398867610.csv`
- `data/hello_world_submission-1773575379610.csv`
- `data/test.csv`

There is an important schema nuance:

- the dictionary file describes a clean semantic schema with columns such as `subject_id`, `age_years`, `sex`, `bmi_kg_m2`, `risk_hepatic_event`, and `risk_death`
- the raw training file currently exposed in `data/` is a wide patient table with many repeated visit-style columns such as `Age_v1`, `BMI_v1`, `alt_v1`, `fibrotest_BM_2_v1`, and outcome-style columns such as `evenements_hepatiques_majeurs`, `death`, and event-age columns

Because of that mismatch, all future work in this repo should be schema-first:

1. inspect the actual raw file
2. map raw columns to semantic concepts
3. only then design features and models

## Modeling Direction

The preferred direction is feature-engineered tabular modeling with strong validation discipline.

Priority feature families:

- clinical burden features
- ratio and interaction features
- threshold and binary risk flags
- log and other nonlinear transforms
- missingness features
- subgroup-relative normalization
- endpoint-specific feature blocks

If the local raw file keeps repeated measurements in wide form, the default interpretation is still tabular:

- derive within-row summaries such as first/last/mean/min/max/std/delta/slope when justified
- do not jump to sequence models unless the user explicitly asks for that

## Important Files

- `AGENTS.md`: repo rules and modeling constraints
- `CHECKLIST.md`: feature-family backlog
- `checklist-prompt.md`: autonomous feature-engineering loop prompt
- `data-description.md`: challenge and data notes
- `JOURNAL.md`: append-only experiment log
- `scripts/README.md`: experiment directory convention

## Experiment Layout

Each experiment should live in its own folder under `scripts/`.

Expected format:

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

- `scripts/expXXX/` contains the code for that experiment
- `scripts/expXXX/outputs/` contains the outputs produced by that experiment
- save fold-level validation metrics in `scripts/expXXX/outputs/fold_scores.csv`
- save the OOF dataframe in `scripts/expXXX/outputs/oof_predictions.csv`
- when `data/test.csv` is available and inference is requested, save predictions in `scripts/expXXX/outputs/submission/`
- build `scripts/expXXX/outputs/submission/submission.csv` from the `trustii_id` order in `data/hello_world_submission-1773575379610.csv`
- do not scatter experiment outputs across ad hoc top-level folders
- when logging to `wandb`, reference the files saved under that experiment's `outputs/` directory

## Recommended Workflow

1. Start with schema inspection, target semantics, ID uniqueness, missingness, and train/submission alignment.
2. Create the experiment folder under `scripts/expXXX/`.
3. Build a minimal baseline on raw columns plus conservative preprocessing.
4. Save outputs for that run under `scripts/expXXX/outputs/`, including fold metrics, OOF predictions, and any requested submission export.
5. Add one feature family at a time.
6. Log each completed experiment to `wandb`.
7. Accept changes only when validation improves under the current repo rules.
8. Log every meaningful experiment in `JOURNAL.md`.
