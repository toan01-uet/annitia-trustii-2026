# JOURNAL.md

## Purpose

- Track temporary analysis runs, diagnostics, and disposable research outputs under `tmp/`.
- Preserve short traceable notes about what was analyzed, what was found, and where the supporting artifacts live.
- Append entries rather than rewriting history.

## Entry Template

### YYYY-MM-DD HH:MM UTC - Short Analysis Name

- Hypothesis:
- Experiment dir:
- Data window:
- Split strategy:
- Targets:
- Feature set:
- Model/config:
- Validation metric:
- Combined score:
- Per-target scores:
- Fold scores:
- Artifacts:
- W&B:
- Notes:
- Next step:

Suggested artifact set when available:

- `error_analysis_plan.md`
- `run_error_analysis.py`
- `error_analysis_outputs/error_analysis_summary.md`
- `error_analysis_outputs/manifest.json`
- target-specific case tables and subgroup summaries

## Entries

### 2026-03-26 06:47 UTC - Best Single-Model Endpoint Error Analysis

- Hypothesis: The current best pre-blend endpoint models may be making systematic mistakes on identifiable patient subgroups, and those shared failure modes can guide the next round of feature engineering or target-side modeling changes.
- Experiment dir: `tmp/`
- Data window: Raw training table `data/DB-1773398340961.csv` joined with OOF predictions from `scripts/exp039/outputs/oof_predictions.csv` for hepatic CatBoost and `scripts/exp042/outputs/oof_predictions.csv` for death LightGBM.
- Split strategy: No retraining. Analysis-only pass over existing OOF predictions, preserving the current validation split behavior implicitly through the saved OOF artifacts.
- Targets: `risk_hepatic_event` analyzed from the best CatBoost single-model lineage; `risk_death` analyzed from the best LightGBM single-model lineage.
- Feature set: Raw patient-level columns plus disposable visit-summary context engineered inside `tmp/run_error_analysis.py` to compare confused vs well-ranked cases across longitudinal biomarker families.
- Model/config: No new model fit. Error analysis references hepatic CatBoost from the exp039 configuration and death LightGBM from the exp042 baseline-selected configuration.
- Validation metric: Descriptive error analysis over ranking outputs; reference threshold diagnostics use Youden-selected cut points for hard FN/FP case slicing.
- Combined score: Not recomputed in this pass. Current canonical ensemble frontier remains `0.752477`, while the analyzed pre-blend single-model sources are the current best endpoint-specific components.
- Per-target scores: Summary inputs correspond to hepatic CatBoost with top-decile event rate `0.183` vs bottom-decile `0.008`, and death LightGBM with top-decile event rate `0.333` vs bottom-decile `0.000`.
- Fold scores: Not generated in this pass; source fold behavior remains in the originating experiment outputs.
- Artifacts: `tmp/error_analysis_plan.md`, `tmp/run_error_analysis.py`, `tmp/error_analysis_outputs/error_analysis_summary.md`, `tmp/error_analysis_outputs/manifest.json`, `tmp/error_analysis_outputs/master_error_analysis_table.csv`, and the per-target decile, subgroup, hard false negative, and hard false positive tables under `tmp/error_analysis_outputs/`.
- W&B: Not logged. This was a disposable local analysis workflow under `tmp/`, not a tracked training experiment.
- Notes: Dataset checks showed `1253` master rows, `1253` unique patients, `47` hepatic positives, and `76` death positives with non-null labels. On the hepatic side, the strongest confusion signals concentrate around fibrosis/stiffness variables, especially `fibs_stiffness_med_BM_1_*`, with hard errors enriched in `visit_coverage_band=dense`, `bariatric_surgery=1.0`, and `T2DM=1.0`. On the death side, the strongest confusion signals concentrate around age and cardiometabolic follow-up context, especially `Age_v2`, `bilirubin_v3`, `gluc_fast_v3`, and `aixp_aix_result_BM_3_v1`, with hard errors enriched in `visit_coverage_band=dense`, `age_band=46-55`, `age_band=66+`, and `Dyslipidaemia=1.0`. Cross-target overlap suggests `Hypertension` and dense visit coverage are shared axes of confusion worth prioritizing.
- Next step: Convert these descriptive findings into targeted hypotheses, starting with death-side features for dense-follow-up and age-risk interactions, plus hepatic-side features that better disambiguate high-stiffness but non-event patients from true hepatic-event cases.
