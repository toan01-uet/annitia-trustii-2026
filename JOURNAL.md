# JOURNAL.md

## Purpose

- Track meaningful experiments for the ANNITIA MASLD risk-stratification project.
- Keep a lightweight record of what was tried, how it scored under the current validation setup, and what to do next.
- Append entries rather than rewriting history.

## Entry Template

### YYYY-MM-DD HH:MM UTC - Short Experiment Name

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

- `validation_summary.json`
- `metrics.json`
- `fold_scores.csv`
- `oof_predictions.csv`
- `submission/submission.csv`

## Entries

- No experiment entries yet for the reset ANNITIA workspace.

### 2026-03-23 00:00 UTC - Repo Reset And Prompt Rewrite

- Hypothesis: Resetting the repo to a clean prompt-driven workspace will make the next modeling loop faster and reduce contamination from the previous unrelated project.
- Experiment dir: Not run.
- Data window: Local files currently present in `data/`: `DB-1773398340961.csv`, `dictionary-1773398867610.csv`, `hello_world_submission-1773575379610.csv`.
- Split strategy: Not run.
- Targets: Planned targets are `risk_hepatic_event` and `risk_death`, pending confirmation against local schema and challenge rules.
- Feature set: None. Prompt, checklist, and documentation reset only.
- Model/config: Not run.
- Validation metric: Not run.
- Combined score: Not run.
- Per-target scores: Not run.
- Fold scores: Not run.
- Artifacts: None.
- W&B: Not run.
- Notes: Removed legacy code, scripts, artifacts, docs, resources, submissions, and temporary analysis files from the prior project. Rewrote repo guidance for ANNITIA MASLD risk stratification and added schema-first instructions because the local raw CSV and semantic dictionary may not use identical column names.
- Next step: Inspect the raw training schema, build a raw-to-semantic mapping, confirm targets and evaluation metric, then establish the first tabular baseline.

### 2026-03-23 07:32 UTC - Exp001 Baseline Raw Features

- Hypothesis: A shared raw-feature baseline using the existing wide patient table can establish a defensible starting point for both endpoints before any engineered visit summaries are added.
- Experiment dir: `scripts/exp001/`
- Data window: `data/DB-1773398340961.csv` with 1,253 patients and 287 raw columns; death modeling excludes rows with null `death` labels.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: All raw non-ID columns except target and event-age leakage columns. This run also completed the raw-to-semantic schema audit and visit-group inventory.
- Model/config: Separate `LightGBMClassifier` models per target with shared raw feature set; `learning_rate=0.03`, `n_estimators=300`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`.
- Validation metric: `roc_auc_surrogate`.
- Combined score: `0.796399`.
- Per-target scores: `risk_hepatic_event roc_auc=0.807787, average_precision=0.142442`; `risk_death roc_auc=0.785010, average_precision=0.305362`.
- Fold scores: `risk_hepatic_event roc_auc folds=[0.842516, 0.782988, 0.868050, 0.740433, 0.812817]`; `risk_death roc_auc folds=[0.753114, 0.764469, 0.766667, 0.806285, 0.841989]`.
- Artifacts: `scripts/exp001/outputs/schema_audit.json`, `scripts/exp001/outputs/raw_to_semantic_map.json`, `scripts/exp001/outputs/feature_columns.txt`, `scripts/exp001/outputs/validation_summary.json`, `scripts/exp001/outputs/metrics.json`, `scripts/exp001/outputs/feature_importance.csv`.
- W&B: project `annitia-trustii-2026`, run `exp001_baseline_raw_features`, mode `offline` because no API key was configured.
- Notes: Official challenge metric was not confirmed from accessible local materials, so ROC AUC was used as a ranking-oriented surrogate. The schema audit found 13 `_v*` visit groups, with 21 to 22 repeated measurements per group.
- Next step: Test `Wide-visit first/last/delta summary features if _v* columns exist` on top of the baseline.

### 2026-03-23 07:33 UTC - Exp002 Wide Visit First/Last/Delta

- Hypothesis: Adding first observed value, last observed value, last-first delta, absolute delta, and observed-count summaries for each repeated visit family will improve ranking quality beyond the raw baseline.
- Experiment dir: `scripts/exp002/`
- Data window: Same raw table as exp001; candidate adds engineered summaries derived from the 13 `_v*` visit groups.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: Exp001 raw baseline features plus 65 engineered visit-summary features across `Age`, `BMI`, `aixp_aix_result_BM_3`, `alt`, `ast`, `bilirubin`, `chol`, `fibrotest_BM_2`, `fibs_stiffness_med_BM_1`, `ggt`, `gluc_fast`, `plt`, and `triglyc`.
- Model/config: Same `LightGBMClassifier` setup as exp001, keeping the model family fixed so the experiment isolates the effect of the added features.
- Validation metric: `roc_auc_surrogate`.
- Combined score: `0.810944`.
- Per-target scores: `risk_hepatic_event roc_auc=0.834762, average_precision=0.189614`; `risk_death roc_auc=0.787126, average_precision=0.330291`.
- Fold scores: `risk_hepatic_event roc_auc folds=[0.848944, 0.776763, 0.870124, 0.853850, 0.838174]`; `risk_death roc_auc folds=[0.740659, 0.775458, 0.772894, 0.779696, 0.873665]`.
- Artifacts: `scripts/exp002/outputs/added_feature_names.txt`, `scripts/exp002/outputs/feature_columns.txt`, `scripts/exp002/outputs/feature_family_summary.json`, `scripts/exp002/outputs/validation_summary.json`, `scripts/exp002/outputs/metrics.json`, `scripts/exp002/outputs/feature_importance.csv`.
- W&B: project `annitia-trustii-2026`, run `exp002_wide_visit_first_last_delta`, mode `offline` because no API key was configured.
- Notes: The first implementation failed because several visit columns were parsed as mixed string/float types; the helper was updated to cast repeated-measure columns to `Float64` with `strict=False` before calculating summaries. The accepted candidate improved combined ROC AUC by `+0.014545` over exp001.
- Next step: Test `Wide-visit min/max/mean/std summary features if _v* columns exist` in `scripts/exp003/`.

### 2026-03-23 08:06 UTC - Exp001 Baseline Export Refresh

- Hypothesis: The baseline run should emit reusable validation artifacts and a template-aligned submission export when `data/test.csv` is available.
- Experiment dir: `scripts/exp001/`
- Data window: Training on `data/DB-1773398340961.csv`; inference on `data/test.csv` aligned to `data/hello_world_submission-1773575379610.csv`.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: Unchanged raw baseline features. Output schema expanded to include fold metrics, OOF predictions, and submission artifacts.
- Model/config: Same `LightGBMClassifier` setup as the original exp001 baseline.
- Validation metric: `roc_auc_surrogate`.
- Combined score: `0.796399`.
- Per-target scores: `risk_hepatic_event roc_auc=0.807787, average_precision=0.142442`; `risk_death roc_auc=0.785010, average_precision=0.305362`.
- Fold scores: Saved to `scripts/exp001/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp001/outputs/fold_scores.csv`, `scripts/exp001/outputs/oof_predictions.csv`, `scripts/exp001/outputs/submission/submission.csv`, `scripts/exp001/outputs/submission/test_predictions.csv`, plus the existing validation summary files.
- W&B: project `annitia-trustii-2026`, run `exp001_baseline_raw_features`, mode `offline`; `.env` contained a `WANDB_KEY` value but it was not a valid full-length API key for online sync.
- Notes: `test.csv` contains `trustii_id`, and its row order matches the local hello-world submission template, so submission export can be built directly against the template.
- Next step: Refresh the current frontier experiment with the same output contract.

### 2026-03-23 08:06 UTC - Exp002 Frontier Export Refresh

- Hypothesis: The current accepted frontier should export the same reusable artifact bundle as the baseline so it can be used immediately for local submission testing.
- Experiment dir: `scripts/exp002/`
- Data window: Training on `data/DB-1773398340961.csv`; inference on `data/test.csv` aligned to `data/hello_world_submission-1773575379610.csv`.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: Exp001 raw baseline features plus the accepted 65 wide-visit first/last/delta features.
- Model/config: Same `LightGBMClassifier` setup as the original exp002 run.
- Validation metric: `roc_auc_surrogate`.
- Combined score: `0.810944`.
- Per-target scores: `risk_hepatic_event roc_auc=0.834762, average_precision=0.189614`; `risk_death roc_auc=0.787126, average_precision=0.330291`.
- Fold scores: Saved to `scripts/exp002/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp002/outputs/fold_scores.csv`, `scripts/exp002/outputs/oof_predictions.csv`, `scripts/exp002/outputs/submission/submission.csv`, `scripts/exp002/outputs/submission/test_predictions.csv`, plus the existing validation summary files.
- W&B: project `annitia-trustii-2026`, run `exp002_wide_visit_first_last_delta`, mode `offline`; `.env` contained a `WANDB_KEY` value but it was not a valid full-length API key for online sync.
- Notes: This remains the best validated model currently present in the repo and is the correct place to take `submission.csv` from until a stronger accepted experiment exists.
- Next step: Continue feature-family iteration from the current frontier, starting with `Wide-visit min/max/mean/std summary features if _v* columns exist`.
