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

### 2026-01-01 00:00 UTC - Exp003–Exp034 Feature Engineering Chain Scaffolded

- Hypothesis: Implementing a full sequential feature engineering chain (exp003–exp034) covering 33 distinct feature families will systematically explore the feature space and identify which additions improve combined ROC AUC over the exp002 baseline.
- Experiment dir: `scripts/exp003/` through `scripts/exp034/` (32 new directories, code only — not yet run).
- Data window: Same as prior experiments.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set summary:
  - exp003: Wide-visit min/max/mean/std/range per visit group
  - exp004: Slope proxy, trend direction, coefficient of variation
  - exp005: Persistence fraction, early/late coverage
  - exp006: First/last observed visit index, observation span
  - exp007: All-null/any-observed flags, panel coverage
  - exp008: log1p transforms + p1/p99 winsorized biomarkers
  - exp009: Comorbidity count and weighted burden score
  - exp010: Obesity/glucose/triglyceride flags + metabolic burden
  - exp011: ALT/AST/GGT/bilirubin flags + inflammatory burden
  - exp012: Hypertension/cholesterol/AIx flags + cardio burden
  - exp013: Low-PLT + HTN + T2DM renal-cardio score
  - exp014: AST/ALT, GGT/ALT, FIB-4 proxy, TG/chol ratios
  - exp015: AIx mean/age-adjusted, stiffness mean, FibroTest mean
  - exp016: Key pairwise products (FIB-4×FibroTest, burden scores, etc.)
  - exp017: Total burden score, high-risk gate, burden×FIB-4
  - exp018: 3×ULN flags, stiffness F2/F3, FibroTest F2 flags
  - exp019: Quintile bins for key biomarkers
  - exp020: Sex-stratified z-scores for key biomarkers
  - exp021: Age-decade z-scores for key biomarkers
  - exp022: Fractional percentile ranks
  - exp023: Median/IQR robust scaling
  - exp024: Hepatic-event-specific: stiffness worsening, FIB-4 risk flags
  - exp025: Death-specific: age features, elderly flags, cardio-age score
  - exp026: Categorical encoding: gender×BMI/T2DM/age, bariatric interactions
  - exp027: Liver injury composite, fibrosis composite, stiffness/FIB-4 products
  - exp028: TyG index, HOMA-IR proxy, atherogenic index
  - exp029: BMI×ALT/GGT/TG inflammation-obesity products
  - exp030: AIx×age, HTN×chol, cardio×age cardiac stress interactions
  - exp031: Stiffness/PLT, FIB-4/PLT, BMI×stiffness/PLT renal-hepatic interactions
  - exp032: Separate vs shared per-target backbone comparison
  - exp033: Post-hoc isotonic regression OOF calibration
  - exp034: OOF ensemble blending (simple avg + rank avg) across all experiments
- Model/config: Shared `LightGBMClassifier` with the same hyperparameters as previous experiments; each experiment adds one feature family cumulatively.
- Validation metric: `roc_auc_surrogate` (combined mean ROC AUC across both targets).
- Combined score: Not yet run.
- Per-target scores: Not yet run.
- Fold scores: Not yet run.
- Artifacts: Directories and `train.py` files created; `outputs/` produced at runtime.
- W&B: Configured per run; will log each experiment individually.
- Notes: All 30 feature engineering functions added to `scripts/exp_shared.py`. The `build_cumulative_features(df, up_to_exp)` helper chains them in order so each experiment implicitly includes all prior feature families. All checklist items now marked complete.
- Next step: Run experiments sequentially and track which feature families improve the combined score.

### 2026-03-26 02:30 UTC - Exp003 Weighted C-index Refresh

- Hypothesis: Wide-visit min/max/mean/std/range aggregates should remain beneficial after switching model selection from ROC AUC to the official weighted C-index.
- Experiment dir: `scripts/exp003/`
- Data window: Training on `data/DB-1773398340961.csv`; inference artifacts written against `data/test.csv` and the local hello-world submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: Exp002 baseline plus wide-visit min/max/mean/std summary features; 130 features added, 412 total features used.
- Model/config: Separate `LightGBMClassifier` models per target with the shared default LightGBM configuration already used in exp001 and exp002.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.728675`.
- Per-target scores: `risk_hepatic_event cindex=0.789930, roc_auc=0.828588, average_precision=0.170554`; `risk_death cindex=0.585746, roc_auc=0.802081, average_precision=0.363048`.
- Fold scores: `risk_hepatic_event cindex folds=[0.788355, 0.770286, 0.867508, 0.769815, 0.781567]`; `risk_death cindex folds=[0.487179, 0.622642, 0.559722, 0.500000, 0.685007]`.
- Artifacts: `scripts/exp003/outputs/validation_summary.json`, `scripts/exp003/outputs/metrics.json`, `scripts/exp003/outputs/fold_scores.csv`, `scripts/exp003/outputs/oof_predictions.csv`, `scripts/exp003/outputs/feature_importance.csv`, `scripts/exp003/outputs/submission/submission.csv`.
- W&B: run logged under the existing ANNITIA project in offline mode.
- Notes: Under the official weighted C-index, this feature family still improved materially over exp002 by `+0.013151` and became the first post-metric-change frontier.
- Next step: Continue the cumulative feature chain and watch whether more clinically-structured interactions can improve death ranking without giving back hepatic-event ranking.

### 2026-03-26 03:10 UTC - Exp014 Ratio Features New Weighted C-index Frontier

- Hypothesis: Ratio features such as AST/ALT-style interactions and related tabular risk ratios can improve ranking quality beyond the earlier burden and flag feature families.
- Experiment dir: `scripts/exp014/`
- Data window: Same train/test setup as the other refreshed runs.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Full cumulative chain through exp014, with ratio features added on top of the earlier summary, trend, persistence, missingness, transformation, burden, and threshold families; 325 added engineered features, 607 total features used.
- Model/config: Separate `LightGBMClassifier` models per endpoint with unchanged base hyperparameters.
- Validation metric: `cindex_weighted` using `concordance_index_censored`.
- Combined score: `0.730723`.
- Per-target scores: `risk_hepatic_event cindex=0.789619, roc_auc=0.831957, average_precision=0.190320`; `risk_death cindex=0.593300, roc_auc=0.808225, average_precision=0.372591`.
- Fold scores: `risk_hepatic_event cindex folds=[0.755083, 0.764571, 0.862250, 0.762226, 0.818433]`; `risk_death cindex folds=[0.506410, 0.635960, 0.575000, 0.486420, 0.697387]`.
- Artifacts: `scripts/exp014/outputs/validation_summary.json`, `scripts/exp014/outputs/metrics.json`, `scripts/exp014/outputs/fold_scores.csv`, `scripts/exp014/outputs/oof_predictions.csv`, `scripts/exp014/outputs/feature_importance.csv`, `scripts/exp014/outputs/submission/submission.csv`.
- W&B: run logged in offline mode.
- Notes: This run became the best metric-aligned frontier among the cumulative single-model feature-family experiments that currently report the official weighted C-index consistently.
- Next step: Use exp014 as the stable aligned baseline while auditing the late-chain experiments that still carry custom combined-score logic.

### 2026-03-26 03:25 UTC - Exp003–Exp031 Weighted C-index Sweep Summary

- Hypothesis: Running the full cumulative feature chain under the official weighted C-index would identify which feature families genuinely improve the challenge objective, not just surrogate ROC AUC.
- Experiment dir: `scripts/exp003/` through `scripts/exp031/`.
- Data window: Same local train/test files used by the earlier baseline runs.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target across the chain.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Cumulative feature-family chain from wide-visit aggregate summaries through renal-hepatic interaction features.
- Model/config: Shared experiment pattern with separate LightGBM models per endpoint and common hyperparameters.
- Validation metric: `cindex_weighted`.
- Combined score: Best metric-aligned run in this block was `exp014` at `0.730723`; worst run in this block was `exp017` at `0.717736`.
- Per-target scores: Best aligned frontier `exp014` produced `risk_hepatic_event cindex=0.789619` and `risk_death cindex=0.593300`.
- Fold scores: Stored per experiment in each `outputs/fold_scores.csv`.
- Artifacts: Validation summaries, OOF tables, fold-score tables, feature-importance files, and submission artifacts are present under each experiment's `outputs/` directory.
- W&B: All runs logged to the ANNITIA project in offline mode.
- Notes: 29 experiments in this block completed with metric-aligned `cindex_weighted` summaries; 16 were accepted against their immediate predecessor and 13 were rejected. The strongest improvements came from early wide-visit summary features and the later ratio-feature family, while several burden and subgroup-normalization families were neutral to negative under the official ranking metric.
- Next step: Rerun or refactor the late-chain special experiments with fully aligned combined-score logic before deciding whether the frontier should move past exp014.

### 2026-03-26 03:35 UTC - Exp032–Exp034 Metric Alignment Audit

- Hypothesis: The late-chain special experiments should be checked separately because they modify backbone structure, calibration, and ensemble reporting logic rather than only adding another tabular feature family.
- Experiment dir: `scripts/exp032/`, `scripts/exp033/`, `scripts/exp034/`.
- Data window: Same local train/test files.
- Split strategy: Existing per-experiment split logic was executed successfully and artifacts were written.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: `exp032` compares separate vs shared backbones, `exp033` applies post-hoc isotonic calibration, and `exp034` performs OOF ensemble blending.
- Model/config: Custom experiment-specific logic layered on top of the shared LightGBM backbone and OOF prediction pipeline.
- Validation metric: Mixed in current outputs; `exp032` still reports `roc_auc_surrogate`, while `exp033` and `exp034` expose per-target C-index values but keep custom combined-score/reporting logic.
- Combined score: Current output files report `exp032=0.817938`, `exp033=0.817877`, and `exp034=0.819819`, but these values are not directly comparable to the official weighted C-index frontier because the combined-score code path is not fully aligned in these specialized experiments.
- Per-target scores: `exp033` output includes `risk_hepatic_event cindex=0.787518`, `risk_death cindex=0.584433`; `exp034` output includes `risk_hepatic_event cindex=0.791370`, `risk_death cindex=0.585801`.
- Fold scores: Available in each experiment's `outputs/fold_scores.csv`.
- Artifacts: `validation_summary.json`, `metrics.json`, `fold_scores.csv`, `oof_predictions.csv`, and experiment-specific output files under `scripts/exp032/outputs/` through `scripts/exp034/outputs/`.
- W&B: Offline logs present for the specialized late-chain runs.
- Notes: These experiments finished and produced artifacts, but their top-line `combined_score` should not yet be treated as the official leaderboard proxy. The current trustworthy frontier under the official weighted C-index remains exp014 until exp032–exp034 are rerun with fully aligned score aggregation.
- Next step: Patch exp032–exp034 so every reported `combined_score` is computed strictly as `0.3 * C-index_death + 0.7 * C-index_hepatic`, then rerun those three experiments and update the frontier if warranted.

### 2026-03-26 05:31 UTC - Exp036 Target-Specific Zero-Importance Pruning On Exp035 Core

- Hypothesis: A fold-safe target-specific pruning pass that removes zero-importance features inside each training fold can tighten the exp035 core and improve the official weighted C-index without changing the backbone model class.
- Experiment dir: `scripts/exp036/`
- Data window: Training on `data/DB-1773398340961.csv`; validation-only run with no submission export.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target, with feature pruning fit only on each fold's training partition.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Same curated exp035 stack with 519 total features, then fold-local zero-importance pruning from LightGBM for each endpoint; final full-model retained 288 hepatic features and 294 death features.
- Model/config: Two-stage `LightGBMClassifier` per endpoint using the shared default LightGBM parameters for both the pre-pruning and post-pruning fits.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.727534`.
- Per-target scores: `risk_hepatic_event cindex=0.785767, roc_auc=0.829840, average_precision=0.189871`; `risk_death cindex=0.591658, roc_auc=0.805023, average_precision=0.386668`.
- Fold scores: Saved to `scripts/exp036/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp036/outputs/validation_summary.json`, `scripts/exp036/outputs/metrics.json`, `scripts/exp036/outputs/fold_scores.csv`, `scripts/exp036/outputs/oof_predictions.csv`, `scripts/exp036/outputs/feature_importance.csv`, `scripts/exp036/outputs/selected_features_union.txt`.
- W&B: project `annitia-trustii-2026`, run `exp036_target_specific_zero_importance_pruning`, mode `offline`.
- Notes: This pruning rule removed many features but also gave back hepatic ranking signal, dropping the weighted score by `-0.004842` versus exp035. The result suggests exp035's curated stack was already compact enough that zero-gain trees were still supporting the final ranker indirectly.
- Next step: Keep the exp035 feature stack fixed and test alternative model classes rather than more aggressive pruning heuristics.

### 2026-03-26 05:32 UTC - Exp037 CatBoost On Exp035 Core New Frontier

- Hypothesis: CatBoost may exploit the exp035 curated feature stack better than LightGBM, especially on the hepatic-heavy weighted objective, even without changing the feature set.
- Experiment dir: `scripts/exp037/`
- Data window: Training on `data/DB-1773398340961.csv`; validation-only run with no submission export.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Unchanged exp035 curated stack with 519 total features and 237 engineered additions from the accepted-only family selection.
- Model/config: Separate `CatBoostClassifier` models per endpoint with `learning_rate=0.03`, `iterations=300`, `depth=6`, `l2_leaf_reg=3.0`, `random_seed=7`, `loss_function=Logloss`, `allow_writing_files=False`.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.739127`.
- Per-target scores: `risk_hepatic_event cindex=0.821447, roc_auc=0.848184, average_precision=0.194206`; `risk_death cindex=0.547047, roc_auc=0.818253, average_precision=0.278831`.
- Fold scores: Saved to `scripts/exp037/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp037/outputs/validation_summary.json`, `scripts/exp037/outputs/metrics.json`, `scripts/exp037/outputs/fold_scores.csv`, `scripts/exp037/outputs/oof_predictions.csv`, `scripts/exp037/outputs/feature_importance.csv`.
- W&B: project `annitia-trustii-2026`, run `exp037_catboost_on_exp035_core`, mode `offline`.
- Notes: This became the new trustworthy frontier, improving the official weighted score by `+0.006750` over exp035. The gain came from a large hepatic-event C-index jump that more than compensated for a weaker death C-index under the 0.7/0.3 weighting.
- Next step: Treat CatBoost on the exp035 stack as the new baseline and test whether any lighter hyperparameter refinement can preserve the hepatic lift while recovering some death ranking.

### 2026-03-26 05:32 UTC - Exp038 XGBoost On Exp035 Core

- Hypothesis: XGBoost may offer a different bias-variance tradeoff on the exp035 curated stack and could outperform LightGBM without changing features.
- Experiment dir: `scripts/exp038/`
- Data window: Training on `data/DB-1773398340961.csv`; validation-only run with no submission export.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Unchanged exp035 curated stack with 519 total features.
- Model/config: Separate `XGBClassifier` models per endpoint with `learning_rate=0.03`, `n_estimators=300`, `max_depth=4`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`, `eval_metric=logloss`.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.726557`.
- Per-target scores: `risk_hepatic_event cindex=0.786623, roc_auc=0.823784, average_precision=0.186125`; `risk_death cindex=0.586403, roc_auc=0.794330, average_precision=0.348190`.
- Fold scores: Saved to `scripts/exp038/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp038/outputs/validation_summary.json`, `scripts/exp038/outputs/metrics.json`, `scripts/exp038/outputs/fold_scores.csv`, `scripts/exp038/outputs/oof_predictions.csv`, `scripts/exp038/outputs/feature_importance.csv`.
- W&B: project `annitia-trustii-2026`, run `exp038_xgboost_on_exp035_core`, mode `offline`.
- Notes: XGBoost under this simple configuration underperformed both exp035 and exp037, losing `-0.005820` versus the exp035 LightGBM baseline. This points to model-class gains being specific to CatBoost rather than a generic benefit from swapping away from LightGBM.
- Next step: Use exp037 as the new baseline for any follow-up hyperparameter or ensemble work.

### 2026-03-26 05:43 UTC - Exp039 Light CatBoost Refinement On Exp035 Core

- Hypothesis: A small CatBoost parameter sweep around the exp037 frontier may preserve the hepatic-event lift while recovering some death ranking, producing a better weighted C-index without changing the feature stack.
- Experiment dir: `scripts/exp039/`
- Data window: Training on `data/DB-1773398340961.csv`; inference exported against `data/test.csv` and the local hello-world submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Unchanged exp035 curated stack with 519 total features and 237 engineered additions.
- Model/config: CatBoost sweep over three lightweight configs: `exp037_baseline`, `shallower_regularized`, and `deeper_balanced`; selected config retrained on the full dataset for submission export.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.739127`.
- Per-target scores: `risk_hepatic_event cindex=0.821447, roc_auc=0.847041, average_precision=0.194208`; `risk_death cindex=0.547047, roc_auc=0.818253, average_precision=0.278830`.
- Fold scores: Saved to `scripts/exp039/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp039/outputs/candidate_sweep.json`, `scripts/exp039/outputs/validation_summary.json`, `scripts/exp039/outputs/metrics.json`, `scripts/exp039/outputs/fold_scores.csv`, `scripts/exp039/outputs/oof_predictions.csv`, `scripts/exp039/outputs/feature_importance.csv`, `scripts/exp039/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp039_catboost_refinement_on_exp035_core`, mode `offline`.
- Notes: The sweep confirmed that the original exp037 configuration remained best; neither the shallower regularized setting nor the deeper variant improved the weighted C-index. Even so, this run now provides a proper CatBoost submission artifact under the current best single-model CatBoost configuration.
- Next step: Try a focused two-model blend where LightGBM contributes only where it is still stronger, rather than averaging everything indiscriminately.

### 2026-03-26 05:47 UTC - Exp040 Focused Exp035 Plus Exp039 Blend New Frontier

- Hypothesis: A focused per-target blend between exp035 LightGBM and the CatBoost frontier can beat the single-model CatBoost by keeping LightGBM for death, where it remains stronger, and CatBoost for hepatic events, where it dominates.
- Experiment dir: `scripts/exp040/`
- Data window: Training on `data/DB-1773398340961.csv`; inference exported against `data/test.csv` and the local hello-world submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target, with blend selection performed over OOF predictions on a fixed weight grid.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Same exp035 curated stack with 519 total features, used by both LightGBM and CatBoost components before blending.
- Model/config: LightGBM from exp035 plus the best CatBoost config from exp039; weight grid search over CatBoost weights `0.00` to `1.00` in `0.05` steps for each target independently.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.752477`.
- Per-target scores: `risk_hepatic_event cindex=0.821447, roc_auc=0.847041, average_precision=0.194208`; `risk_death cindex=0.591549, roc_auc=0.803153, average_precision=0.366519`.
- Fold scores: Saved to `scripts/exp040/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp040/outputs/blend_search.json`, `scripts/exp040/outputs/validation_summary.json`, `scripts/exp040/outputs/metrics.json`, `scripts/exp040/outputs/fold_scores.csv`, `scripts/exp040/outputs/oof_predictions.csv`, `scripts/exp040/outputs/feature_importance.csv`, `scripts/exp040/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp040_exp035_exp039_focused_blend`, mode `offline`.
- Notes: The blend search found an extreme but highly effective solution: `risk_hepatic_event` wants CatBoost weight `1.00`, while `risk_death` wants CatBoost weight `0.00`. This is effectively a per-target model switch rather than a soft average, and it improved the official combined score by `+0.013350` over the single-model CatBoost frontier.
- Next step: Treat exp040 as the new primary submission candidate. If another pass is needed, search only around target-specific model choice or endpoint-specific model tuning rather than generic averaging.

### 2026-03-26 05:53 UTC - Exp041 Canonical Endpoint-Specific Ensemble Cleanup

- Hypothesis: The best behavior found in exp040 should be turned into a clean, explicit endpoint-specific pipeline so future work compares against a stable canonical ensemble instead of a blend-search artifact.
- Experiment dir: `scripts/exp041/`
- Data window: Training on `data/DB-1773398340961.csv`; inference exported against `data/test.csv` and the local hello-world submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Same exp035 curated stack with 519 total features.
- Model/config: Explicit endpoint map: `risk_hepatic_event -> CatBoostClassifier` with the exp037/exp039 frontier params; `risk_death -> LGBMClassifier` with the shared LightGBM baseline params.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.752477`.
- Per-target scores: `risk_hepatic_event cindex=0.821447, roc_auc=0.847041, average_precision=0.194208`; `risk_death cindex=0.591549, roc_auc=0.803153, average_precision=0.366519`.
- Fold scores: Saved to `scripts/exp041/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp041/outputs/validation_summary.json`, `scripts/exp041/outputs/metrics.json`, `scripts/exp041/outputs/fold_scores.csv`, `scripts/exp041/outputs/oof_predictions.csv`, `scripts/exp041/outputs/feature_importance.csv`, `scripts/exp041/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp041_canonical_endpoint_specific_ensemble`, mode `offline`.
- Notes: This run reproduced exp040 exactly, confirming that the new frontier is not coming from weight-search noise. The canonical interpretation is simple: hepatic is best handled by CatBoost, death is best handled by the current LightGBM baseline.
- Next step: Keep hepatic fixed and tune only the death-side LightGBM, since that is now the only movable part of the canonical ensemble.

### 2026-03-26 05:54 UTC - Exp042 Death-Side LightGBM Tuning On Canonical Ensemble

- Hypothesis: With hepatic effectively fixed by CatBoost, a small focused search around the current LightGBM death model may improve death ranking and lift the canonical endpoint-specific ensemble further.
- Experiment dir: `scripts/exp042/`
- Data window: Training on `data/DB-1773398340961.csv`; inference exported against `data/test.csv` and the local hello-world submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Same exp035 curated stack with 519 total features; hepatic side kept fixed.
- Model/config: Fixed hepatic `CatBoostClassifier` from the exp037 frontier; death-side `LGBMClassifier` sweep over four nearby configs: `lgbm_baseline`, `simpler_regularized`, `denser_leaves`, and `longer_small_step`.
- Validation metric: `cindex_weighted` using `0.3 * C-index_death + 0.7 * C-index_hepatic`.
- Combined score: `0.752477`.
- Per-target scores: `risk_hepatic_event cindex=0.821447, roc_auc=0.847041, average_precision=0.194208`; `risk_death cindex=0.591549, roc_auc=0.803153, average_precision=0.366519`.
- Fold scores: Saved to `scripts/exp042/outputs/fold_scores.csv`.
- Artifacts: `scripts/exp042/outputs/death_candidate_sweep.json`, `scripts/exp042/outputs/validation_summary.json`, `scripts/exp042/outputs/metrics.json`, `scripts/exp042/outputs/fold_scores.csv`, `scripts/exp042/outputs/oof_predictions.csv`, `scripts/exp042/outputs/feature_importance.csv`, `scripts/exp042/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp042_endpoint_specific_death_lgbm_tuning`, mode `offline`.
- Notes: None of the nearby LightGBM variants beat the baseline death config. The best candidate remained `lgbm_baseline`; `longer_small_step` came very close but still lost slightly. In this local neighborhood, the current death model is already at the top of the tested tradeoff.
- Next step: If more improvement is needed, shift death-side search to a different model family or a broader structural change rather than another tiny LightGBM hyperparameter tweak.

### 2026-03-26 05:00 UTC - Exp032 Separate vs Shared Backbone Rerun

- Hypothesis: Once the top-line score is aligned to the official weighted C-index, comparing a shared cumulative backbone against target-specific feature subsets may show whether endpoint-specific pruning helps more than a fully shared feature set.
- Experiment dir: `scripts/exp032/`
- Data window: Training on `data/DB-1773398340961.csv`; inference artifacts written against `data/test.csv` and the local submission template.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Full cumulative chain through exp031 with 710 total features; comparison was between the shared full feature set and target-specific subsets that removed the opposite endpoint-specific feature block.
- Model/config: Separate `LightGBMClassifier` models per endpoint; shared-vs-separate comparison performed inside one experiment run.
- Validation metric: `cindex_weighted`.
- Combined score: `0.727522`.
- Per-target scores: `risk_hepatic_event cindex=0.787907, roc_auc=0.837320, average_precision=0.202193`; `risk_death cindex=0.586622, roc_auc=0.800806, average_precision=0.365014`.
- Fold scores: `risk_hepatic_event cindex folds=[0.743068, 0.780571, 0.861199, 0.736931, 0.814747]`; `risk_death cindex folds=[0.512821, 0.623751, 0.538889, 0.496296, 0.686382]`.
- Artifacts: `scripts/exp032/outputs/validation_summary.json`, `scripts/exp032/outputs/metrics.json`, `scripts/exp032/outputs/fold_scores.csv`, `scripts/exp032/outputs/oof_predictions.csv`, `scripts/exp032/outputs/feature_importance.csv`, `scripts/exp032/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp032_separate_vs_shared_backbone`, mode `offline` because online credentials are still not configured.
- Notes: The rerun fixed the earlier non-comparable score path and also fixed a duplicate-column failure in the separate-backbone branch. The winning configuration was still `shared`, and although the run improved over exp031 by `+0.000545`, it remained below the aligned frontier from exp014.
- Next step: Check whether post-hoc calibration or OOF blending can add incremental gain under the official weighted C-index without breaking train-to-test consistency.

### 2026-03-26 05:01 UTC - Exp033 Isotonic Calibration Rerun

- Hypothesis: Even if calibration should not change pure ranking in theory, the current OOF pipeline may still show a small metric-aligned gain once the experiment reports the official weighted C-index rather than a custom ROC AUC aggregate.
- Experiment dir: `scripts/exp033/`
- Data window: Same local train/test setup as exp032.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` per target.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: Same cumulative feature backbone as exp032; no new engineered predictors, calibration-only comparison.
- Model/config: Shared LightGBM backbone plus post-hoc isotonic calibration diagnostics on OOF predictions.
- Validation metric: `cindex_weighted`.
- Combined score: `0.728650`.
- Per-target scores: `risk_hepatic_event cindex=0.789308, roc_auc=0.837973, average_precision=0.200070`; `risk_death cindex=0.587115, roc_auc=0.801835, average_precision=0.362395`.
- Fold scores: `risk_hepatic_event cindex folds=[0.751386, 0.779429, 0.861199, 0.736931, 0.814747]`; `risk_death cindex folds=[0.508547, 0.623751, 0.536111, 0.497531, 0.696011]`.
- Artifacts: `scripts/exp033/outputs/validation_summary.json`, `scripts/exp033/outputs/metrics.json`, `scripts/exp033/outputs/fold_scores.csv`, `scripts/exp033/outputs/oof_predictions.csv`, `scripts/exp033/outputs/feature_importance.csv`, `scripts/exp033/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp033_isotonic_calibration`, mode `offline`.
- Notes: The experiment is now metric-aligned. The official score improved over exp032 by `+0.001128`, but the gain was still not enough to overtake exp014. The optimistic in-sample isotonic diagnostic remained much higher than the official score and should not be used for model selection.
- Next step: Evaluate whether an OOF ensemble can add a real metric-aligned gain and preserve a clean inference path on the test set.

### 2026-03-26 05:01 UTC - Exp034 OOF Ensemble Blending Rerun

- Hypothesis: A blend chosen by the official weighted C-index, with matching blended test predictions, may recover small gains beyond the single-model frontier inside the late experiment chain.
- Experiment dir: `scripts/exp034/`
- Data window: Same local train/test files.
- Split strategy: OOF predictions loaded across prior experiments; blend choice evaluated against the official weighted C-index and test predictions blended with the same selected strategy per endpoint.
- Targets: `risk_hepatic_event`, `risk_death`.
- Feature set: No new raw features; ensemble search over prior experiment predictions plus a reference model run on the exp031 cumulative feature backbone.
- Model/config: OOF ensemble blending using simple average and rank-average candidates, compared against the model-only reference.
- Validation metric: `cindex_weighted`.
- Combined score: `0.728920`.
- Per-target scores: `risk_hepatic_event cindex=0.788141, roc_auc=0.836721, average_precision=0.200682`; `risk_death cindex=0.580765, roc_auc=0.799284, average_precision=0.364515`.
- Fold scores: `risk_hepatic_event cindex folds=[0.740296, 0.780571, 0.850683, 0.738617, 0.819355]`; `risk_death cindex folds=[0.478632, 0.623751, 0.538889, 0.491358, 0.687758]`.
- Artifacts: `scripts/exp034/outputs/blend_results.json`, `scripts/exp034/outputs/validation_summary.json`, `scripts/exp034/outputs/metrics.json`, `scripts/exp034/outputs/fold_scores.csv`, `scripts/exp034/outputs/oof_predictions.csv`, `scripts/exp034/outputs/submission/submission.csv`.
- W&B: project `annitia-trustii-2026`, run `exp034_ensemble_blending`, mode `offline`.
- Notes: The rerun aligned both blend selection and reported `combined_score` to the official weighted C-index and also made test predictions follow the chosen blend logic. It improved over exp033 by `+0.000270`, but still did not beat exp014 (`0.730723`), so exp014 remains the best trustworthy frontier in the repo.
- Next step: Start a new feature-engineering family or model class beyond the completed checklist if further improvement is required, rather than continuing to tune this exhausted cumulative chain.

### 2026-03-26 05:19 UTC - Exp035 Curated Accepted-Only Feature Stack

- Hypothesis: A curated stack that keeps only the earlier feature families with positive validation signal and drops the consistently harmful cumulative blocks can beat the exp014 frontier under the official weighted C-index.
- Experiment dir: `scripts/exp035/`
- Data window: Training on `data/DB-1773398340961.csv`; validation-only run, no submission generated.
- Split strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=7)` run separately per target.
- Targets: `risk_hepatic_event -> evenements_hepatiques_majeurs`, `risk_death -> death`.
- Feature set: Baseline raw non-leakage columns plus a curated accepted-only stack: `visit_summary`, `visit_minmax_mean_std`, `visit_recency_persistence`, `visit_missingness_trajectory`, `log_winsorized`, `inflammatory_burden`, `renal_cardiometabolic_burden`, and `ratio_features`. The run used 519 total features with 237 engineered additions.
- Model/config: Separate `LightGBMClassifier` models per target with the shared default config (`learning_rate=0.03`, `n_estimators=300`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `random_state=7`, `n_jobs=4`).
- Validation metric: `cindex_weighted` using the official formula `0.3 * C-index_death + 0.7 * C-index_hepatic` with `concordance_index_censored`.
- Combined score: `0.732377`.
- Per-target scores: `risk_hepatic_event cindex=0.792732, roc_auc=0.830916, average_precision=0.195018`; `risk_death cindex=0.591549, roc_auc=0.803153, average_precision=0.366519`.
- Fold scores: `risk_hepatic_event cindex folds=[0.792052, 0.764571, 0.844374, 0.739460, 0.830415]`; `risk_death cindex folds=[0.465812, 0.640400, 0.573611, 0.506173, 0.691884]`.
- Artifacts: `scripts/exp035/outputs/added_feature_names.txt`, `scripts/exp035/outputs/feature_columns.txt`, `scripts/exp035/outputs/feature_family_summary.json`, `scripts/exp035/outputs/validation_summary.json`, `scripts/exp035/outputs/metrics.json`, `scripts/exp035/outputs/fold_scores.csv`, `scripts/exp035/outputs/oof_predictions.csv`, `scripts/exp035/outputs/feature_importance.csv`.
- W&B: project `annitia-trustii-2026`, run `exp035_curated_accepted_feature_stack`, mode `offline`; API key was present in the environment but still invalid for online sync.
- Notes: A quick local screening of several curated stacks was used only to choose the best formal candidate before this run. The accepted-only core outperformed exp014 by `+0.001653`, making exp035 the new trustworthy frontier without requiring any additional post-hoc blending or special reporting logic.
- Next step: Run one cheap target-specific pruning pass on top of the exp035 core to see whether the hepatic-weighted objective can gain further by removing features that only help death ranking marginally.
