# Error Analysis Summary

## Dataset checks
- Master table rows: 1253
- Unique patient IDs: 1253
- Hepatic positives: 47
- Death positives (non-null labels only): 76

## Hepatic CatBoost
- Rows analyzed: 1253
- Positive count: 47
- Mean OOF score: 0.018881
- Youden threshold: 0.013084
- Top decile event rate: 0.183; bottom decile event rate: 0.008.
- Highest hard-FN subgroup: visit_coverage_band=dense with hard_fn_rate=0.667 over 3 positives.
- Highest hard-FN subgroup: bariatric_surgery=1.0 with hard_fn_rate=0.500 over 4 positives.
- Highest hard-FP subgroup: visit_coverage_band=dense with hard_fp_rate=0.315 over 54 negatives.
- Highest hard-FP subgroup: T2DM=1.0 with hard_fp_rate=0.196 over 342 negatives.
- Hard FN vs well-ranked positive differs most on `fibs_stiffness_med_BM_1_first_non_null` (|median delta|=16.2552).
- Hard FN vs well-ranked positive differs most on `fibs_stiffness_med_BM_1_visit_mean` (|median delta|=9.9698).
- Hard FN vs well-ranked positive differs most on `fibs_stiffness_med_BM_1_visit_min` (|median delta|=5.0502).
- Hard FP vs well-ranked negative differs most on `Age_visit_mean` (|median delta|=7.0000).
- Hard FP vs well-ranked negative differs most on `fibs_stiffness_med_BM_1_first_non_null` (|median delta|=6.0379).
- Hard FP vs well-ranked negative differs most on `fibs_stiffness_med_BM_1_visit_mean` (|median delta|=6.0188).

## Death LightGBM
- Rows analyzed: 984
- Positive count: 76
- Mean OOF score: 0.028433
- Youden threshold: 0.013041
- Top decile event rate: 0.333; bottom decile event rate: 0.000.
- Highest hard-FN subgroup: visit_coverage_band=dense with hard_fn_rate=0.750 over 4 positives.
- Highest hard-FN subgroup: age_band=46-55 with hard_fn_rate=0.714 over 7 positives.
- Highest hard-FP subgroup: age_band=66+ with hard_fp_rate=0.205 over 327 negatives.
- Highest hard-FP subgroup: Dyslipidaemia=1.0 with hard_fp_rate=0.160 over 332 negatives.
- Hard FN vs well-ranked positive differs most on `Age_v2` (|median delta|=9.5000).
- Hard FN vs well-ranked positive differs most on `bilirubin_v3` (|median delta|=0.5190).
- Hard FN vs well-ranked positive differs most on `gluc_fast_v3` (|median delta|=0.4216).
- Hard FP vs well-ranked negative differs most on `Age_v2` (|median delta|=13.0000).
- Hard FP vs well-ranked negative differs most on `aixp_aix_result_BM_3_v1` (|median delta|=3.2128).
- Hard FP vs well-ranked negative differs most on `gluc_fast_v3` (|median delta|=0.9228).

## Cross-target common signals
- Shared subgroup axes appearing near the top of both error analyses: Hypertension, visit_coverage_band.
- Hepatic analysis should be read mainly through fibrosis/stiffness and liver-enzyme context.
- Death analysis should be read mainly through age/follow-up burden and cardiometabolic context.
