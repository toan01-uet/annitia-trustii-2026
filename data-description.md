# ANNITIA Data Challenge Notes

## 1. Problem Overview

The current project focuses on risk stratification for MASLD, or metabolic dysfunction-associated steatotic liver disease.

The practical goal is to predict patient-level risk scores for severe outcomes so the model can support clinical monitoring and prioritization.

The expected outputs are:

- `risk_hepatic_event`
- `risk_death`

This is better framed as a tabular risk-scoring problem than as a plain binary classification task. The model should rank patients by severity and future risk in a useful way.

## 2. Local Files In This Repo

The repository currently contains:

- `data/DB-1773398340961.csv`
- `data/dictionary-1773398867610.csv`
- `data/hello_world_submission-1773575379610.csv`

### Important schema note

There is a mismatch between the semantic dictionary and the current raw training file:

- the dictionary file uses clean challenge-style column names such as `subject_id`, `age_years`, `sex`, `bmi_kg_m2`, `smoking_status`, `risk_hepatic_event`, and `risk_death`
- the raw training file currently visible in `data/` is a wide table with many columns such as `Age_v1`, `BMI_v1`, `alt_v1`, `ast_v1`, `bilirubin_v1`, `fibrotest_BM_2_v1`, `fibs_stiffness_med_BM_1_v1`, plus outcomes such as `evenements_hepatiques_majeurs`, `death`, and event-age columns

For that reason, future work must start by mapping the raw physical schema to the semantic challenge concepts. Do not assume the dictionary names can be used directly against the training CSV.

## 3. How To View The Data

At modeling time, the dataset should be treated as a wide patient table:

- one patient corresponds to one row
- repeated measurements, if present, are encoded as additional columns
- feature engineering should turn those columns into meaningful tabular predictors

This means the project should not default to a time-series or sequence-modeling setup. If repeated visit columns exist, the default move is to derive summary features inside each row.

Examples of useful within-row summaries when `_v1`, `_v2`, and similar columns exist:

- first observed value
- last observed value
- delta between last and first
- mean, min, max, standard deviation
- count of observed visits
- simple slope or trend proxy
- missingness persistence across visits

## 4. Conceptual Variable Groups

The challenge description and the semantic dictionary suggest the following clinical groups:

### Background and demographics

- subject identifier
- age
- sex
- smoking status

### Body composition and hemodynamics

- BMI
- systolic blood pressure
- diastolic blood pressure
- heart rate

### Metabolic comorbidity

- diabetes
- hypertension
- HbA1c
- LDL-C
- HDL-C

### Renal and inflammatory state

- eGFR
- CRP
- IL-6
- TNF-alpha

### Cardiac stress markers

- troponin I
- BNP

### Outcome targets

- `risk_hepatic_event`
- `risk_death`

There may also be auxiliary or proxy outcome columns in the raw data. These must be handled carefully and never used as predictors without checking the challenge rules.

## 5. Modeling Interpretation

The most useful framing for this repo is:

- multi-target tabular risk scoring
- shared clinical signal with endpoint-specific differences
- strong emphasis on validation and feature engineering

Common feature blocks can be reused across both targets, but each endpoint may benefit from its own specialized features or even its own model.

## 6. Feature-Engineering Direction

Priority feature families for this project:

### 6.1 Clinical burden features

These describe how broadly unwell the patient is.

Examples:

- `n_comorbidities`
- `metabolic_burden`
- `inflammatory_burden`
- `cardio_burden`
- `renal_cardiometabolic_burden`

### 6.2 Ratio and interaction features

These capture non-additive clinical risk.

Examples:

- `ldl_hdl_ratio`
- `pulse_pressure`
- `map_approx`
- `bmi_hba1c`
- `age_bmi`
- `age_egfr`
- `diabetes_hba1c`
- `inflammation_x_obesity`
- `il6_tnf`
- `cardiac_stress`

### 6.3 Threshold and binary flags

Clinical thresholds often matter more than tiny linear changes.

Examples:

- `age_ge_50`, `age_ge_60`, `age_ge_70`
- `bmi_overweight`, `bmi_obese`
- `egfr_lt_90`, `egfr_lt_60`, `egfr_lt_45`
- high-value flags for HbA1c, CRP, IL-6, BNP, troponin

Thresholds may come from:

- clinical cutoffs
- data-driven train-set quantiles

### 6.4 Log and nonlinear transforms

Many biomarkers are right-skewed and should be transformed before modeling.

Good candidates include:

- CRP
- IL-6
- TNF-alpha
- troponin I
- BNP

Examples:

- `log_crp`
- `log_il6`
- `log_tnf`
- `log_troponin`
- `log_bnp`

Also consider winsorization, rank transforms, or percentile transforms when outliers are influential.

### 6.5 Missingness features

Missing data can itself contain signal in clinical datasets.

Good defaults:

- `is_missing_<feature>`
- `num_missing`
- `missing_inflammatory_panel`
- `missing_cardiac_panel`

### 6.6 Group-relative features

A raw value may have different meaning across sex or age groups.

Potential features:

- sex-specific z-scores
- age-band percentiles
- within-group ranks

## 7. Endpoint-Specific Thinking

### For `risk_hepatic_event`

Prioritize features connected to:

- age
- BMI
- diabetes
- glycemic burden
- inflammation
- renal function
- liver injury and fibrosis markers if present in the raw file

### For `risk_death`

Prioritize features connected to:

- age
- blood pressure
- heart rate
- renal function
- cardiac biomarkers
- systemic inflammation
- overall comorbidity burden

## 8. Practical Rule For This Repo

When the semantic challenge description and the raw training file disagree on column names or layout:

1. trust the raw file for executable schema
2. trust the dictionary for semantic interpretation
3. document the mapping explicitly before feature engineering
