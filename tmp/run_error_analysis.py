from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

ROOT = Path('/home/azureuser/toanvt7/test_data/test_contest/annitia-trustii-2026')
TMP_DIR = ROOT / 'tmp'
OUTPUT_DIR = TMP_DIR / 'error_analysis_outputs'
RAW_PATH = ROOT / 'data' / 'DB-1773398340961.csv'
HEPATIC_OOF_PATH = ROOT / 'scripts' / 'exp039' / 'outputs' / 'oof_predictions.csv'
DEATH_OOF_PATH = ROOT / 'scripts' / 'exp042' / 'outputs' / 'oof_predictions.csv'
HEPATIC_IMPORTANCE_PATH = ROOT / 'scripts' / 'exp039' / 'outputs' / 'feature_importance.csv'
DEATH_IMPORTANCE_PATH = ROOT / 'scripts' / 'exp042' / 'outputs' / 'feature_importance.csv'
SUMMARY_PATH = OUTPUT_DIR / 'error_analysis_summary.md'

ID_COL = 'patient_id_anon'
TARGET_CONFIG = {
    'hepatic': {
        'target_col': 'evenements_hepatiques_majeurs',
        'score_col': 'risk_hepatic_event_oof_score',
        'event_age_col': 'evenements_hepatiques_age_occur',
        'oof_target_col': 'risk_hepatic_event_target',
        'importance_target': 'risk_hepatic_event',
        'title': 'Hepatic CatBoost',
    },
    'death': {
        'target_col': 'death',
        'score_col': 'risk_death_oof_score',
        'event_age_col': 'death_age_occur',
        'oof_target_col': 'risk_death_target',
        'importance_target': 'risk_death',
        'title': 'Death LightGBM',
    },
}
VISIT_BASES = [
    'Age',
    'BMI',
    'alt',
    'ast',
    'bilirubin',
    'chol',
    'ggt',
    'gluc_fast',
    'plt',
    'triglyc',
    'fibrotest_BM_2',
    'fibs_stiffness_med_BM_1',
]
SUBGROUP_COLUMNS = [
    'gender',
    'T2DM',
    'Hypertension',
    'Dyslipidaemia',
    'bariatric_surgery',
    'age_band',
    'bmi_band',
    'visit_coverage_band',
]
DISPLAY_COLUMNS = [
    ID_COL,
    'gender',
    'T2DM',
    'Hypertension',
    'Dyslipidaemia',
    'bariatric_surgery',
    'age_last_non_null',
    'BMI_last_non_null',
    'BMI_visit_mean',
    'alt_last_non_null',
    'ast_last_non_null',
    'ggt_last_non_null',
    'gluc_fast_last_non_null',
    'triglyc_last_non_null',
    'plt_last_non_null',
    'fibrotest_BM_2_last_non_null',
    'fibs_stiffness_med_BM_1_last_non_null',
    'fibs_stiffness_med_BM_1_visit_mean',
    'fibs_stiffness_med_BM_1_visit_min',
]
CURATED_FEATURE_FALLBACKS = {
    'hepatic': [
        'fibs_stiffness_med_BM_1_visit_mean',
        'fibs_stiffness_med_BM_1_last_non_null',
        'fibs_stiffness_med_BM_1_visit_min',
        'ast_visit_mean',
        'alt_visit_mean',
        'plt_visit_mean',
    ],
    'death': [
        'age_last_non_null',
        'BMI_visit_mean',
        'gluc_fast_visit_mean',
        'triglyc_visit_mean',
        'ggt_visit_mean',
        'plt_visit_mean',
    ],
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def visit_columns(columns: Iterable[str], base: str) -> list[str]:
    pattern = re.compile(rf'^{re.escape(base)}_v(\d+)$')
    matches: list[tuple[int, str]] = []
    for column in columns:
        matched = pattern.match(column)
        if matched:
            matches.append((int(matched.group(1)), column))
    return [column for _, column in sorted(matches)]



def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_csv(RAW_PATH)
    hepatic_oof = pd.read_csv(HEPATIC_OOF_PATH)
    death_oof = pd.read_csv(DEATH_OOF_PATH)
    hepatic_importance = pd.read_csv(HEPATIC_IMPORTANCE_PATH)
    death_importance = pd.read_csv(DEATH_IMPORTANCE_PATH)
    return raw_df, hepatic_oof, death_oof, hepatic_importance, death_importance



def validate_inputs(raw_df: pd.DataFrame, hepatic_oof: pd.DataFrame, death_oof: pd.DataFrame) -> None:
    if raw_df[ID_COL].duplicated().any():
        raise ValueError('patient_id_anon is not unique in raw training data.')
    for frame, name in ((hepatic_oof, 'hepatic_oof'), (death_oof, 'death_oof')):
        if frame[ID_COL].duplicated().any():
            raise ValueError(f'{name} contains duplicated patient IDs.')



def build_master_table(raw_df: pd.DataFrame, hepatic_oof: pd.DataFrame, death_oof: pd.DataFrame) -> pd.DataFrame:
    hepatic_subset = hepatic_oof[[
        ID_COL,
        'risk_hepatic_event_actual',
        'risk_hepatic_event_target',
        'risk_hepatic_event_oof_score',
    ]].rename(columns={
        'risk_hepatic_event_actual': 'hepatic_actual_from_oof',
        'risk_hepatic_event_target': 'hepatic_target_from_oof',
    })
    death_subset = death_oof[[
        ID_COL,
        'risk_death_actual',
        'risk_death_target',
        'risk_death_oof_score',
    ]].rename(columns={
        'risk_death_actual': 'death_actual_from_oof',
        'risk_death_target': 'death_target_from_oof',
    })
    merged = raw_df.merge(hepatic_subset, on=ID_COL, how='left').merge(death_subset, on=ID_COL, how='left')
    if len(merged) != len(raw_df):
        raise ValueError('Merged master table changed row count.')
    return merged



def add_visit_summaries(master_df: pd.DataFrame, base: str) -> None:
    cols = visit_columns(master_df.columns, base)
    if not cols:
        return
    values = master_df[cols].apply(pd.to_numeric, errors='coerce')
    master_df[f'{base}_visit_obs_count'] = values.notna().sum(axis=1)
    master_df[f'{base}_visit_missing_fraction'] = 1.0 - (master_df[f'{base}_visit_obs_count'] / len(cols))
    master_df[f'{base}_first_non_null'] = values.bfill(axis=1).iloc[:, 0]
    master_df[f'{base}_last_non_null'] = values.ffill(axis=1).iloc[:, -1]
    master_df[f'{base}_visit_min'] = values.min(axis=1)
    master_df[f'{base}_visit_max'] = values.max(axis=1)
    master_df[f'{base}_visit_mean'] = values.mean(axis=1)
    master_df[f'{base}_visit_std'] = values.std(axis=1)
    master_df[f'{base}_visit_delta'] = master_df[f'{base}_last_non_null'] - master_df[f'{base}_first_non_null']



def add_engineered_context(master_df: pd.DataFrame) -> pd.DataFrame:
    for base in VISIT_BASES:
        add_visit_summaries(master_df, base)
    master_df['last_observed_age'] = master_df['Age_last_non_null']
    master_df['age_band'] = pd.cut(
        master_df['Age_last_non_null'],
        bins=[0, 45, 55, 65, 200],
        labels=['<=45', '46-55', '56-65', '66+'],
        include_lowest=True,
    ).astype(str)
    master_df['bmi_band'] = pd.cut(
        master_df['BMI_visit_mean'],
        bins=[0, 25, 30, 35, 100],
        labels=['normal_or_less', 'overweight', 'obesity_I', 'obesity_II_plus'],
        include_lowest=True,
    ).astype(str)
    coverage = master_df['Age_visit_obs_count'].fillna(0)
    master_df['visit_coverage_band'] = pd.cut(
        coverage,
        bins=[-1, 6, 12, 100],
        labels=['sparse', 'medium', 'dense'],
        include_lowest=True,
    ).astype(str)
    return master_df



def compute_youden_threshold(df: pd.DataFrame, target_col: str, score_col: str) -> float:
    fpr, tpr, thresholds = roc_curve(df[target_col].astype(int), df[score_col].astype(float))
    best_index = int(np.argmax(tpr - fpr))
    return float(thresholds[best_index])



def assign_error_buckets(df: pd.DataFrame, target_col: str, score_col: str) -> pd.DataFrame:
    positive_scores = df.loc[df[target_col] == 1, score_col].astype(float)
    negative_scores = df.loc[df[target_col] == 0, score_col].astype(float)
    positive_low = float(positive_scores.quantile(0.25)) if len(positive_scores) else 0.0
    positive_high = float(positive_scores.quantile(0.75)) if len(positive_scores) else 1.0
    negative_low = float(negative_scores.quantile(0.25)) if len(negative_scores) else 0.0
    negative_high = float(negative_scores.quantile(0.90)) if len(negative_scores) else 1.0

    df = df.copy()
    df['score_rank_pct'] = df[score_col].rank(method='average', pct=True)
    df['score_decile'] = pd.qcut(
        df[score_col].rank(method='first'),
        q=10,
        labels=list(range(1, 11)),
        duplicates='drop',
    ).astype(int)
    df['error_bucket'] = 'ambiguous'
    df.loc[(df[target_col] == 1) & (df[score_col] <= positive_low), 'error_bucket'] = 'hard_fn'
    df.loc[(df[target_col] == 0) & (df[score_col] >= negative_high), 'error_bucket'] = 'hard_fp'
    df.loc[(df[target_col] == 1) & (df[score_col] >= positive_high), 'error_bucket'] = 'well_ranked_positive'
    df.loc[(df[target_col] == 0) & (df[score_col] <= negative_low), 'error_bucket'] = 'well_ranked_negative'
    df['hard_fn_flag'] = ((df[target_col] == 1) & (df['error_bucket'] == 'hard_fn')).astype(int)
    df['hard_fp_flag'] = ((df[target_col] == 0) & (df['error_bucket'] == 'hard_fp')).astype(int)
    df['youden_threshold'] = compute_youden_threshold(df, target_col, score_col)
    df['pred_label_youden'] = (df[score_col] >= df['youden_threshold']).astype(int)
    return df



def build_decile_summary(df: pd.DataFrame, target_col: str, score_col: str, event_age_col: str) -> pd.DataFrame:
    decile_df = (
        df.groupby('score_decile', dropna=False)
        .agg(
            row_count=(ID_COL, 'count'),
            positive_count=(target_col, 'sum'),
            event_rate=(target_col, 'mean'),
            mean_score=(score_col, 'mean'),
            median_score=(score_col, 'median'),
            mean_event_age=(event_age_col, 'mean'),
            median_event_age=(event_age_col, 'median'),
        )
        .reset_index()
        .sort_values('score_decile', ascending=False)
    )
    return decile_df



def subgroup_summary(df: pd.DataFrame, target_col: str, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subgroup in SUBGROUP_COLUMNS:
        if subgroup not in df.columns:
            continue
        grouped = df.groupby(subgroup, dropna=False)
        for level, group in grouped:
            positive_mask = group[target_col] == 1
            negative_mask = group[target_col] == 0
            positive_count = int(positive_mask.sum())
            negative_count = int(negative_mask.sum())
            hard_fn_count = int(((group['error_bucket'] == 'hard_fn') & positive_mask).sum())
            hard_fp_count = int(((group['error_bucket'] == 'hard_fp') & negative_mask).sum())
            rows.append(
                {
                    'subgroup': subgroup,
                    'level': str(level),
                    'row_count': int(len(group)),
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'event_rate': float(group[target_col].mean()),
                    'mean_score': float(group[score_col].mean()),
                    'hard_fn_count': hard_fn_count,
                    'hard_fp_count': hard_fp_count,
                    'hard_fn_rate': float(hard_fn_count / positive_count) if positive_count else np.nan,
                    'hard_fp_rate': float(hard_fp_count / negative_count) if negative_count else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(['subgroup', 'row_count'], ascending=[True, False])



def top_importance_features(importance_df: pd.DataFrame, importance_target: str, master_df: pd.DataFrame, analysis_key: str) -> list[str]:
    filtered = importance_df.loc[importance_df['target'] == importance_target].copy()
    filtered = filtered.sort_values('importance', ascending=False)
    features = [feature for feature in filtered['feature'].tolist() if feature in master_df.columns]
    fallback = [feature for feature in CURATED_FEATURE_FALLBACKS[analysis_key] if feature in master_df.columns]
    ordered: list[str] = []
    for feature in features + fallback:
        if feature not in ordered:
            ordered.append(feature)
    return ordered[:6]



def build_case_table(df: pd.DataFrame, bucket: str, target_col: str, score_col: str, event_age_col: str, top_n: int = 30) -> pd.DataFrame:
    case_df = df.loc[df['error_bucket'] == bucket].copy()
    ascending = bucket == 'hard_fn'
    case_df = case_df.sort_values(score_col, ascending=ascending).head(top_n)
    columns = [
        ID_COL,
        target_col,
        score_col,
        'score_rank_pct',
        event_age_col,
        'age_band',
        'bmi_band',
        'visit_coverage_band',
        *[column for column in DISPLAY_COLUMNS if column in case_df.columns and column != ID_COL],
    ]
    return case_df.loc[:, list(dict.fromkeys(columns))]



def build_feature_bucket_summary(df: pd.DataFrame, analysis_key: str, features: list[str]) -> pd.DataFrame:
    compare_df = df.loc[df['error_bucket'].isin(['hard_fn', 'hard_fp', 'well_ranked_positive', 'well_ranked_negative'])].copy()
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in compare_df.columns:
            continue
        numeric = pd.to_numeric(compare_df[feature], errors='coerce')
        compare_df[feature] = numeric
        for bucket, group in compare_df.groupby('error_bucket'):
            series = pd.to_numeric(group[feature], errors='coerce')
            rows.append(
                {
                    'analysis_target': analysis_key,
                    'feature': feature,
                    'error_bucket': bucket,
                    'count': int(series.notna().sum()),
                    'mean': float(series.mean()) if series.notna().any() else np.nan,
                    'median': float(series.median()) if series.notna().any() else np.nan,
                    'std': float(series.std()) if series.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)



def write_top_importance_table(importance_df: pd.DataFrame, importance_target: str, out_path: Path) -> pd.DataFrame:
    filtered = importance_df.loc[importance_df['target'] == importance_target].copy().sort_values('importance', ascending=False).head(20)
    filtered.to_csv(out_path, index=False)
    return filtered



def plot_score_distribution(df: pd.DataFrame, score_col: str, target_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plot_df = df[[score_col, target_col]].copy()
    plot_df[target_col] = plot_df[target_col].map({0: 'non_event', 1: 'event'})
    sns.histplot(data=plot_df, x=score_col, hue=target_col, bins=30, stat='density', common_norm=False, element='step')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_decile_lift(decile_df: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=decile_df, x='score_decile', y='event_rate', color='#3a6ea5')
    plt.title(title)
    plt.ylabel('Event rate')
    plt.xlabel('Score decile (10 = highest risk)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_feature_boxplots(df: pd.DataFrame, features: list[str], score_col: str, title: str, out_path: Path) -> None:
    selected_buckets = ['hard_fn', 'hard_fp', 'well_ranked_positive', 'well_ranked_negative']
    plot_df = df.loc[df['error_bucket'].isin(selected_buckets), ['error_bucket', *features]].copy()
    if plot_df.empty or not features:
        return
    melted = plot_df.melt(id_vars='error_bucket', value_vars=features, var_name='feature', value_name='value').dropna()
    if melted.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='feature', y='value', hue='error_bucket')
    plt.xticks(rotation=30, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def plot_subgroup_error_rates(subgroup_df: pd.DataFrame, title: str, out_path: Path) -> None:
    filtered = subgroup_df.loc[subgroup_df['row_count'] >= 20].copy()
    if filtered.empty:
        return
    filtered['max_error_rate'] = filtered[['hard_fn_rate', 'hard_fp_rate']].max(axis=1)
    top = filtered.sort_values('max_error_rate', ascending=False).head(12).copy()
    top['subgroup_level'] = top['subgroup'] + '=' + top['level']
    melted = top.melt(id_vars='subgroup_level', value_vars=['hard_fn_rate', 'hard_fp_rate'], var_name='metric', value_name='rate')
    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted, x='subgroup_level', y='rate', hue='metric')
    plt.xticks(rotation=35, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def summarize_feature_differences(feature_bucket_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if feature_bucket_df.empty:
        return lines
    pivot = feature_bucket_df.pivot_table(index='feature', columns='error_bucket', values='median', aggfunc='first')
    if {'hard_fn', 'well_ranked_positive'}.issubset(pivot.columns):
        fn_diff = (pivot['hard_fn'] - pivot['well_ranked_positive']).abs().sort_values(ascending=False).head(3)
        for feature, diff in fn_diff.items():
            lines.append(f'- Hard FN vs well-ranked positive differs most on `{feature}` (|median delta|={diff:.4f}).')
    if {'hard_fp', 'well_ranked_negative'}.issubset(pivot.columns):
        fp_diff = (pivot['hard_fp'] - pivot['well_ranked_negative']).abs().sort_values(ascending=False).head(3)
        for feature, diff in fp_diff.items():
            lines.append(f'- Hard FP vs well-ranked negative differs most on `{feature}` (|median delta|={diff:.4f}).')
    return lines



def top_subgroup_observations(subgroup_df: pd.DataFrame) -> list[str]:
    observations: list[str] = []
    filtered = subgroup_df.loc[subgroup_df['row_count'] >= 20].copy()
    if filtered.empty:
        return observations
    fn_rows = (
        filtered.loc[filtered['positive_count'] >= 3]
        .dropna(subset=['hard_fn_rate'])
        .sort_values(['hard_fn_rate', 'positive_count'], ascending=[False, False])
        .head(2)
    )
    fp_rows = (
        filtered.loc[filtered['negative_count'] >= 20]
        .dropna(subset=['hard_fp_rate'])
        .sort_values(['hard_fp_rate', 'negative_count'], ascending=[False, False])
        .head(2)
    )
    for _, row in fn_rows.iterrows():
        observations.append(
            f"- Highest hard-FN subgroup: {row['subgroup']}={row['level']} with hard_fn_rate={row['hard_fn_rate']:.3f} over {int(row['positive_count'])} positives."
        )
    for _, row in fp_rows.iterrows():
        observations.append(
            f"- Highest hard-FP subgroup: {row['subgroup']}={row['level']} with hard_fp_rate={row['hard_fp_rate']:.3f} over {int(row['negative_count'])} negatives."
        )
    return observations



def build_target_summary_section(
    analysis_key: str,
    config: dict[str, str],
    df: pd.DataFrame,
    decile_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    feature_bucket_df: pd.DataFrame,
) -> list[str]:
    lines = [f"## {config['title']}"]
    lines.append(f"- Rows analyzed: {len(df)}")
    lines.append(f"- Positive count: {int(df[config['target_col']].sum())}")
    lines.append(f"- Mean OOF score: {df[config['score_col']].mean():.6f}")
    lines.append(f"- Youden threshold: {df['youden_threshold'].iloc[0]:.6f}")
    top_decile = decile_df.iloc[0]
    bottom_decile = decile_df.iloc[-1]
    lines.append(
        f"- Top decile event rate: {top_decile['event_rate']:.3f}; bottom decile event rate: {bottom_decile['event_rate']:.3f}."
    )
    lines.extend(top_subgroup_observations(subgroup_df))
    lines.extend(summarize_feature_differences(feature_bucket_df))
    return lines



def write_summary(master_df: pd.DataFrame, per_target_outputs: dict[str, dict[str, object]]) -> None:
    lines = ['# Error Analysis Summary', '']
    lines.append('## Dataset checks')
    lines.append(f"- Master table rows: {len(master_df)}")
    lines.append(f"- Unique patient IDs: {master_df[ID_COL].nunique()}")
    lines.append(f"- Hepatic positives: {int(master_df['evenements_hepatiques_majeurs'].fillna(0).sum())}")
    lines.append(f"- Death positives (non-null labels only): {int(master_df['death'].fillna(0).sum())}")
    lines.append('')
    shared_groups: set[str] = set()
    subgroup_name_sets: list[set[str]] = []
    for target_key, payload in per_target_outputs.items():
        config = TARGET_CONFIG[target_key]
        lines.extend(build_target_summary_section(target_key, config, payload['analysis_df'], payload['decile_df'], payload['subgroup_df'], payload['feature_bucket_df']))
        lines.append('')
        subgroup_df = payload['subgroup_df']
        filtered = subgroup_df.loc[subgroup_df['row_count'] >= 20].copy()
        top_groups = set(filtered.sort_values('hard_fn_rate', ascending=False)['subgroup'].head(3).tolist() + filtered.sort_values('hard_fp_rate', ascending=False)['subgroup'].head(3).tolist())
        subgroup_name_sets.append(top_groups)
    if subgroup_name_sets:
        shared_groups = set.intersection(*subgroup_name_sets) if len(subgroup_name_sets) > 1 else subgroup_name_sets[0]
    lines.append('## Cross-target common signals')
    if shared_groups:
        lines.append(f"- Shared subgroup axes appearing near the top of both error analyses: {', '.join(sorted(shared_groups))}.")
    else:
        lines.append('- No single subgroup axis dominated both targets in the first-pass summaries.')
    lines.append('- Hepatic analysis should be read mainly through fibrosis/stiffness and liver-enzyme context.')
    lines.append('- Death analysis should be read mainly through age/follow-up burden and cardiometabolic context.')
    SUMMARY_PATH.write_text('\n'.join(lines) + '\n')



def run_target_analysis(
    master_df: pd.DataFrame,
    analysis_key: str,
    importance_df: pd.DataFrame,
) -> dict[str, object]:
    config = TARGET_CONFIG[analysis_key]
    target_col = config['target_col']
    score_col = config['score_col']
    event_age_col = config['event_age_col']
    importance_target = config['importance_target']

    analysis_df = master_df.loc[master_df[target_col].notna() & master_df[score_col].notna()].copy()
    analysis_df[target_col] = pd.to_numeric(analysis_df[target_col], errors='coerce').fillna(0).astype(int)
    analysis_df[score_col] = pd.to_numeric(analysis_df[score_col], errors='coerce')
    analysis_df[event_age_col] = pd.to_numeric(analysis_df[event_age_col], errors='coerce')
    analysis_df = assign_error_buckets(analysis_df, target_col, score_col)

    decile_df = build_decile_summary(analysis_df, target_col, score_col, event_age_col)
    subgroup_df = subgroup_summary(analysis_df, target_col, score_col)
    top_features = top_importance_features(importance_df, importance_target, analysis_df, analysis_key)
    feature_bucket_df = build_feature_bucket_summary(analysis_df, analysis_key, top_features)

    hard_fn_df = build_case_table(analysis_df, 'hard_fn', target_col, score_col, event_age_col)
    hard_fp_df = build_case_table(analysis_df, 'hard_fp', target_col, score_col, event_age_col)

    decile_df.to_csv(OUTPUT_DIR / f'{analysis_key}_decile_summary.csv', index=False)
    subgroup_df.to_csv(OUTPUT_DIR / f'{analysis_key}_subgroup_summary.csv', index=False)
    hard_fn_df.to_csv(OUTPUT_DIR / f'{analysis_key}_hard_false_negatives.csv', index=False)
    hard_fp_df.to_csv(OUTPUT_DIR / f'{analysis_key}_hard_false_positives.csv', index=False)
    feature_bucket_df.to_csv(OUTPUT_DIR / f'{analysis_key}_feature_bucket_summary.csv', index=False)
    write_top_importance_table(importance_df, importance_target, OUTPUT_DIR / f'{analysis_key}_top_feature_importance.csv')

    plot_score_distribution(
        analysis_df,
        score_col=score_col,
        target_col=target_col,
        title=f"{config['title']} score distribution by outcome",
        out_path=OUTPUT_DIR / f'{analysis_key}_score_distribution.png',
    )
    plot_decile_lift(
        decile_df,
        title=f"{config['title']} event rate by score decile",
        out_path=OUTPUT_DIR / f'{analysis_key}_decile_lift.png',
    )
    plot_feature_boxplots(
        analysis_df,
        features=top_features[:4],
        score_col=score_col,
        title=f"{config['title']} key features by error bucket",
        out_path=OUTPUT_DIR / f'{analysis_key}_key_feature_boxplots.png',
    )
    plot_subgroup_error_rates(
        subgroup_df,
        title=f"{config['title']} subgroup hard-error rates",
        out_path=OUTPUT_DIR / f'{analysis_key}_subgroup_error_rates.png',
    )

    return {
        'analysis_df': analysis_df,
        'decile_df': decile_df,
        'subgroup_df': subgroup_df,
        'feature_bucket_df': feature_bucket_df,
    }



def main() -> None:
    ensure_output_dir()
    sns.set_theme(style='whitegrid')

    raw_df, hepatic_oof, death_oof, hepatic_importance, death_importance = load_inputs()
    validate_inputs(raw_df, hepatic_oof, death_oof)

    master_df = build_master_table(raw_df, hepatic_oof, death_oof)
    master_df = add_engineered_context(master_df)

    if not np.allclose(
        master_df['evenements_hepatiques_majeurs'].fillna(-1).astype(float),
        master_df['hepatic_target_from_oof'].fillna(-1).astype(float),
        equal_nan=True,
    ):
        raise ValueError('Hepatic raw target and OOF target are misaligned.')

    death_comparable = master_df.loc[master_df['death'].notna(), ['death', 'death_target_from_oof']].copy()
    if not np.allclose(
        death_comparable['death'].astype(float),
        death_comparable['death_target_from_oof'].astype(float),
        equal_nan=True,
    ):
        raise ValueError('Death raw target and OOF target are misaligned.')

    master_df.to_csv(OUTPUT_DIR / 'master_error_analysis_table.csv', index=False)

    per_target_outputs = {
        'hepatic': run_target_analysis(master_df, 'hepatic', hepatic_importance),
        'death': run_target_analysis(master_df, 'death', death_importance),
    }
    write_summary(master_df, per_target_outputs)

    manifest = {
        'master_table': str((OUTPUT_DIR / 'master_error_analysis_table.csv').relative_to(ROOT)),
        'summary': str(SUMMARY_PATH.relative_to(ROOT)),
        'targets': {
            analysis_key: {
                'decile_summary': str((OUTPUT_DIR / f'{analysis_key}_decile_summary.csv').relative_to(ROOT)),
                'subgroup_summary': str((OUTPUT_DIR / f'{analysis_key}_subgroup_summary.csv').relative_to(ROOT)),
                'hard_false_negatives': str((OUTPUT_DIR / f'{analysis_key}_hard_false_negatives.csv').relative_to(ROOT)),
                'hard_false_positives': str((OUTPUT_DIR / f'{analysis_key}_hard_false_positives.csv').relative_to(ROOT)),
            }
            for analysis_key in TARGET_CONFIG
        },
    }
    (OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('error analysis complete')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
