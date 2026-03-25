from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
import wandb
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "DB-1773398340961.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
DICTIONARY_PATH = DATA_DIR / "dictionary-1773398867610.csv"
SUBMISSION_TEMPLATE_PATH = DATA_DIR / "hello_world_submission-1773575379610.csv"
ENV_PATH = PROJECT_ROOT / ".env"

ID_COLUMN = "patient_id_anon"
SUBMISSION_ID_COLUMN = "trustii_id"
TARGET_COLUMN_MAP = {
    "risk_hepatic_event": "evenements_hepatiques_majeurs",
    "risk_death": "death",
}
TARGET_EVENT_AGE_COLUMNS = {
    "risk_hepatic_event": "evenements_hepatiques_age_occur",
    "risk_death": "death_age_occur",
}
LEAKAGE_COLUMNS = set(TARGET_COLUMN_MAP.values()) | set(TARGET_EVENT_AGE_COLUMNS.values())
NON_FEATURE_COLUMNS = {ID_COLUMN, SUBMISSION_ID_COLUMN} | LEAKAGE_COLUMNS
VISIT_PATTERN = re.compile(r"^(?P<base>.+)_v(?P<visit>\d+)$")
PRIMARY_VALIDATION_METRIC = "roc_auc_surrogate"
DEFAULT_WANDB_PROJECT = "annitia-trustii-2026"
DEFAULT_RANDOM_STATE = 7
DEFAULT_FOLDS = 5
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 300,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": 4,
    "verbosity": -1,
}


def now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def load_env_file() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def configure_wandb_env() -> None:
    parsed_env = load_env_file()
    for key, value in parsed_env.items():
        os.environ.setdefault(key, value)

    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]


def wandb_api_key_present() -> bool:
    configure_wandb_env()
    return bool(os.environ.get("WANDB_API_KEY"))


def wandb_api_key_valid() -> bool:
    configure_wandb_env()
    api_key = os.environ.get("WANDB_API_KEY", "")
    return len(api_key) >= 40


def load_raw_data() -> pl.DataFrame:
    return pl.read_csv(RAW_DATA_PATH)


def load_test_data() -> pl.DataFrame:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing requested inference file at {TEST_DATA_PATH}")
    return pl.read_csv(TEST_DATA_PATH)


def load_dictionary_preview() -> pl.DataFrame:
    return pl.read_csv(DICTIONARY_PATH, encoding="latin1")


def visit_column_groups(columns: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for column_name in columns:
        matched = VISIT_PATTERN.match(column_name)
        if matched is None:
            continue
        grouped.setdefault(matched.group("base"), []).append((int(matched.group("visit")), column_name))
    return {
        base_name: [column_name for _, column_name in sorted(pairs)]
        for base_name, pairs in sorted(grouped.items())
    }


def raw_to_semantic_map() -> dict[str, Any]:
    return {
        "patient_id_anon": "subject_id",
        "trustii_id": "submission row identifier",
        "gender": "sex",
        "T2DM": "diabetes",
        "Hypertension": "hypertension",
        "Dyslipidaemia": "dyslipidemia",
        "bariatric_surgery": "bariatric_surgery",
        "bariatric_surgery_age": "bariatric_surgery_age",
        "Age_v*": "age_years longitudinal visits",
        "BMI_v*": "bmi_kg_m2 longitudinal visits",
        "alt_v*": "ALT longitudinal visits",
        "ast_v*": "AST longitudinal visits",
        "bilirubin_v*": "bilirubin longitudinal visits",
        "chol_v*": "cholesterol longitudinal visits",
        "ggt_v*": "GGT longitudinal visits",
        "gluc_fast_v*": "fasting_glucose longitudinal visits",
        "plt_v*": "platelets longitudinal visits",
        "triglyc_v*": "triglycerides longitudinal visits",
        "aixp_aix_result_BM_3_v*": "AIx longitudinal visits",
        "fibrotest_BM_2_v*": "FibroTest longitudinal visits",
        "fibs_stiffness_med_BM_1_v*": "liver_stiffness longitudinal visits",
        "evenements_hepatiques_majeurs": "risk_hepatic_event observed binary label",
        "evenements_hepatiques_age_occur": "hepatic_event_age",
        "death": "risk_death observed binary label",
        "death_age_occur": "death_age",
    }


def build_schema_audit(raw_df: pl.DataFrame) -> dict[str, Any]:
    visit_groups = visit_column_groups(raw_df.columns)
    static_columns = [column_name for column_name in raw_df.columns if VISIT_PATTERN.match(column_name) is None]
    target_summary = {}
    for semantic_target, raw_target in TARGET_COLUMN_MAP.items():
        target_frame = raw_df.select(raw_target).with_columns(pl.col(raw_target).cast(pl.Float64, strict=False))
        positives = int(target_frame.filter(pl.col(raw_target) == 1).height)
        negatives = int(target_frame.filter(pl.col(raw_target) == 0).height)
        nulls = int(target_frame.get_column(raw_target).null_count())
        target_summary[semantic_target] = {
            "raw_target_column": raw_target,
            "positive_count": positives,
            "negative_count": negatives,
            "null_count": nulls,
        }

    audit_payload = {
        "generated_at_utc": now_utc_iso(),
        "row_count": raw_df.height,
        "column_count": raw_df.width,
        "unique_id_count": int(raw_df.get_column(ID_COLUMN).n_unique()),
        "duplicate_id_count": int(raw_df.height - raw_df.get_column(ID_COLUMN).n_unique()),
        "static_column_count": len(static_columns),
        "visit_group_count": len(visit_groups),
        "visit_groups": {base_name: len(columns) for base_name, columns in visit_groups.items()},
        "target_summary": target_summary,
        "submission_columns": pl.read_csv(SUBMISSION_TEMPLATE_PATH, n_rows=1).columns,
        "raw_to_semantic_map": raw_to_semantic_map(),
    }
    if TEST_DATA_PATH.exists():
        test_df = pl.read_csv(TEST_DATA_PATH)
        audit_payload["test_schema"] = {
            "row_count": test_df.height,
            "column_count": test_df.width,
            "columns": test_df.columns,
        }
    return audit_payload


def baseline_feature_columns(raw_df: pl.DataFrame) -> list[str]:
    return [
        column_name
        for column_name in raw_df.columns
        if column_name not in NON_FEATURE_COLUMNS
    ]


def add_visit_summary_features(raw_df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(raw_df.columns)
    lazy_frame = raw_df.lazy()
    summary_expressions: list[pl.Expr] = []
    summary_feature_names: list[str] = []

    for base_name, ordered_columns in visit_groups.items():
        first_name = f"{base_name}_first_non_null"
        last_name = f"{base_name}_last_non_null"
        delta_name = f"{base_name}_delta_last_first"
        abs_delta_name = f"{base_name}_abs_delta_last_first"
        count_name = f"{base_name}_observed_count"

        numeric_columns = [pl.col(column_name).cast(pl.Float64, strict=False) for column_name in ordered_columns]
        numeric_columns_reversed = [pl.col(column_name).cast(pl.Float64, strict=False) for column_name in reversed(ordered_columns)]

        first_expr = pl.coalesce(numeric_columns).alias(first_name)
        last_expr = pl.coalesce(numeric_columns_reversed).alias(last_name)
        delta_expr = (pl.coalesce(numeric_columns_reversed) - pl.coalesce(numeric_columns)).alias(delta_name)
        abs_delta_expr = (
            (pl.coalesce(numeric_columns_reversed) - pl.coalesce(numeric_columns)).abs().alias(abs_delta_name)
        )
        count_expr = sum(pl.col(column_name).is_not_null().cast(pl.Int16) for column_name in ordered_columns).alias(count_name)

        summary_expressions.extend([first_expr, last_expr, delta_expr, abs_delta_expr, count_expr])
        summary_feature_names.extend([first_name, last_name, delta_name, abs_delta_name, count_name])

    summary_frame = lazy_frame.select(pl.col(ID_COLUMN), *summary_expressions).collect()
    merged = raw_df.join(summary_frame, on=ID_COLUMN, how="left")
    return merged, summary_feature_names


def ensure_feature_columns(frame: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    missing_columns = [column_name for column_name in feature_columns if column_name not in frame.columns]
    if not missing_columns:
        return frame
    return frame.with_columns(
        [pl.lit(None).cast(pl.Float64).alias(column_name) for column_name in missing_columns]
    )


def target_ready_frame(
    raw_df: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subset = ensure_feature_columns(raw_df, feature_columns).filter(pl.col(target_column).is_not_null()).with_columns(
        pl.col(target_column).cast(pl.Int8, strict=False),
        *[pl.col(column_name).cast(pl.Float64, strict=False) for column_name in feature_columns],
    )
    features = subset.select(feature_columns).fill_null(np.nan).to_numpy().astype(np.float32)
    target = subset.get_column(target_column).to_numpy().astype(np.int8)
    row_ids = subset.get_column(ID_COLUMN).to_list()
    return features, target, row_ids


def inference_ready_matrix(frame: pl.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, list[int]]:
    subset = ensure_feature_columns(frame, feature_columns).with_columns(
        *[pl.col(column_name).cast(pl.Float64, strict=False) for column_name in feature_columns],
    )
    features = subset.select(feature_columns).fill_null(np.nan).to_numpy().astype(np.float32)
    submission_ids = subset.get_column(SUBMISSION_ID_COLUMN).to_list()
    return features, submission_ids


def evaluate_target(
    raw_df: pl.DataFrame,
    feature_columns: list[str],
    semantic_target: str,
) -> dict[str, Any]:
    detailed = evaluate_target_detailed(raw_df, None, feature_columns, semantic_target)
    return {
        **detailed["summary"],
        "feature_importance": detailed["feature_importance"],
    }


def evaluate_target_detailed(
    raw_df: pl.DataFrame,
    test_df: pl.DataFrame | None,
    feature_columns: list[str],
    semantic_target: str,
) -> dict[str, Any]:
    raw_target = TARGET_COLUMN_MAP[semantic_target]
    features, target, row_ids = target_ready_frame(raw_df, feature_columns, raw_target)
    split = StratifiedKFold(n_splits=DEFAULT_FOLDS, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    oof_scores = np.zeros(len(target), dtype=np.float64)
    fold_summaries: list[dict[str, Any]] = []

    test_features: np.ndarray | None = None
    submission_ids: list[int] | None = None
    if test_df is not None:
        test_features, submission_ids = inference_ready_matrix(test_df, feature_columns)

    for fold_index, (train_index, valid_index) in enumerate(split.split(features, target), start=1):
        model = LGBMClassifier(**LIGHTGBM_PARAMS)
        model.fit(features[train_index], target[train_index])
        probabilities = model.predict_proba(features[valid_index])[:, 1]
        oof_scores[valid_index] = probabilities
        fold_summaries.append(
            {
                "fold": fold_index,
                "roc_auc": float(roc_auc_score(target[valid_index], probabilities)),
                "average_precision": float(average_precision_score(target[valid_index], probabilities)),
                "row_count": int(len(valid_index)),
                "positive_count": int(target[valid_index].sum()),
            }
        )

    full_model = LGBMClassifier(**LIGHTGBM_PARAMS)
    full_model.fit(features, target)
    feature_importance = pl.DataFrame(
        {
            "feature": feature_columns,
            "importance": full_model.feature_importances_.tolist(),
            "target": [semantic_target] * len(feature_columns),
        }
    ).sort("importance", descending=True)

    detailed_payload: dict[str, Any] = {
        "summary": {
            "semantic_target": semantic_target,
            "raw_target_column": raw_target,
            "row_count": int(len(target)),
            "positive_count": int(target.sum()),
            "null_excluded_count": int(raw_df.height - len(target)),
            "mean_roc_auc": float(roc_auc_score(target, oof_scores)),
            "mean_average_precision": float(average_precision_score(target, oof_scores)),
            "folds": fold_summaries,
        },
        "feature_importance": feature_importance,
        "oof_predictions": pl.DataFrame(
            {
                ID_COLUMN: row_ids,
                f"{semantic_target}_target": target.tolist(),
                f"{semantic_target}_oof_score": oof_scores.tolist(),
            }
        ),
    }

    if test_df is not None and test_features is not None and submission_ids is not None:
        full_model_predictions = full_model.predict_proba(test_features)[:, 1]
        detailed_payload["test_predictions"] = pl.DataFrame(
            {
                SUBMISSION_ID_COLUMN: submission_ids,
                f"{semantic_target}_prediction": full_model_predictions.tolist(),
            }
        )

    return detailed_payload


def evaluate_experiment(raw_df: pl.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    target_results = {
        semantic_target: evaluate_target(raw_df, feature_columns, semantic_target)
        for semantic_target in TARGET_COLUMN_MAP
    }
    combined_score = float(np.mean([result["mean_roc_auc"] for result in target_results.values()]))
    combined_average_precision = float(np.mean([result["mean_average_precision"] for result in target_results.values()]))
    return {
        "generated_at_utc": now_utc_iso(),
        "official_metric_confirmed": False,
        "primary_validation_metric": PRIMARY_VALIDATION_METRIC,
        "split_strategy": f"StratifiedKFold(n_splits={DEFAULT_FOLDS}, shuffle=True, random_state={DEFAULT_RANDOM_STATE})",
        "feature_count": len(feature_columns),
        "targets": {
            semantic_target: {
                key: value
                for key, value in result.items()
                if key != "feature_importance"
            }
            for semantic_target, result in target_results.items()
        },
        "combined_score": combined_score,
        "combined_average_precision": combined_average_precision,
        "lightgbm_params": LIGHTGBM_PARAMS,
    }


def combined_feature_importance_table(target_results: dict[str, Any]) -> pl.DataFrame:
    return pl.concat([result["feature_importance"] for result in target_results.values()], how="vertical_relaxed")


def build_fold_scores_table(target_results: dict[str, Any]) -> pl.DataFrame:
    fold_rows: list[dict[str, Any]] = []
    for semantic_target, result in target_results.items():
        for fold_summary in result["summary"]["folds"]:
            fold_rows.append(
                {
                    "target": semantic_target,
                    "fold": fold_summary["fold"],
                    "roc_auc": fold_summary["roc_auc"],
                    "average_precision": fold_summary["average_precision"],
                    "row_count": fold_summary["row_count"],
                    "positive_count": fold_summary["positive_count"],
                }
            )
    return pl.DataFrame(fold_rows).sort(["target", "fold"])


def build_oof_table(raw_df: pl.DataFrame, target_results: dict[str, Any]) -> pl.DataFrame:
    oof_table = raw_df.select(
        pl.col(ID_COLUMN),
        pl.col(TARGET_COLUMN_MAP["risk_hepatic_event"]).cast(pl.Float64, strict=False).alias("risk_hepatic_event_actual"),
        pl.col(TARGET_COLUMN_MAP["risk_death"]).cast(pl.Float64, strict=False).alias("risk_death_actual"),
    )
    for result in target_results.values():
        oof_table = oof_table.join(result["oof_predictions"], on=ID_COLUMN, how="left")
    return oof_table


def build_test_prediction_table(target_results: dict[str, Any]) -> pl.DataFrame:
    if any("test_predictions" not in result for result in target_results.values()):
        raise ValueError("Test predictions are not available for all targets.")

    prediction_table: pl.DataFrame | None = None
    for result in target_results.values():
        target_prediction_table = result["test_predictions"]
        prediction_table = (
            target_prediction_table
            if prediction_table is None
            else prediction_table.join(target_prediction_table, on=SUBMISSION_ID_COLUMN, how="inner")
        )
    if prediction_table is None:
        raise ValueError("No test predictions were built.")
    return prediction_table.sort(SUBMISSION_ID_COLUMN)


def build_submission_frame(target_results: dict[str, Any]) -> pl.DataFrame:
    template = pl.read_csv(SUBMISSION_TEMPLATE_PATH).select([SUBMISSION_ID_COLUMN])
    prediction_table = build_test_prediction_table(target_results).rename(
        {
            "risk_hepatic_event_prediction": "risk_hepatic_event",
            "risk_death_prediction": "risk_death",
        }
    )
    submission = template.join(prediction_table, on=SUBMISSION_ID_COLUMN, how="left")
    if submission.get_column("risk_hepatic_event").null_count() or submission.get_column("risk_death").null_count():
        raise ValueError("Submission contains null predictions after joining template and test predictions.")
    return submission


def compare_against_baseline(candidate_score: float, baseline_score: float) -> dict[str, Any]:
    improvement = candidate_score - baseline_score
    return {
        "baseline_combined_score": baseline_score,
        "candidate_combined_score": candidate_score,
        "absolute_improvement": improvement,
        "accepted": bool(improvement > 0.0),
    }


def wandb_mode() -> str:
    return "online" if wandb_api_key_valid() else "offline"


def log_experiment_to_wandb(
    experiment_name: str,
    experiment_dir: Path,
    summary: dict[str, Any],
    extra_config: dict[str, Any],
    artifact_paths: list[Path],
) -> dict[str, Any]:
    configure_wandb_env()
    output_dir = experiment_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=DEFAULT_WANDB_PROJECT,
        name=experiment_name,
        mode=wandb_mode(),
        dir=str(output_dir),
        config={
            "experiment_name": experiment_name,
            "experiment_dir": str(experiment_dir.relative_to(PROJECT_ROOT)),
            **extra_config,
        },
        reinit=True,
    )
    run.log(
        {
            "combined_score": summary["combined_score"],
            "combined_average_precision": summary["combined_average_precision"],
            "risk_hepatic_event_mean_roc_auc": summary["targets"]["risk_hepatic_event"]["mean_roc_auc"],
            "risk_death_mean_roc_auc": summary["targets"]["risk_death"]["mean_roc_auc"],
            "risk_hepatic_event_mean_average_precision": summary["targets"]["risk_hepatic_event"]["mean_average_precision"],
            "risk_death_mean_average_precision": summary["targets"]["risk_death"]["mean_average_precision"],
        }
    )
    run.summary.update(
        {
            "primary_validation_metric": summary["primary_validation_metric"],
            "official_metric_confirmed": summary["official_metric_confirmed"],
            "artifact_paths": [str(path.relative_to(PROJECT_ROOT)) for path in artifact_paths if path.exists()],
            "wandb_mode": wandb_mode(),
            "accepted": summary.get("accepted"),
        }
    )
    for path in artifact_paths:
        if path.exists():
            run.save(str(path), base_path=str(PROJECT_ROOT))
    run.finish()
    return {
        "project": DEFAULT_WANDB_PROJECT,
        "run_name": experiment_name,
        "mode": wandb_mode(),
        "api_key_present": wandb_api_key_present(),
        "api_key_valid": wandb_api_key_valid(),
        "artifact_paths": [str(path.relative_to(PROJECT_ROOT)) for path in artifact_paths if path.exists()],
    }


# ---------------------------------------------------------------------------
# Feature engineering functions for exp003-exp031
# ---------------------------------------------------------------------------


def add_visit_minmax_mean_std_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(df.columns)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for base, cols in visit_groups.items():
        float_cols = [pl.col(c).cast(pl.Float64, strict=False) for c in cols]
        min_name = f"{base}_visit_min"
        max_name = f"{base}_visit_max"
        mean_name = f"{base}_visit_mean"
        std_name = f"{base}_visit_std"
        range_name = f"{base}_visit_range"
        exprs.append(pl.min_horizontal(*float_cols).alias(min_name))
        exprs.append(pl.max_horizontal(*float_cols).alias(max_name))
        exprs.append(pl.concat_list(float_cols).list.mean().alias(mean_name))
        exprs.append(pl.concat_list(float_cols).list.std(ddof=1).alias(std_name))
        exprs.append((pl.max_horizontal(*float_cols) - pl.min_horizontal(*float_cols)).alias(range_name))
        names.extend([min_name, max_name, mean_name, std_name, range_name])
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_visit_slope_trend_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(df.columns)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for base in visit_groups:
        delta_col = f"{base}_delta_last_first"
        count_col = f"{base}_observed_count"
        mean_col = f"{base}_visit_mean"
        std_col = f"{base}_visit_std"
        if delta_col not in df.columns or count_col not in df.columns:
            continue
        slope_name = f"{base}_slope_proxy"
        pos_name = f"{base}_trend_positive"
        neg_name = f"{base}_trend_negative"
        exprs.append(
            (pl.col(delta_col).cast(pl.Float64, strict=False) / pl.col(count_col).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)).alias(slope_name)
        )
        exprs.append((pl.col(delta_col).cast(pl.Float64, strict=False) > 0).cast(pl.Float64).alias(pos_name))
        exprs.append((pl.col(delta_col).cast(pl.Float64, strict=False) < 0).cast(pl.Float64).alias(neg_name))
        names.extend([slope_name, pos_name, neg_name])
        if mean_col in df.columns and std_col in df.columns:
            cv_name = f"{base}_cv"
            exprs.append(
                (
                    pl.col(std_col).cast(pl.Float64, strict=False) /
                    pl.col(mean_col).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)
                ).alias(cv_name)
            )
            names.append(cv_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_visit_recency_persistence_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(df.columns)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for base, cols in visit_groups.items():
        total = len(cols)
        count_col = f"{base}_observed_count"
        if count_col not in df.columns:
            continue
        persist_name = f"{base}_persistence"
        exprs.append((pl.col(count_col).cast(pl.Float64, strict=False) / float(total)).alias(persist_name))
        names.append(persist_name)
        mid = total // 2
        early_cols = cols[:mid] if mid > 0 else cols[:1]
        late_cols = cols[mid:] if mid < total else cols[-1:]
        if early_cols:
            early_name = f"{base}_early_coverage"
            early_count = sum(pl.col(c).is_not_null().cast(pl.Int16) for c in early_cols)
            exprs.append((early_count.cast(pl.Float64) / float(len(early_cols))).alias(early_name))
            names.append(early_name)
        if late_cols:
            late_name = f"{base}_late_coverage"
            late_count = sum(pl.col(c).is_not_null().cast(pl.Int16) for c in late_cols)
            exprs.append((late_count.cast(pl.Float64) / float(len(late_cols))).alias(late_name))
            names.append(late_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_visit_missingness_trajectory_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(df.columns)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for base, cols in visit_groups.items():
        first_visit_expr = pl.lit(0.0)
        last_visit_expr = pl.lit(0.0)
        for i, c in enumerate(cols, start=1):
            not_null = pl.col(c).is_not_null()
            first_visit_expr = pl.when(first_visit_expr == 0.0).then(pl.when(not_null).then(pl.lit(float(i))).otherwise(pl.lit(0.0))).otherwise(first_visit_expr)
            last_visit_expr = pl.when(not_null).then(pl.lit(float(i))).otherwise(last_visit_expr)
        fov_name = f"{base}_first_observed_visit"
        lov_name = f"{base}_last_observed_visit"
        span_name = f"{base}_obs_span"
        exprs.append(first_visit_expr.alias(fov_name))
        exprs.append(last_visit_expr.alias(lov_name))
        exprs.append((last_visit_expr - first_visit_expr).alias(span_name))
        names.extend([fov_name, lov_name, span_name])
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_missingness_flag_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    visit_groups = visit_column_groups(df.columns)
    exprs: list[pl.Expr] = []
    names: list[str] = []
    panel_coverage_exprs: list[pl.Expr] = []
    for base, cols in visit_groups.items():
        all_null_name = f"{base}_all_null_flag"
        any_obs_name = f"{base}_any_observed_flag"
        null_checks = [pl.col(c).is_null() for c in cols]
        all_null_expr = pl.all_horizontal(*null_checks).cast(pl.Float64)
        any_obs_expr = (~pl.all_horizontal(*null_checks)).cast(pl.Float64)
        exprs.append(all_null_expr.alias(all_null_name))
        exprs.append(any_obs_expr.alias(any_obs_name))
        names.extend([all_null_name, any_obs_name])
        count_col = f"{base}_observed_count"
        if count_col in df.columns:
            panel_coverage_exprs.append(pl.col(count_col).cast(pl.Float64, strict=False) / float(len(cols)))
    if panel_coverage_exprs:
        coverage_name = "total_panel_coverage"
        exprs.append(pl.concat_list(panel_coverage_exprs).list.mean().alias(coverage_name))
        names.append(coverage_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_log_winsorized_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    biomarkers = ["alt", "ast", "ggt", "bilirubin", "triglyc", "gluc_fast", "plt", "BMI", "chol"]
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for base in biomarkers:
        mean_col = f"{base}_visit_mean"
        first_col = f"{base}_first_non_null"
        src_col = mean_col if mean_col in df.columns else (first_col if first_col in df.columns else None)
        if src_col is None:
            continue
        log_name = f"log1p_{base}"
        exprs.append(pl.col(src_col).cast(pl.Float64, strict=False).clip(lower_bound=0.0).log1p().alias(log_name))
        names.append(log_name)
        series = df.get_column(src_col).cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        if len(series) > 0:
            p1 = float(np.nanpercentile(series, 1))
            p99 = float(np.nanpercentile(series, 99))
            wins_name = f"wins_{base}"
            exprs.append(pl.col(src_col).cast(pl.Float64, strict=False).clip(lower_bound=p1, upper_bound=p99).alias(wins_name))
            names.append(wins_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_clinical_burden_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    burden_cols = ["T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]
    available = [c for c in burden_cols if c in df.columns]
    if not available:
        return df, []
    bool_exprs = [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) > 0.0 for c in available]
    count_name = "comorbidity_count"
    score_name = "comorbidity_burden_score"
    weights = {"T2DM": 2.0, "Hypertension": 1.0, "Dyslipidaemia": 1.0, "bariatric_surgery": 1.0}
    weight_exprs = [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) * weights.get(c, 1.0) for c in available]
    count_expr = pl.sum_horizontal(*bool_exprs).cast(pl.Float64).alias(count_name)
    score_expr = pl.sum_horizontal(*weight_exprs).alias(score_name)
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), count_expr, score_expr).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, [count_name, score_name]


def add_metabolic_burden_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    bmi_col = "BMI_visit_mean" if "BMI_visit_mean" in df.columns else ("BMI_first_non_null" if "BMI_first_non_null" in df.columns else None)
    gluc_col = "gluc_fast_visit_mean" if "gluc_fast_visit_mean" in df.columns else ("gluc_fast_first_non_null" if "gluc_fast_first_non_null" in df.columns else None)
    trig_col = "triglyc_visit_mean" if "triglyc_visit_mean" in df.columns else ("triglyc_first_non_null" if "triglyc_first_non_null" in df.columns else None)
    flag_exprs: list[pl.Expr] = []
    if bmi_col:
        flag_exprs.append((pl.col(bmi_col).cast(pl.Float64, strict=False) >= 30.0).cast(pl.Float64).fill_null(0.0))
        exprs.append((pl.col(bmi_col).cast(pl.Float64, strict=False) >= 30.0).cast(pl.Float64).fill_null(0.0).alias("obese_flag"))
        names.append("obese_flag")
    if gluc_col:
        flag_exprs.append((pl.col(gluc_col).cast(pl.Float64, strict=False) >= 7.0).cast(pl.Float64).fill_null(0.0))
        exprs.append((pl.col(gluc_col).cast(pl.Float64, strict=False) >= 7.0).cast(pl.Float64).fill_null(0.0).alias("high_glucose_flag"))
        names.append("high_glucose_flag")
    if trig_col:
        flag_exprs.append((pl.col(trig_col).cast(pl.Float64, strict=False) >= 1.7).cast(pl.Float64).fill_null(0.0))
        exprs.append((pl.col(trig_col).cast(pl.Float64, strict=False) >= 1.7).cast(pl.Float64).fill_null(0.0).alias("high_triglyc_flag"))
        names.append("high_triglyc_flag")
    if "T2DM" in df.columns:
        flag_exprs.append(pl.col("T2DM").cast(pl.Float64, strict=False).fill_null(0.0))
    if "Dyslipidaemia" in df.columns:
        flag_exprs.append(pl.col("Dyslipidaemia").cast(pl.Float64, strict=False).fill_null(0.0))
    if flag_exprs:
        exprs.append(pl.sum_horizontal(*flag_exprs).cast(pl.Float64).alias("metabolic_burden_score"))
        names.append("metabolic_burden_score")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_inflammatory_burden_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    thresholds = {
        "alt": ("alt_visit_mean", "alt_first_non_null", 40.0, "high_alt_flag"),
        "ast": ("ast_visit_mean", "ast_first_non_null", 40.0, "high_ast_flag"),
        "ggt": ("ggt_visit_mean", "ggt_first_non_null", 50.0, "high_ggt_flag"),
        "bilirubin": ("bilirubin_visit_mean", "bilirubin_first_non_null", 1.2, "high_bilirubin_flag"),
    }
    flag_exprs: list[pl.Expr] = []
    for _, (mean_c, first_c, thresh, flag_name) in thresholds.items():
        src = mean_c if mean_c in df.columns else (first_c if first_c in df.columns else None)
        if src is None:
            continue
        flag = (pl.col(src).cast(pl.Float64, strict=False) > thresh).cast(pl.Float64).fill_null(0.0)
        exprs.append(flag.alias(flag_name))
        names.append(flag_name)
        flag_exprs.append(flag)
    if flag_exprs:
        exprs.append(pl.sum_horizontal(*flag_exprs).cast(pl.Float64).alias("inflammatory_burden_score"))
        names.append("inflammatory_burden_score")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_cardio_burden_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    flag_exprs: list[pl.Expr] = []
    if "Hypertension" in df.columns:
        flag = pl.col("Hypertension").cast(pl.Float64, strict=False).fill_null(0.0)
        flag_exprs.append(flag)
    chol_col = "chol_visit_mean" if "chol_visit_mean" in df.columns else ("chol_first_non_null" if "chol_first_non_null" in df.columns else None)
    if chol_col:
        flag = (pl.col(chol_col).cast(pl.Float64, strict=False) > 5.0).cast(pl.Float64).fill_null(0.0)
        exprs.append(flag.alias("high_chol_flag"))
        names.append("high_chol_flag")
        flag_exprs.append(flag)
    aix_col = "aixp_aix_result_BM_3_visit_mean" if "aixp_aix_result_BM_3_visit_mean" in df.columns else (
        "aixp_aix_result_BM_3_first_non_null" if "aixp_aix_result_BM_3_first_non_null" in df.columns else None
    )
    if aix_col:
        flag = (pl.col(aix_col).cast(pl.Float64, strict=False) > 30.0).cast(pl.Float64).fill_null(0.0)
        exprs.append(flag.alias("high_aix_flag"))
        names.append("high_aix_flag")
        flag_exprs.append(flag)
    if flag_exprs:
        exprs.append(pl.sum_horizontal(*flag_exprs).cast(pl.Float64).alias("cardio_burden_score"))
        names.append("cardio_burden_score")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_renal_cardiometabolic_burden_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    plt_col = "plt_visit_mean" if "plt_visit_mean" in df.columns else ("plt_first_non_null" if "plt_first_non_null" in df.columns else None)
    flag_exprs: list[pl.Expr] = []
    if plt_col:
        flag = (pl.col(plt_col).cast(pl.Float64, strict=False) < 150.0).cast(pl.Float64).fill_null(0.0)
        exprs.append(flag.alias("low_plt_flag"))
        names.append("low_plt_flag")
        flag_exprs.append(flag)
    if "Hypertension" in df.columns:
        flag_exprs.append(pl.col("Hypertension").cast(pl.Float64, strict=False).fill_null(0.0))
    if "T2DM" in df.columns:
        flag_exprs.append(pl.col("T2DM").cast(pl.Float64, strict=False).fill_null(0.0))
    if flag_exprs:
        exprs.append(pl.sum_horizontal(*flag_exprs).cast(pl.Float64).alias("renal_cardio_burden_score"))
        names.append("renal_cardio_burden_score")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_ratio_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    alt_c = _src("alt")
    ast_c = _src("ast")
    ggt_c = _src("ggt")
    plt_c = _src("plt")
    age_c = _src("Age")
    if alt_c and ast_c:
        exprs.append((pl.col(ast_c).cast(pl.Float64, strict=False) / pl.col(alt_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)).alias("ast_alt_ratio"))
        names.append("ast_alt_ratio")
    if alt_c and ggt_c:
        exprs.append((pl.col(ggt_c).cast(pl.Float64, strict=False) / pl.col(alt_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)).alias("ggt_alt_ratio"))
        names.append("ggt_alt_ratio")
    if alt_c and plt_c and age_c and ast_c:
        fib4_expr = (
            pl.col(age_c).cast(pl.Float64, strict=False) *
            pl.col(ast_c).cast(pl.Float64, strict=False)
        ) / (
            pl.col(plt_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0) *
            pl.col(alt_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0).sqrt()
        )
        exprs.append(fib4_expr.alias("fib4_proxy"))
        names.append("fib4_proxy")
    chol_c = _src("chol")
    trig_c = _src("triglyc")
    if chol_c and trig_c:
        exprs.append((pl.col(trig_c).cast(pl.Float64, strict=False) / pl.col(chol_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)).alias("triglyc_chol_ratio"))
        names.append("triglyc_chol_ratio")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_hemodynamic_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    aix_c = _src("aixp_aix_result_BM_3")
    stiff_c = _src("fibs_stiffness_med_BM_1")
    age_c = _src("Age")
    bmi_c = _src("BMI")
    if aix_c:
        exprs.append(pl.col(aix_c).cast(pl.Float64, strict=False).alias("aix_mean"))
        names.append("aix_mean")
        if age_c:
            adj_expr = (pl.col(aix_c).cast(pl.Float64, strict=False) - (pl.col(age_c).cast(pl.Float64, strict=False) - 25.0) * 0.33)
            exprs.append(adj_expr.alias("aix_age_adjusted"))
            names.append("aix_age_adjusted")
    if stiff_c:
        exprs.append(pl.col(stiff_c).cast(pl.Float64, strict=False).alias("liver_stiffness_mean"))
        names.append("liver_stiffness_mean")
        if bmi_c:
            exprs.append((pl.col(stiff_c).cast(pl.Float64, strict=False) * pl.col(bmi_c).cast(pl.Float64, strict=False)).alias("stiffness_bmi_product"))
            names.append("stiffness_bmi_product")
    fibro_c = _src("fibrotest_BM_2")
    if fibro_c:
        exprs.append(pl.col(fibro_c).cast(pl.Float64, strict=False).alias("fibrotest_mean"))
        names.append("fibrotest_mean")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_pairwise_interaction_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    key_pairs = [
        ("fib4_proxy", "fibrotest_mean"),
        ("metabolic_burden_score", "inflammatory_burden_score"),
        ("comorbidity_count", "ast_alt_ratio"),
        ("log1p_alt", "log1p_ast"),
        ("BMI_visit_mean", "gluc_fast_visit_mean"),
        ("liver_stiffness_mean", "fibrotest_mean"),
    ]
    fallback = {
        "BMI_visit_mean": "BMI_first_non_null",
        "gluc_fast_visit_mean": "gluc_fast_first_non_null",
    }
    for col_a, col_b in key_pairs:
        ca = col_a if col_a in df.columns else fallback.get(col_a)
        cb = col_b if col_b in df.columns else fallback.get(col_b)
        if ca is None or cb is None or ca not in df.columns or cb not in df.columns:
            continue
        inter_name = f"inter_{ca[:20]}_{cb[:20]}"
        exprs.append((pl.col(ca).cast(pl.Float64, strict=False) * pl.col(cb).cast(pl.Float64, strict=False)).alias(inter_name))
        names.append(inter_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_higher_risk_interaction_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    burden_cols = [c for c in ["comorbidity_burden_score", "metabolic_burden_score", "inflammatory_burden_score", "cardio_burden_score"] if c in df.columns]
    if len(burden_cols) >= 2:
        total_burden = pl.sum_horizontal(*[pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in burden_cols])
        exprs.append(total_burden.alias("total_burden_score"))
        names.append("total_burden_score")
        exprs.append((total_burden > 3.0).cast(pl.Float64).alias("high_risk_gate"))
        names.append("high_risk_gate")
        if "fib4_proxy" in df.columns:
            exprs.append((total_burden * pl.col("fib4_proxy").cast(pl.Float64, strict=False)).alias("burden_fib4_interaction"))
            names.append("burden_fib4_interaction")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_threshold_flag_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    thresholds_list = [
        ("alt", 3.0 * 40.0, "alt_3x_uln_flag"),
        ("ast", 3.0 * 40.0, "ast_3x_uln_flag"),
        ("ggt", 100.0, "ggt_high_flag"),
        ("plt", 100.0, "plt_severe_low_flag"),
        ("BMI", 35.0, "severe_obese_flag"),
        ("gluc_fast", 11.0, "very_high_glucose_flag"),
        ("fibs_stiffness_med_BM_1", 8.0, "stiffness_f2_flag"),
        ("fibs_stiffness_med_BM_1", 13.0, "stiffness_f3_flag"),
        ("fibrotest_BM_2", 0.48, "fibrotest_f2_flag"),
    ]
    for base, thresh, flag_name in thresholds_list:
        c = _src(base)
        if c is None:
            continue
        exprs.append((pl.col(c).cast(pl.Float64, strict=False) >= thresh).cast(pl.Float64).fill_null(0.0).alias(flag_name))
        names.append(flag_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_quantile_bin_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    quant_bases = ["alt", "ast", "ggt", "BMI", "fib4_proxy", "total_burden_score"]
    for base in quant_bases:
        if base in df.columns:
            c = base
        else:
            c = _src(base)
        if c is None or c not in df.columns:
            continue
        series = df.get_column(c).cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        if len(series) < 5:
            continue
        quintiles = np.nanpercentile(series, [20, 40, 60, 80])
        q1, q2, q3, q4 = float(quintiles[0]), float(quintiles[1]), float(quintiles[2]), float(quintiles[3])
        bin_name = f"{c[:25]}_qbin5"
        bin_expr = (
            pl.when(pl.col(c).cast(pl.Float64, strict=False) <= q1).then(pl.lit(1.0))
            .when(pl.col(c).cast(pl.Float64, strict=False) <= q2).then(pl.lit(2.0))
            .when(pl.col(c).cast(pl.Float64, strict=False) <= q3).then(pl.lit(3.0))
            .when(pl.col(c).cast(pl.Float64, strict=False) <= q4).then(pl.lit(4.0))
            .otherwise(pl.lit(5.0))
        )
        exprs.append(bin_expr.alias(bin_name))
        names.append(bin_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_group_relative_sex_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    if "gender" not in df.columns:
        return df, []
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    biomarkers = ["alt", "ast", "ggt", "BMI", "plt"]
    gender_col = pl.col("gender").cast(pl.Float64, strict=False)
    for base in biomarkers:
        c = _src(base)
        if c is None:
            continue
        for g_val in [0.0, 1.0]:
            mask = df.get_column("gender").cast(pl.Float64, strict=False) == g_val
            vals = df.filter(mask).get_column(c).cast(pl.Float64, strict=False).drop_nulls().to_numpy()
            if len(vals) < 3:
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals)) or 1.0
            z_name = f"{c[:20]}_sex{int(g_val)}_zscore"
            exprs.append(
                pl.when(gender_col == g_val)
                .then((pl.col(c).cast(pl.Float64, strict=False) - mu) / sigma)
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias(z_name)
            )
            names.append(z_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_group_relative_age_band_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    age_c = _src("Age")
    if age_c is None:
        return df, []
    age_band = (df.get_column(age_c).cast(pl.Float64, strict=False) / 10.0).floor().cast(pl.Int32)
    df_tmp = df.with_columns(age_band.alias("_age_band_tmp"))
    exprs: list[pl.Expr] = []
    names: list[str] = []
    biomarkers = ["alt", "BMI", "fib4_proxy"]
    try:
        for base in biomarkers:
            c = _src(base) if base not in df.columns else base
            if c is None or c not in df.columns:
                continue
            band_vals = df_tmp.group_by("_age_band_tmp").agg(
                pl.col(c).cast(pl.Float64, strict=False).mean().alias("band_mean"),
                pl.col(c).cast(pl.Float64, strict=False).std(ddof=1).alias("band_std"),
            )
            for row in band_vals.iter_rows(named=True):
                band = row["_age_band_tmp"]
                mu = row["band_mean"] or 0.0
                sigma = row["band_std"] or 1.0
                z_name = f"{c[:20]}_aband{band}_zscore"
                exprs.append(
                    pl.when(
                        (pl.col(age_c).cast(pl.Float64, strict=False) / 10.0).floor().cast(pl.Int32) == band
                    )
                    .then((pl.col(c).cast(pl.Float64, strict=False) - mu) / (sigma if sigma > 0 else 1.0))
                    .otherwise(pl.lit(None).cast(pl.Float64))
                    .alias(z_name)
                )
                names.append(z_name)
    finally:
        df_tmp = df_tmp.drop("_age_band_tmp")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_percentile_rank_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []
    rank_cols = [
        c for c in ["fib4_proxy", "fibrotest_mean", "liver_stiffness_mean", "total_burden_score", "comorbidity_burden_score"]
        if c in df.columns
    ]

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    for base in ["alt", "ast", "BMI"]:
        c = _src(base)
        if c and c not in rank_cols:
            rank_cols.append(c)
    for c in rank_cols:
        if c not in df.columns:
            continue
        rank_name = f"{c[:25]}_pctrank"
        exprs.append(
            pl.col(c).cast(pl.Float64, strict=False).rank(method="average").alias(rank_name) / float(df.height)
        )
        names.append(rank_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_robust_scaling_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    scale_bases = ["alt", "ast", "ggt", "BMI", "fib4_proxy"]
    for base in scale_bases:
        c = _src(base) if base not in df.columns else base
        if c is None or c not in df.columns:
            continue
        series = df.get_column(c).cast(pl.Float64, strict=False).drop_nulls().to_numpy()
        if len(series) < 5:
            continue
        med = float(np.median(series))
        q25 = float(np.percentile(series, 25))
        q75 = float(np.percentile(series, 75))
        iqr = q75 - q25 or 1.0
        rs_name = f"{c[:25]}_robust"
        exprs.append(((pl.col(c).cast(pl.Float64, strict=False) - med) / iqr).alias(rs_name))
        names.append(rs_name)
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_hepatic_event_specific_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    stiff_c = _src("fibs_stiffness_med_BM_1")
    fibro_c = _src("fibrotest_BM_2")
    if stiff_c and fibro_c:
        exprs.append((pl.col(stiff_c).cast(pl.Float64, strict=False) * pl.col(fibro_c).cast(pl.Float64, strict=False)).alias("hepatic_stiffness_fibrotest"))
        names.append("hepatic_stiffness_fibrotest")
    stiff_delta = "fibs_stiffness_med_BM_1_delta_last_first"
    if stiff_delta in df.columns:
        exprs.append(pl.col(stiff_delta).cast(pl.Float64, strict=False).alias("hepatic_stiffness_delta"))
        names.append("hepatic_stiffness_delta")
        exprs.append((pl.col(stiff_delta).cast(pl.Float64, strict=False) > 0).cast(pl.Float64).alias("hepatic_stiffness_worsening"))
        names.append("hepatic_stiffness_worsening")
    if "fib4_proxy" in df.columns:
        exprs.append((pl.col("fib4_proxy").cast(pl.Float64, strict=False) > 2.67).cast(pl.Float64).fill_null(0.0).alias("fib4_high_risk_flag"))
        names.append("fib4_high_risk_flag")
        exprs.append((pl.col("fib4_proxy").cast(pl.Float64, strict=False) > 1.3).cast(pl.Float64).fill_null(0.0).alias("fib4_intermediate_flag"))
        names.append("fib4_intermediate_flag")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_death_specific_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    age_c = _src("Age")
    aix_c = _src("aixp_aix_result_BM_3")
    if age_c:
        exprs.append(pl.col(age_c).cast(pl.Float64, strict=False).alias("death_age_feature"))
        names.append("death_age_feature")
        exprs.append((pl.col(age_c).cast(pl.Float64, strict=False) > 65.0).cast(pl.Float64).fill_null(0.0).alias("elderly_flag"))
        names.append("elderly_flag")
        exprs.append((pl.col(age_c).cast(pl.Float64, strict=False) > 75.0).cast(pl.Float64).fill_null(0.0).alias("very_elderly_flag"))
        names.append("very_elderly_flag")
    if age_c and "comorbidity_burden_score" in df.columns:
        exprs.append((pl.col(age_c).cast(pl.Float64, strict=False) * pl.col("comorbidity_burden_score").cast(pl.Float64, strict=False)).alias("age_comorbidity_interaction"))
        names.append("age_comorbidity_interaction")
    if aix_c and age_c:
        exprs.append((pl.col(aix_c).cast(pl.Float64, strict=False) + pl.col(age_c).cast(pl.Float64, strict=False) * 0.1).alias("death_cardio_age_score"))
        names.append("death_cardio_age_score")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_categorical_encoding_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    if "gender" in df.columns:
        bmi_c = _src("BMI")
        if bmi_c:
            exprs.append((pl.col("gender").cast(pl.Float64, strict=False) * pl.col(bmi_c).cast(pl.Float64, strict=False)).alias("gender_bmi_interact"))
            names.append("gender_bmi_interact")
        if "T2DM" in df.columns:
            exprs.append((pl.col("gender").cast(pl.Float64, strict=False) * pl.col("T2DM").cast(pl.Float64, strict=False)).alias("gender_t2dm_interact"))
            names.append("gender_t2dm_interact")
        age_c = _src("Age")
        if age_c:
            exprs.append((pl.col("gender").cast(pl.Float64, strict=False) * pl.col(age_c).cast(pl.Float64, strict=False)).alias("gender_age_interact"))
            names.append("gender_age_interact")
    if "bariatric_surgery" in df.columns:
        bmi_c = _src("BMI")
        if bmi_c:
            exprs.append((pl.col("bariatric_surgery").cast(pl.Float64, strict=False) * pl.col(bmi_c).cast(pl.Float64, strict=False)).alias("bariatric_bmi_interact"))
            names.append("bariatric_bmi_interact")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_liver_fibrosis_composite_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    alt_c = _src("alt")
    ast_c = _src("ast")
    ggt_c = _src("ggt")
    stiff_c = _src("fibs_stiffness_med_BM_1")
    fibro_c = _src("fibrotest_BM_2")
    if alt_c and ast_c and ggt_c:
        liver_injury = (
            pl.col(alt_c).cast(pl.Float64, strict=False).fill_null(0.0) / 40.0 +
            pl.col(ast_c).cast(pl.Float64, strict=False).fill_null(0.0) / 40.0 +
            pl.col(ggt_c).cast(pl.Float64, strict=False).fill_null(0.0) / 50.0
        ) / 3.0
        exprs.append(liver_injury.alias("liver_injury_composite"))
        names.append("liver_injury_composite")
    if stiff_c and fibro_c:
        fibro_comp = (
            pl.col(stiff_c).cast(pl.Float64, strict=False).fill_null(0.0) / 13.0 +
            pl.col(fibro_c).cast(pl.Float64, strict=False).fill_null(0.0)
        ) / 2.0
        exprs.append(fibro_comp.alias("fibrosis_composite"))
        names.append("fibrosis_composite")
    if alt_c and stiff_c:
        exprs.append((pl.col(alt_c).cast(pl.Float64, strict=False) * pl.col(stiff_c).cast(pl.Float64, strict=False)).alias("alt_stiffness_product"))
        names.append("alt_stiffness_product")
    plt_delta = "plt_delta_last_first"
    if plt_delta in df.columns:
        exprs.append((pl.col(plt_delta).cast(pl.Float64, strict=False) < -20.0).cast(pl.Float64).fill_null(0.0).alias("plt_declining_flag"))
        names.append("plt_declining_flag")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_glucose_lipid_composite_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    gluc_c = _src("gluc_fast")
    trig_c = _src("triglyc")
    chol_c = _src("chol")
    bmi_c = _src("BMI")
    if gluc_c and trig_c:
        tyg = (
            pl.col(trig_c).cast(pl.Float64, strict=False).fill_null(1.0).clip(lower_bound=0.01) *
            pl.col(gluc_c).cast(pl.Float64, strict=False).fill_null(1.0).clip(lower_bound=0.01) /
            2.0
        ).log(base=float(np.e))
        exprs.append(tyg.alias("tyg_index"))
        names.append("tyg_index")
    if gluc_c and bmi_c:
        homa_proxy = pl.col(gluc_c).cast(pl.Float64, strict=False) * pl.col(bmi_c).cast(pl.Float64, strict=False) / 22.5
        exprs.append(homa_proxy.alias("homa_ir_proxy"))
        names.append("homa_ir_proxy")
    if chol_c and trig_c:
        atherogenic = pl.col(trig_c).cast(pl.Float64, strict=False) / pl.col(chol_c).cast(pl.Float64, strict=False).fill_null(1.0).replace(0.0, 1.0)
        exprs.append(atherogenic.alias("atherogenic_index"))
        names.append("atherogenic_index")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_inflammation_obesity_interaction_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    bmi_c = _src("BMI")
    alt_c = _src("alt")
    ggt_c = _src("ggt")
    trig_c = _src("triglyc")
    if bmi_c and alt_c:
        exprs.append((pl.col(bmi_c).cast(pl.Float64, strict=False) * pl.col(alt_c).cast(pl.Float64, strict=False)).alias("bmi_alt_product"))
        names.append("bmi_alt_product")
    if bmi_c and ggt_c:
        exprs.append((pl.col(bmi_c).cast(pl.Float64, strict=False) * pl.col(ggt_c).cast(pl.Float64, strict=False)).alias("bmi_ggt_product"))
        names.append("bmi_ggt_product")
    if "obese_flag" in df.columns and "inflammatory_burden_score" in df.columns:
        exprs.append((pl.col("obese_flag").cast(pl.Float64, strict=False) * pl.col("inflammatory_burden_score").cast(pl.Float64, strict=False)).alias("obese_inflam_gate"))
        names.append("obese_inflam_gate")
    if bmi_c and trig_c:
        exprs.append((pl.col(bmi_c).cast(pl.Float64, strict=False) * pl.col(trig_c).cast(pl.Float64, strict=False)).alias("bmi_triglyc_product"))
        names.append("bmi_triglyc_product")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_cardiac_stress_interaction_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    aix_c = _src("aixp_aix_result_BM_3")
    age_c = _src("Age")
    chol_c = _src("chol")
    trig_c = _src("triglyc")
    if aix_c and age_c:
        exprs.append((pl.col(aix_c).cast(pl.Float64, strict=False) * pl.col(age_c).cast(pl.Float64, strict=False)).alias("aix_age_product"))
        names.append("aix_age_product")
    if "Hypertension" in df.columns and chol_c:
        exprs.append((pl.col("Hypertension").cast(pl.Float64, strict=False) * pl.col(chol_c).cast(pl.Float64, strict=False)).alias("htn_chol_interact"))
        names.append("htn_chol_interact")
    if aix_c and trig_c:
        exprs.append((pl.col(aix_c).cast(pl.Float64, strict=False) * pl.col(trig_c).cast(pl.Float64, strict=False)).alias("aix_triglyc_product"))
        names.append("aix_triglyc_product")
    if "cardio_burden_score" in df.columns and age_c:
        exprs.append((pl.col("cardio_burden_score").cast(pl.Float64, strict=False) * pl.col(age_c).cast(pl.Float64, strict=False)).alias("cardio_age_product"))
        names.append("cardio_age_product")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def add_renal_hepatic_interaction_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    exprs: list[pl.Expr] = []
    names: list[str] = []

    def _src(base: str) -> str | None:
        mean = f"{base}_visit_mean"
        first = f"{base}_first_non_null"
        return mean if mean in df.columns else (first if first in df.columns else None)

    plt_c = _src("plt")
    stiff_c = _src("fibs_stiffness_med_BM_1")
    bmi_c = _src("BMI")
    if plt_c and stiff_c:
        exprs.append((pl.col(stiff_c).cast(pl.Float64, strict=False) / pl.col(plt_c).cast(pl.Float64, strict=False).fill_null(150.0).replace(0.0, 150.0)).alias("stiffness_per_plt"))
        names.append("stiffness_per_plt")
    if "fib4_proxy" in df.columns and plt_c:
        exprs.append((pl.col("fib4_proxy").cast(pl.Float64, strict=False) * (1.0 / pl.col(plt_c).cast(pl.Float64, strict=False).fill_null(150.0).replace(0.0, 150.0))).alias("fib4_per_plt"))
        names.append("fib4_per_plt")
    if bmi_c and stiff_c and plt_c:
        exprs.append((pl.col(bmi_c).cast(pl.Float64, strict=False) * pl.col(stiff_c).cast(pl.Float64, strict=False) / pl.col(plt_c).cast(pl.Float64, strict=False).fill_null(150.0).replace(0.0, 150.0)).alias("bmi_stiff_per_plt"))
        names.append("bmi_stiff_per_plt")
    if not exprs:
        return df, []
    feature_frame = df.lazy().select(pl.col(ID_COLUMN), *exprs).collect()
    merged = df.join(feature_frame, on=ID_COLUMN, how="left")
    return merged, names


def build_cumulative_features(df: pl.DataFrame, up_to_exp: int) -> tuple[pl.DataFrame, list[str]]:
    """Build cumulative feature set up to and including the given experiment number."""
    enriched = df
    feature_names: list[str] = []

    enriched, names = add_visit_summary_features(enriched)
    feature_names.extend(names)

    if up_to_exp >= 3:
        enriched, names = add_visit_minmax_mean_std_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 4:
        enriched, names = add_visit_slope_trend_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 5:
        enriched, names = add_visit_recency_persistence_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 6:
        enriched, names = add_visit_missingness_trajectory_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 7:
        enriched, names = add_missingness_flag_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 8:
        enriched, names = add_log_winsorized_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 9:
        enriched, names = add_clinical_burden_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 10:
        enriched, names = add_metabolic_burden_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 11:
        enriched, names = add_inflammatory_burden_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 12:
        enriched, names = add_cardio_burden_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 13:
        enriched, names = add_renal_cardiometabolic_burden_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 14:
        enriched, names = add_ratio_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 15:
        enriched, names = add_hemodynamic_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 16:
        enriched, names = add_pairwise_interaction_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 17:
        enriched, names = add_higher_risk_interaction_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 18:
        enriched, names = add_threshold_flag_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 19:
        enriched, names = add_quantile_bin_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 20:
        enriched, names = add_group_relative_sex_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 21:
        enriched, names = add_group_relative_age_band_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 22:
        enriched, names = add_percentile_rank_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 23:
        enriched, names = add_robust_scaling_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 24:
        enriched, names = add_hepatic_event_specific_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 25:
        enriched, names = add_death_specific_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 26:
        enriched, names = add_categorical_encoding_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 27:
        enriched, names = add_liver_fibrosis_composite_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 28:
        enriched, names = add_glucose_lipid_composite_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 29:
        enriched, names = add_inflammation_obesity_interaction_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 30:
        enriched, names = add_cardiac_stress_interaction_features(enriched)
        feature_names.extend(names)

    if up_to_exp >= 31:
        enriched, names = add_renal_hepatic_interaction_features(enriched)
        feature_names.extend(names)

    base_cols = baseline_feature_columns(df)
    all_feature_cols = base_cols + feature_names
    return enriched, all_feature_cols
