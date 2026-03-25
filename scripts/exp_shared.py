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
