from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import orjson
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

exp035_core = importlib.import_module("exp035_core")
exp_model_utils = importlib.import_module("exp_model_utils")
shared = importlib.import_module("exp_shared")

EXPERIMENT_NAME = "exp040_exp035_exp039_focused_blend"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp040"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
CATBOOST_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp039" / "outputs" / "validation_summary.json"
FEATURE_FAMILY = "Focused two-model blend between exp035 LightGBM and refined CatBoost on the exp035 core"
WEIGHTS = [round(index * 0.05, 2) for index in range(21)]


def build_catboost_model(params: dict[str, float | int]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=float(params["learning_rate"]),
        iterations=int(params["iterations"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_strength=float(params.get("random_strength", 1.0)),
        random_seed=shared.DEFAULT_RANDOM_STATE,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def _target_metric_inputs(raw_df: pl.DataFrame, feature_columns: list[str], semantic_target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw_target = shared.TARGET_COLUMN_MAP[semantic_target]
    age_occur_col = shared.TARGET_EVENT_AGE_COLUMNS[semantic_target]
    features, target, row_ids = shared.target_ready_frame(raw_df, feature_columns, raw_target)
    kept_mask = raw_df.get_column(raw_target).is_not_null().to_numpy()
    event_times = shared._build_event_time(raw_df, raw_target, age_occur_col)[kept_mask]
    return features, target, event_times, row_ids


def _score_blend(event_indicator: np.ndarray, event_times: np.ndarray, scores: np.ndarray) -> float:
    cindex, *_ = shared.concordance_index_censored(event_indicator, event_times, scores)
    return float(cindex)


def _build_blend_target_result(
    raw_df: pl.DataFrame,
    test_ids: list[int],
    feature_columns: list[str],
    semantic_target: str,
    lgb_result: dict,
    cat_result: dict,
) -> tuple[dict, dict]:
    features, target, event_times, row_ids = _target_metric_inputs(raw_df, feature_columns, semantic_target)
    event_indicator = target.astype(bool)
    lgb_oof = lgb_result["oof_predictions"].get_column(f"{semantic_target}_oof_score").to_numpy().astype(float)
    cat_oof = cat_result["oof_predictions"].get_column(f"{semantic_target}_oof_score").to_numpy().astype(float)
    lgb_test = lgb_result["test_predictions"].get_column(f"{semantic_target}_prediction").to_numpy().astype(float)
    cat_test = cat_result["test_predictions"].get_column(f"{semantic_target}_prediction").to_numpy().astype(float)

    weight_search: list[dict[str, float]] = []
    best_weight = 0.0
    best_cindex = -1.0
    best_oof_scores = lgb_oof
    best_test_scores = lgb_test
    for cat_weight in WEIGHTS:
        blended_oof = (1.0 - cat_weight) * lgb_oof + cat_weight * cat_oof
        blended_test = (1.0 - cat_weight) * lgb_test + cat_weight * cat_test
        blended_cindex = _score_blend(event_indicator, event_times, blended_oof)
        weight_search.append({"catboost_weight": cat_weight, "lightgbm_weight": 1.0 - cat_weight, "cindex": blended_cindex})
        if blended_cindex > best_cindex:
            best_cindex = blended_cindex
            best_weight = cat_weight
            best_oof_scores = blended_oof
            best_test_scores = blended_test

    split = StratifiedKFold(
        n_splits=shared.DEFAULT_FOLDS,
        shuffle=True,
        random_state=shared.DEFAULT_RANDOM_STATE,
    )
    fold_summaries: list[dict[str, float | int]] = []
    for fold_index, (_, valid_index) in enumerate(split.split(features, target), start=1):
        fold_scores = best_oof_scores[valid_index]
        fold_cindex = _score_blend(event_indicator[valid_index], event_times[valid_index], fold_scores)
        fold_summaries.append(
            {
                "fold": fold_index,
                "cindex": fold_cindex,
                "roc_auc": float(roc_auc_score(target[valid_index], fold_scores)),
                "average_precision": float(average_precision_score(target[valid_index], fold_scores)),
                "row_count": int(len(valid_index)),
                "positive_count": int(target[valid_index].sum()),
            }
        )

    summary = {
        "semantic_target": semantic_target,
        "raw_target_column": shared.TARGET_COLUMN_MAP[semantic_target],
        "row_count": int(len(target)),
        "positive_count": int(target.sum()),
        "null_excluded_count": int(raw_df.height - len(target)),
        "mean_cindex": best_cindex,
        "mean_roc_auc": float(roc_auc_score(target, best_oof_scores)),
        "mean_average_precision": float(average_precision_score(target, best_oof_scores)),
        "selected_catboost_weight": best_weight,
        "selected_lightgbm_weight": 1.0 - best_weight,
        "folds": fold_summaries,
    }
    detail = {
        "summary": summary,
        "oof_predictions": pl.DataFrame(
            {
                shared.ID_COLUMN: row_ids,
                f"{semantic_target}_target": target.tolist(),
                f"{semantic_target}_oof_score": best_oof_scores.tolist(),
            }
        ),
        "test_predictions": pl.DataFrame(
            {
                shared.SUBMISSION_ID_COLUMN: test_ids,
                f"{semantic_target}_prediction": best_test_scores.tolist(),
            }
        ),
    }
    search_summary = {
        "selected_catboost_weight": best_weight,
        "selected_lightgbm_weight": 1.0 - best_weight,
        "weight_search": weight_search,
        "lightgbm_only_cindex": _score_blend(event_indicator, event_times, lgb_oof),
        "catboost_only_cindex": _score_blend(event_indicator, event_times, cat_oof),
    }
    return detail, search_summary


def _blend_feature_importance(lightgbm_results: dict[str, dict], catboost_results: dict[str, dict]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for source_name, result_map in (("lightgbm", lightgbm_results), ("catboost", catboost_results)):
        for target_result in result_map.values():
            frames.append(
                target_result["feature_importance"].with_columns(
                    pl.col("importance").cast(pl.Float64, strict=False),
                    pl.lit(source_name).alias("source_model"),
                )
            )
    return pl.concat(frames, how="vertical")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if not CATBOOST_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing CatBoost summary at {CATBOOST_SUMMARY_PATH}")

    shared.configure_wandb_env()
    catboost_summary = orjson.loads(CATBOOST_SUMMARY_PATH.read_bytes())
    baseline_score = float(catboost_summary["combined_score"])
    catboost_params = catboost_summary["model_params"]

    raw_df = shared.load_raw_data()
    test_df = shared.load_test_data()
    enriched_df, feature_columns, added_feature_names = exp035_core.build_selected_features(raw_df)
    enriched_test_df, _, _ = exp035_core.build_selected_features(test_df)
    test_ids = test_df.get_column(shared.SUBMISSION_ID_COLUMN).to_list()

    lightgbm_results = {
        semantic_target: shared.evaluate_target_detailed(enriched_df, enriched_test_df, feature_columns, semantic_target)
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    catboost_results = {
        semantic_target: exp_model_utils.evaluate_target_detailed_generic(
            enriched_df,
            enriched_test_df,
            feature_columns,
            semantic_target,
            model_factory=lambda params=catboost_params: build_catboost_model(params),
            importance_getter=lambda model: np.asarray(model.get_feature_importance()),
        )
        for semantic_target in shared.TARGET_COLUMN_MAP
    }

    target_details: dict[str, dict] = {}
    blend_search: dict[str, dict] = {}
    for semantic_target in shared.TARGET_COLUMN_MAP:
        detail, search_summary = _build_blend_target_result(
            enriched_df,
            test_ids,
            feature_columns,
            semantic_target,
            lightgbm_results[semantic_target],
            catboost_results[semantic_target],
        )
        target_details[semantic_target] = detail
        blend_search[semantic_target] = search_summary

    combined_score = shared.compute_combined_score(target_details)
    lightgbm_score = shared.compute_combined_score(lightgbm_results)
    catboost_score = shared.compute_combined_score(catboost_results)

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")
    shared.save_json(OUTPUT_DIR / "blend_search.json", blend_search)
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "selected_families": exp035_core.selected_family_names(),
            "added_feature_count": len(added_feature_names),
            "lightgbm_combined_score": lightgbm_score,
            "catboost_combined_score": catboost_score,
            "blended_combined_score": combined_score,
        },
    )

    validation_summary = {
        "generated_at_utc": shared.now_utc_iso(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": str(EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
        "official_metric_confirmed": True,
        "primary_validation_metric": shared.PRIMARY_VALIDATION_METRIC,
        "metric_note": "Official metric confirmed from challenge instructions: 0.3 * C-index_death + 0.7 * C-index_hepatic using concordance_index_censored.",
        "split_strategy": f"StratifiedKFold(n_splits={shared.DEFAULT_FOLDS}, shuffle=True, random_state={shared.DEFAULT_RANDOM_STATE})",
        "feature_count": len(feature_columns),
        "added_feature_count": len(added_feature_names),
        "targets": {
            semantic_target: {key: value for key, value in result["summary"].items()}
            for semantic_target, result in target_details.items()
        },
        "combined_score": combined_score,
        "combined_average_precision": float(
            sum(result["summary"]["mean_average_precision"] for result in target_details.values()) / len(target_details)
        ),
        "lightgbm_combined_score": lightgbm_score,
        "catboost_combined_score": catboost_score,
        "blend_search": blend_search,
        "baseline_comparison": shared.compare_against_baseline(candidate_score=combined_score, baseline_score=baseline_score),
        "wandb_mode": shared.wandb_mode(),
        "wandb_api_key_present": shared.wandb_api_key_present(),
    }
    validation_summary["accepted"] = bool(validation_summary["baseline_comparison"]["accepted"])
    validation_summary["decision_reason"] = (
        "Accepted because the focused LightGBM/CatBoost blend improved the official weighted C-index over the single-model CatBoost frontier."
        if validation_summary["accepted"]
        else "Rejected because the focused LightGBM/CatBoost blend did not improve the official weighted C-index over the single-model CatBoost frontier."
    )

    metrics_payload = {
        "combined_score": validation_summary["combined_score"],
        "combined_average_precision": validation_summary["combined_average_precision"],
        "risk_hepatic_event_mean_cindex": target_details["risk_hepatic_event"]["summary"]["mean_cindex"],
        "risk_hepatic_event_mean_roc_auc": target_details["risk_hepatic_event"]["summary"]["mean_roc_auc"],
        "risk_death_mean_cindex": target_details["risk_death"]["summary"]["mean_cindex"],
        "risk_death_mean_roc_auc": target_details["risk_death"]["summary"]["mean_roc_auc"],
        "risk_hepatic_event_mean_average_precision": target_details["risk_hepatic_event"]["summary"]["mean_average_precision"],
        "risk_death_mean_average_precision": target_details["risk_death"]["summary"]["mean_average_precision"],
        "baseline_combined_score": baseline_score,
        "combined_improvement": validation_summary["baseline_comparison"]["absolute_improvement"],
        "lightgbm_combined_score": lightgbm_score,
        "catboost_combined_score": catboost_score,
    }

    shared.save_json(OUTPUT_DIR / "validation_summary.json", validation_summary)
    shared.save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    shared.build_fold_scores_table(target_details).write_csv(OUTPUT_DIR / "fold_scores.csv")
    shared.build_oof_table(enriched_df, target_details).write_csv(OUTPUT_DIR / "oof_predictions.csv")
    _blend_feature_importance(lightgbm_results, catboost_results).write_csv(OUTPUT_DIR / "feature_importance.csv")
    test_predictions = shared.build_test_prediction_table(target_details)
    test_predictions.write_csv(SUBMISSION_DIR / "test_predictions.csv")
    submission = shared.build_submission_frame(target_details)
    submission.write_csv(SUBMISSION_DIR / "submission.csv")

    wandb_summary = shared.log_experiment_to_wandb(
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=EXPERIMENT_DIR,
        summary=validation_summary,
        extra_config={
            "feature_family": FEATURE_FAMILY,
            "target_setup": list(shared.TARGET_COLUMN_MAP.keys()),
            "feature_count": len(feature_columns),
            "added_feature_count": len(added_feature_names),
            "selected_families": exp035_core.selected_family_names(),
            "baseline_combined_score": baseline_score,
            "lightgbm_combined_score": lightgbm_score,
            "catboost_combined_score": catboost_score,
            "accepted": validation_summary["accepted"],
        },
        artifact_paths=[
            OUTPUT_DIR / "added_feature_names.txt",
            OUTPUT_DIR / "feature_columns.txt",
            OUTPUT_DIR / "blend_search.json",
            OUTPUT_DIR / "feature_family_summary.json",
            OUTPUT_DIR / "validation_summary.json",
            OUTPUT_DIR / "metrics.json",
            OUTPUT_DIR / "fold_scores.csv",
            OUTPUT_DIR / "oof_predictions.csv",
            OUTPUT_DIR / "feature_importance.csv",
            SUBMISSION_DIR / "test_predictions.csv",
            SUBMISSION_DIR / "submission.csv",
        ],
    )
    shared.save_json(OUTPUT_DIR / "wandb_summary.json", wandb_summary)

    print(f"{EXPERIMENT_NAME}: combined_score={validation_summary['combined_score']:.6f}")
    print(
        f"baseline={baseline_score:.6f}, improvement={validation_summary['baseline_comparison']['absolute_improvement']:.6f}, accepted={validation_summary['accepted']}"
    )
    print(f"lightgbm_only={lightgbm_score:.6f}, catboost_only={catboost_score:.6f}")
    for semantic_target in shared.TARGET_COLUMN_MAP:
        print(
            f"  {semantic_target}: catboost_weight={blend_search[semantic_target]['selected_catboost_weight']:.2f}, cindex={target_details[semantic_target]['summary']['mean_cindex']:.6f}"
        )


if __name__ == "__main__":
    main()
