from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import orjson
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

exp035_core = importlib.import_module("exp035_core")
exp_model_utils = importlib.import_module("exp_model_utils")
shared = importlib.import_module("exp_shared")

EXPERIMENT_NAME = "exp039_catboost_refinement_on_exp035_core"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp039"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp037" / "outputs" / "validation_summary.json"
FEATURE_FAMILY = "Light CatBoost parameter refinement on exp035 curated feature stack"

CANDIDATE_CONFIGS = [
    {
        "name": "exp037_baseline",
        "params": {
            "learning_rate": 0.03,
            "iterations": 300,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
        },
    },
    {
        "name": "shallower_regularized",
        "params": {
            "learning_rate": 0.02,
            "iterations": 450,
            "depth": 5,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.5,
        },
    },
    {
        "name": "deeper_balanced",
        "params": {
            "learning_rate": 0.025,
            "iterations": 360,
            "depth": 7,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.2,
        },
    },
]


def build_model(params: dict[str, float | int]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=float(params["learning_rate"]),
        iterations=int(params["iterations"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_strength=float(params["random_strength"]),
        random_seed=shared.DEFAULT_RANDOM_STATE,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def evaluate_candidate(
    raw_df,
    feature_columns: list[str],
    config_name: str,
    params: dict[str, float | int],
) -> dict:
    target_details = {
        semantic_target: exp_model_utils.evaluate_target_detailed_generic(
            raw_df,
            None,
            feature_columns,
            semantic_target,
            model_factory=lambda params=params: build_model(params),
            importance_getter=lambda model: np.asarray(model.get_feature_importance()),
        )
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    combined_score = shared.compute_combined_score(target_details)
    return {
        "config_name": config_name,
        "params": params,
        "combined_score": combined_score,
        "targets": {
            semantic_target: {
                "mean_cindex": result["summary"]["mean_cindex"],
                "mean_roc_auc": result["summary"]["mean_roc_auc"],
                "mean_average_precision": result["summary"]["mean_average_precision"],
            }
            for semantic_target, result in target_details.items()
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing baseline summary at {BASELINE_SUMMARY_PATH}")

    shared.configure_wandb_env()
    baseline_summary = orjson.loads(BASELINE_SUMMARY_PATH.read_bytes())
    raw_df = shared.load_raw_data()
    test_df = shared.load_test_data()
    enriched_df, feature_columns, added_feature_names = exp035_core.build_selected_features(raw_df)
    enriched_test_df, _, _ = exp035_core.build_selected_features(test_df)

    candidate_results = [
        evaluate_candidate(enriched_df, feature_columns, candidate["name"], candidate["params"])
        for candidate in CANDIDATE_CONFIGS
    ]
    best_candidate = max(candidate_results, key=lambda item: item["combined_score"])
    best_params = best_candidate["params"]

    target_details = {
        semantic_target: exp_model_utils.evaluate_target_detailed_generic(
            enriched_df,
            enriched_test_df,
            feature_columns,
            semantic_target,
            model_factory=lambda params=best_params: build_model(params),
            importance_getter=lambda model: np.asarray(model.get_feature_importance()),
        )
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    combined_score = shared.compute_combined_score(target_details)

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")
    shared.save_json(OUTPUT_DIR / "candidate_sweep.json", {"candidates": candidate_results, "selected_config": best_candidate})
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "selected_families": exp035_core.selected_family_names(),
            "added_feature_count": len(added_feature_names),
            "model_class": "CatBoostClassifier",
            "selected_config_name": best_candidate["config_name"],
            "selected_model_params": best_params,
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
        "model_class": "CatBoostClassifier",
        "model_params": best_params,
        "selected_config_name": best_candidate["config_name"],
        "candidate_sweep": candidate_results,
        "baseline_comparison": shared.compare_against_baseline(
            candidate_score=combined_score,
            baseline_score=float(baseline_summary["combined_score"]),
        ),
        "wandb_mode": shared.wandb_mode(),
        "wandb_api_key_present": shared.wandb_api_key_present(),
    }
    validation_summary["accepted"] = bool(validation_summary["baseline_comparison"]["accepted"])
    validation_summary["decision_reason"] = (
        "Accepted because the CatBoost refinement improved the official weighted C-index over exp037."
        if validation_summary["accepted"]
        else "Rejected because the CatBoost refinement did not improve the official weighted C-index over exp037."
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
        "baseline_combined_score": baseline_summary["combined_score"],
        "combined_improvement": validation_summary["baseline_comparison"]["absolute_improvement"],
    }

    shared.save_json(OUTPUT_DIR / "validation_summary.json", validation_summary)
    shared.save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    shared.build_fold_scores_table(target_details).write_csv(OUTPUT_DIR / "fold_scores.csv")
    shared.build_oof_table(enriched_df, target_details).write_csv(OUTPUT_DIR / "oof_predictions.csv")
    shared.combined_feature_importance_table(target_details).write_csv(OUTPUT_DIR / "feature_importance.csv")
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
            "baseline_combined_score": float(baseline_summary["combined_score"]),
            "selected_config_name": best_candidate["config_name"],
            "accepted": validation_summary["accepted"],
        },
        artifact_paths=[
            OUTPUT_DIR / "added_feature_names.txt",
            OUTPUT_DIR / "feature_columns.txt",
            OUTPUT_DIR / "candidate_sweep.json",
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
        f"baseline={baseline_summary['combined_score']:.6f}, improvement={validation_summary['baseline_comparison']['absolute_improvement']:.6f}, accepted={validation_summary['accepted']}"
    )
    print(f"selected_config={best_candidate['config_name']}")


if __name__ == "__main__":
    main()
