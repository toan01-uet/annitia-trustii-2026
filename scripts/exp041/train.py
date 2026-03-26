from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import orjson
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

exp035_core = importlib.import_module("exp035_core")
exp_model_utils = importlib.import_module("exp_model_utils")
shared = importlib.import_module("exp_shared")

EXPERIMENT_NAME = "exp041_canonical_endpoint_specific_ensemble"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp041"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp040" / "outputs" / "validation_summary.json"
CATBOOST_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp039" / "outputs" / "validation_summary.json"
FEATURE_FAMILY = "Canonical endpoint-specific ensemble with hepatic CatBoost and death LightGBM on the exp035 core"


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


def build_lightgbm_model(params: dict[str, float | int | str]) -> LGBMClassifier:
    return LGBMClassifier(**params)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing baseline summary at {BASELINE_SUMMARY_PATH}")
    if not CATBOOST_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing CatBoost summary at {CATBOOST_SUMMARY_PATH}")

    shared.configure_wandb_env()
    baseline_summary = orjson.loads(BASELINE_SUMMARY_PATH.read_bytes())
    catboost_summary = orjson.loads(CATBOOST_SUMMARY_PATH.read_bytes())
    catboost_params = catboost_summary["model_params"]
    lightgbm_params = dict(shared.LIGHTGBM_PARAMS)

    raw_df = shared.load_raw_data()
    test_df = shared.load_test_data()
    enriched_df, feature_columns, added_feature_names = exp035_core.build_selected_features(raw_df)
    enriched_test_df, _, _ = exp035_core.build_selected_features(test_df)

    target_details = {
        "risk_hepatic_event": exp_model_utils.evaluate_target_detailed_generic(
            enriched_df,
            enriched_test_df,
            feature_columns,
            "risk_hepatic_event",
            model_factory=lambda: build_catboost_model(catboost_params),
            importance_getter=lambda model: np.asarray(model.get_feature_importance(), dtype=float),
        ),
        "risk_death": exp_model_utils.evaluate_target_detailed_generic(
            enriched_df,
            enriched_test_df,
            feature_columns,
            "risk_death",
            model_factory=lambda: build_lightgbm_model(lightgbm_params),
            importance_getter=lambda model: np.asarray(model.feature_importances_, dtype=float),
        ),
    }
    combined_score = shared.compute_combined_score(target_details)

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "selected_families": exp035_core.selected_family_names(),
            "added_feature_count": len(added_feature_names),
            "endpoint_model_map": {
                "risk_hepatic_event": "CatBoostClassifier",
                "risk_death": "LGBMClassifier",
            },
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
        "endpoint_model_map": {
            "risk_hepatic_event": {
                "model_class": "CatBoostClassifier",
                "model_params": catboost_params,
            },
            "risk_death": {
                "model_class": "LGBMClassifier",
                "model_params": lightgbm_params,
            },
        },
        "baseline_comparison": shared.compare_against_baseline(
            candidate_score=combined_score,
            baseline_score=float(baseline_summary["combined_score"]),
        ),
        "wandb_mode": shared.wandb_mode(),
        "wandb_api_key_present": shared.wandb_api_key_present(),
    }
    validation_summary["accepted"] = bool(validation_summary["baseline_comparison"]["accepted"])
    validation_summary["decision_reason"] = (
        "Accepted because the canonical endpoint-specific ensemble improved the official weighted C-index over exp040."
        if validation_summary["accepted"]
        else "Rejected because the canonical endpoint-specific ensemble matched or underperformed exp040 on the official weighted C-index."
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
            "accepted": validation_summary["accepted"],
        },
        artifact_paths=[
            OUTPUT_DIR / "added_feature_names.txt",
            OUTPUT_DIR / "feature_columns.txt",
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


if __name__ == "__main__":
    main()
