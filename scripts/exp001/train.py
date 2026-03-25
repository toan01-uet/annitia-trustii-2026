from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import exp_shared as shared

EXPERIMENT_NAME = "exp001_baseline_raw_features"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp001"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_dir = OUTPUT_DIR / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    shared.configure_wandb_env()
    raw_df = shared.load_raw_data()
    test_df = shared.load_test_data()
    schema_audit = shared.build_schema_audit(raw_df)
    shared.save_json(OUTPUT_DIR / "schema_audit.json", schema_audit)
    shared.save_json(OUTPUT_DIR / "raw_to_semantic_map.json", shared.raw_to_semantic_map())

    feature_columns = shared.baseline_feature_columns(raw_df)
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")

    target_details = {
        semantic_target: shared.evaluate_target_detailed(raw_df, test_df, feature_columns, semantic_target)
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    validation_summary = {
        "generated_at_utc": shared.now_utc_iso(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": str(EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
        "official_metric_confirmed": False,
        "primary_validation_metric": shared.PRIMARY_VALIDATION_METRIC,
        "metric_note": "Official challenge metric not confirmed from accessible local materials; using ROC AUC as a ranking-oriented surrogate.",
        "split_strategy": f"StratifiedKFold(n_splits={shared.DEFAULT_FOLDS}, shuffle=True, random_state={shared.DEFAULT_RANDOM_STATE})",
        "feature_count": len(feature_columns),
        "targets": {
            semantic_target: {
                key: value
                for key, value in result["summary"].items()
            }
            for semantic_target, result in target_details.items()
        },
        "combined_score": float(
            sum(result["summary"]["mean_roc_auc"] for result in target_details.values()) / len(target_details)
        ),
        "combined_average_precision": float(
            sum(result["summary"]["mean_average_precision"] for result in target_details.values()) / len(target_details)
        ),
        "lightgbm_params": shared.LIGHTGBM_PARAMS,
        "accepted": True,
        "decision_reason": "First runnable baseline for the reset ANNITIA workspace.",
        "wandb_mode": shared.wandb_mode(),
        "wandb_api_key_present": shared.wandb_api_key_present(),
    }
    metrics_payload = {
        "combined_score": validation_summary["combined_score"],
        "combined_average_precision": validation_summary["combined_average_precision"],
        "risk_hepatic_event_mean_roc_auc": target_details["risk_hepatic_event"]["summary"]["mean_roc_auc"],
        "risk_death_mean_roc_auc": target_details["risk_death"]["summary"]["mean_roc_auc"],
        "risk_hepatic_event_mean_average_precision": target_details["risk_hepatic_event"]["summary"]["mean_average_precision"],
        "risk_death_mean_average_precision": target_details["risk_death"]["summary"]["mean_average_precision"],
    }

    shared.save_json(OUTPUT_DIR / "validation_summary.json", validation_summary)
    shared.save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    fold_scores = shared.build_fold_scores_table(target_details)
    fold_scores.write_csv(OUTPUT_DIR / "fold_scores.csv")
    oof_predictions = shared.build_oof_table(raw_df, target_details)
    oof_predictions.write_csv(OUTPUT_DIR / "oof_predictions.csv")
    test_predictions = shared.build_test_prediction_table(target_details)
    test_predictions.write_csv(submission_dir / "test_predictions.csv")
    submission = shared.build_submission_frame(target_details)
    submission.write_csv(submission_dir / "submission.csv")
    feature_importance = shared.combined_feature_importance_table(target_details)
    feature_importance.write_csv(OUTPUT_DIR / "feature_importance.csv")

    wandb_summary = shared.log_experiment_to_wandb(
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=EXPERIMENT_DIR,
        summary=validation_summary,
        extra_config={
            "feature_family": "baseline_raw_features",
            "target_setup": list(shared.TARGET_COLUMN_MAP.keys()),
            "feature_count": len(feature_columns),
        },
        artifact_paths=[
            OUTPUT_DIR / "schema_audit.json",
            OUTPUT_DIR / "raw_to_semantic_map.json",
            OUTPUT_DIR / "feature_columns.txt",
            OUTPUT_DIR / "validation_summary.json",
            OUTPUT_DIR / "metrics.json",
            OUTPUT_DIR / "fold_scores.csv",
            OUTPUT_DIR / "oof_predictions.csv",
            OUTPUT_DIR / "feature_importance.csv",
            submission_dir / "test_predictions.csv",
            submission_dir / "submission.csv",
        ],
    )
    shared.save_json(OUTPUT_DIR / "wandb_summary.json", wandb_summary)

    print(f"{EXPERIMENT_NAME}: combined_score={validation_summary['combined_score']:.6f}")
    for semantic_target, target_result in target_details.items():
        print(
            f"  {semantic_target}: roc_auc={target_result['summary']['mean_roc_auc']:.6f}, "
            f"average_precision={target_result['summary']['mean_average_precision']:.6f}"
        )
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
