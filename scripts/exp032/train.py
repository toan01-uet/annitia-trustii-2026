from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import exp_shared as shared
import orjson

EXPERIMENT_NAME = "exp032_separate_vs_shared_backbone"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp032"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp031" / "outputs" / "validation_summary.json"
EXPERIMENT_NUMBER = 32
FEATURE_FAMILY = "Separate vs shared backbone per-target comparison"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_dir = OUTPUT_DIR / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing baseline summary at {BASELINE_SUMMARY_PATH}")

    shared.configure_wandb_env()
    baseline_summary = orjson.loads(BASELINE_SUMMARY_PATH.read_bytes())
    raw_df = shared.load_raw_data()
    test_df = shared.load_test_data()

    # Config A: shared backbone (cumulative features up to exp 31)
    enriched_df_shared, feature_columns_shared = shared.build_cumulative_features(raw_df, up_to_exp=31)
    enriched_test_shared, _ = shared.build_cumulative_features(test_df, up_to_exp=31)

    target_details_shared = {
        semantic_target: shared.evaluate_target_detailed(
            enriched_df_shared, enriched_test_shared, feature_columns_shared, semantic_target
        )
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    shared_combined = float(
        sum(r["summary"]["mean_roc_auc"] for r in target_details_shared.values()) / len(target_details_shared)
    )

    # Config B: separate features per target
    # Hepatic: shared + hepatic_specific (exp024 block added on top)
    enriched_hepatic, feat_hepatic = shared.build_cumulative_features(raw_df, up_to_exp=31)
    enriched_test_hepatic, _ = shared.build_cumulative_features(test_df, up_to_exp=31)
    enriched_hepatic, hepatic_extra = shared.add_hepatic_event_specific_features(enriched_hepatic)
    enriched_test_hepatic, _ = shared.add_hepatic_event_specific_features(enriched_test_hepatic)
    feat_hepatic = feat_hepatic + hepatic_extra

    # Death: shared + death_specific (exp025 block added on top)
    enriched_death, feat_death = shared.build_cumulative_features(raw_df, up_to_exp=31)
    enriched_test_death, _ = shared.build_cumulative_features(test_df, up_to_exp=31)
    enriched_death, death_extra = shared.add_death_specific_features(enriched_death)
    enriched_test_death, _ = shared.add_death_specific_features(enriched_test_death)
    feat_death = feat_death + death_extra

    hepatic_result = shared.evaluate_target_detailed(
        enriched_hepatic, enriched_test_hepatic, feat_hepatic, "risk_hepatic_event"
    )
    death_result = shared.evaluate_target_detailed(
        enriched_death, enriched_test_death, feat_death, "risk_death"
    )
    target_details_separate = {"risk_hepatic_event": hepatic_result, "risk_death": death_result}
    separate_combined = float(
        sum(r["summary"]["mean_roc_auc"] for r in target_details_separate.values()) / len(target_details_separate)
    )

    # Pick winner
    if separate_combined > shared_combined:
        winning_config = "separate"
        target_details = target_details_separate
        combined_score = separate_combined
        enriched_final = enriched_hepatic
        feature_columns_final = feat_hepatic
    else:
        winning_config = "shared"
        target_details = target_details_shared
        combined_score = shared_combined
        enriched_final = enriched_df_shared
        feature_columns_final = feature_columns_shared

    added_feature_names = [c for c in feature_columns_final if c not in shared.baseline_feature_columns(raw_df)]

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns_final) + "\n")
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "shared_combined": shared_combined,
            "separate_combined": separate_combined,
            "winning_config": winning_config,
        },
    )

    validation_summary = {
        "generated_at_utc": shared.now_utc_iso(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": str(EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
        "official_metric_confirmed": False,
        "primary_validation_metric": shared.PRIMARY_VALIDATION_METRIC,
        "metric_note": "Official challenge metric not confirmed; using ROC AUC surrogate.",
        "split_strategy": f"StratifiedKFold(n_splits={shared.DEFAULT_FOLDS}, shuffle=True, random_state={shared.DEFAULT_RANDOM_STATE})",
        "shared_combined_score": shared_combined,
        "separate_combined_score": separate_combined,
        "winning_config": winning_config,
        "feature_count": len(feature_columns_final),
        "added_feature_count": len(added_feature_names),
        "targets": {
            semantic_target: {k: v for k, v in result["summary"].items()}
            for semantic_target, result in target_details.items()
        },
        "combined_score": combined_score,
        "combined_average_precision": float(
            sum(r["summary"]["mean_average_precision"] for r in target_details.values()) / len(target_details)
        ),
        "lightgbm_params": shared.LIGHTGBM_PARAMS,
        "baseline_comparison": shared.compare_against_baseline(
            candidate_score=combined_score,
            baseline_score=float(baseline_summary["combined_score"]),
        ),
        "wandb_mode": shared.wandb_mode(),
        "wandb_api_key_present": shared.wandb_api_key_present(),
    }
    validation_summary["accepted"] = bool(validation_summary["baseline_comparison"]["accepted"])
    validation_summary["decision_reason"] = (
        f"Winning config: {winning_config}. "
        + (
            "Accepted because combined surrogate ROC AUC improved over baseline."
            if validation_summary["accepted"]
            else "Rejected because combined surrogate ROC AUC did not improve over baseline."
        )
    )
    metrics_payload = {
        "combined_score": validation_summary["combined_score"],
        "combined_average_precision": validation_summary["combined_average_precision"],
        "risk_hepatic_event_mean_roc_auc": target_details["risk_hepatic_event"]["summary"]["mean_roc_auc"],
        "risk_death_mean_roc_auc": target_details["risk_death"]["summary"]["mean_roc_auc"],
        "risk_hepatic_event_mean_average_precision": target_details["risk_hepatic_event"]["summary"]["mean_average_precision"],
        "risk_death_mean_average_precision": target_details["risk_death"]["summary"]["mean_average_precision"],
        "baseline_combined_score": baseline_summary["combined_score"],
        "combined_improvement": validation_summary["baseline_comparison"]["absolute_improvement"],
        "shared_combined_score": shared_combined,
        "separate_combined_score": separate_combined,
        "winning_config": winning_config,
    }

    shared.save_json(OUTPUT_DIR / "validation_summary.json", validation_summary)
    shared.save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    fold_scores = shared.build_fold_scores_table(target_details)
    fold_scores.write_csv(OUTPUT_DIR / "fold_scores.csv")
    oof_predictions = shared.build_oof_table(enriched_final, target_details)
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
            "feature_family": FEATURE_FAMILY,
            "target_setup": list(shared.TARGET_COLUMN_MAP.keys()),
            "feature_count": len(feature_columns_final),
            "added_feature_count": len(added_feature_names),
            "baseline_combined_score": float(baseline_summary["combined_score"]),
            "accepted": validation_summary["accepted"],
            "shared_combined_score": shared_combined,
            "separate_combined_score": separate_combined,
            "winning_config": winning_config,
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
            submission_dir / "test_predictions.csv",
            submission_dir / "submission.csv",
        ],
    )
    shared.save_json(OUTPUT_DIR / "wandb_summary.json", wandb_summary)

    print(f"{EXPERIMENT_NAME}: combined_score={validation_summary['combined_score']:.6f}")
    print(f"  shared={shared_combined:.6f}, separate={separate_combined:.6f}, winner={winning_config}")
    print(
        f"baseline={baseline_summary['combined_score']:.6f}, "
        f"improvement={validation_summary['baseline_comparison']['absolute_improvement']:.6f}, "
        f"accepted={validation_summary['accepted']}"
    )
    for semantic_target, target_result in target_details.items():
        print(
            f"  {semantic_target}: roc_auc={target_result['summary']['mean_roc_auc']:.6f}, "
            f"average_precision={target_result['summary']['mean_average_precision']:.6f}"
        )
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
