from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import exp_shared as shared

EXPERIMENT_NAME = "exp033_isotonic_calibration"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp033"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp032" / "outputs" / "validation_summary.json"
EXPERIMENT_NUMBER = 33
FEATURE_FAMILY = "Isotonic regression calibration on OOF predictions"


def _calibrate_oof_leave_one_fold_out(
    oof_scores: np.ndarray,
    labels: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Apply leave-one-fold-out isotonic calibration.

    For each fold, fit IsotonicRegression on the OOF scores from all other folds,
    then predict on this fold.
    """
    calibrated = np.empty_like(oof_scores, dtype=float)
    for val_train_idx, val_val_idx in fold_indices:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_scores[val_train_idx], labels[val_train_idx])
        calibrated[val_val_idx] = iso.predict(oof_scores[val_val_idx])
    return calibrated


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

    enriched_df, feature_columns = shared.build_cumulative_features(raw_df, up_to_exp=31)
    enriched_test_df, _ = shared.build_cumulative_features(test_df, up_to_exp=31)
    added_feature_names = [c for c in feature_columns if c not in shared.baseline_feature_columns(raw_df)]

    target_details = {
        semantic_target: shared.evaluate_target_detailed(enriched_df, enriched_test_df, feature_columns, semantic_target)
        for semantic_target in shared.TARGET_COLUMN_MAP
    }

    raw_combined = float(
        sum(r["summary"]["mean_roc_auc"] for r in target_details.values()) / len(target_details)
    )

    # Post-hoc global isotonic calibration comparison (optimistic but illustrative)
    calibration_results: dict[str, dict] = {}
    for semantic_target, target_col in shared.TARGET_COLUMN_MAP.items():
        result = target_details[semantic_target]
        oof_frame = shared.build_oof_table(enriched_df, {semantic_target: result})
        oof_scores_col = f"{semantic_target}_oof_score"
        if oof_scores_col not in oof_frame.columns:
            calibration_results[semantic_target] = {
                "raw_roc_auc": result["summary"]["mean_roc_auc"],
                "calibrated_roc_auc": result["summary"]["mean_roc_auc"],
                "calibration_gain": 0.0,
            }
            continue

        import polars as pl

        oof_scores = oof_frame.get_column(oof_scores_col).to_numpy().astype(float)
        labels_series = enriched_df.get_column(target_col)
        labels = labels_series.to_numpy().astype(float)

        # Filter to non-null labels
        valid_mask = ~np.isnan(labels) & ~np.isnan(oof_scores)
        oof_valid = oof_scores[valid_mask]
        labels_valid = labels[valid_mask]

        raw_auc = float(roc_auc_score(labels_valid, oof_valid)) if len(np.unique(labels_valid)) > 1 else 0.5

        # Global in-sample isotonic calibration (upper bound comparison)
        iso_global = IsotonicRegression(out_of_bounds="clip")
        iso_global.fit(oof_valid, labels_valid)
        cal_scores_global = iso_global.predict(oof_valid)
        cal_auc_global = float(roc_auc_score(labels_valid, cal_scores_global)) if len(np.unique(labels_valid)) > 1 else 0.5

        calibration_results[semantic_target] = {
            "raw_roc_auc": raw_auc,
            "calibrated_roc_auc_global_insample": cal_auc_global,
            "calibration_gain_global": cal_auc_global - raw_auc,
        }

    calibrated_combined = float(
        sum(v.get("calibrated_roc_auc_global_insample", v["raw_roc_auc"]) for v in calibration_results.values())
        / len(calibration_results)
    )

    # Use raw combined as the real score (calibration is post-hoc comparison only)
    combined_score = raw_combined

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "added_feature_count": len(added_feature_names),
            "raw_combined_score": raw_combined,
            "calibrated_combined_score_global_insample": calibrated_combined,
            "calibration_results": calibration_results,
            "note": (
                "Calibrated scores use global in-sample isotonic regression (optimistic upper bound). "
                "The official combined_score uses raw OOF AUC."
            ),
        },
    )

    validation_summary = {
        "generated_at_utc": shared.now_utc_iso(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": str(EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
        "official_metric_confirmed": False,
        "primary_validation_metric": shared.PRIMARY_VALIDATION_METRIC,
        "metric_note": "Official challenge metric not confirmed; using ROC AUC surrogate. Calibration explored as post-hoc analysis.",
        "split_strategy": f"StratifiedKFold(n_splits={shared.DEFAULT_FOLDS}, shuffle=True, random_state={shared.DEFAULT_RANDOM_STATE})",
        "feature_count": len(feature_columns),
        "added_feature_count": len(added_feature_names),
        "targets": {
            semantic_target: {k: v for k, v in result["summary"].items()}
            for semantic_target, result in target_details.items()
        },
        "combined_score": combined_score,
        "combined_average_precision": float(
            sum(r["summary"]["mean_average_precision"] for r in target_details.values()) / len(target_details)
        ),
        "raw_combined_score": raw_combined,
        "calibrated_combined_score_global_insample": calibrated_combined,
        "calibration_results": calibration_results,
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
        "Accepted because combined surrogate ROC AUC improved over baseline."
        if validation_summary["accepted"]
        else "Rejected because combined surrogate ROC AUC did not improve over baseline."
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
        "raw_combined_score": raw_combined,
        "calibrated_combined_score_global_insample": calibrated_combined,
    }

    shared.save_json(OUTPUT_DIR / "validation_summary.json", validation_summary)
    shared.save_json(OUTPUT_DIR / "metrics.json", metrics_payload)
    fold_scores = shared.build_fold_scores_table(target_details)
    fold_scores.write_csv(OUTPUT_DIR / "fold_scores.csv")
    oof_predictions = shared.build_oof_table(enriched_df, target_details)
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
            "feature_count": len(feature_columns),
            "added_feature_count": len(added_feature_names),
            "baseline_combined_score": float(baseline_summary["combined_score"]),
            "accepted": validation_summary["accepted"],
            "raw_combined_score": raw_combined,
            "calibrated_combined_score_global_insample": calibrated_combined,
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
    print(f"  raw={raw_combined:.6f}, calibrated_insample={calibrated_combined:.6f}")
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
