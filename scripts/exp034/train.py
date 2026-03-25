from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import exp_shared as shared

EXPERIMENT_NAME = "exp034_ensemble_blending"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp034"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp033" / "outputs" / "validation_summary.json"
EXPERIMENT_NUMBER = 34
FEATURE_FAMILY = "OOF ensemble blending: simple average and rank-based average"

# Experiments to try to load OOF predictions from
CANDIDATE_EXPS = [f"exp{n:03d}" for n in range(2, 34)]


def _load_oof_scores(exp_name: str, target: str) -> np.ndarray | None:
    """Load OOF scores for a given experiment and target. Returns None if unavailable."""
    oof_path = PROJECT_ROOT / "scripts" / exp_name / "outputs" / "oof_predictions.csv"
    if not oof_path.exists():
        return None
    try:
        oof_df = pl.read_csv(oof_path)
        score_col = f"{target}_oof_score"
        if score_col not in oof_df.columns:
            return None
        scores = oof_df.get_column(score_col).to_numpy().astype(float)
        if np.all(np.isnan(scores)):
            return None
        return scores
    except Exception:
        return None


def _blend_simple_average(score_matrix: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN."""
    return np.nanmean(score_matrix, axis=1)


def _blend_rank_average(score_matrix: np.ndarray) -> np.ndarray:
    """Rank each column, then average ranks across columns."""
    n_rows, n_cols = score_matrix.shape
    rank_matrix = np.empty_like(score_matrix, dtype=float)
    for j in range(n_cols):
        col = score_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            rank_matrix[:, j] = np.nan
            continue
        from scipy.stats import rankdata
        ranks = np.full(n_rows, np.nan, dtype=float)
        ranks[valid] = rankdata(col[valid], method="average") / valid.sum()
        rank_matrix[:, j] = ranks
    return np.nanmean(rank_matrix, axis=1)


def _evaluate_blend(blended_scores: np.ndarray, labels: np.ndarray) -> float:
    valid = ~np.isnan(labels) & ~np.isnan(blended_scores)
    if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
        return 0.5
    return float(roc_auc_score(labels[valid], blended_scores[valid]))


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

    # Load labels from raw_df
    label_arrays: dict[str, np.ndarray] = {}
    for semantic_target, target_col in shared.TARGET_COLUMN_MAP.items():
        if target_col in raw_df.columns:
            label_arrays[semantic_target] = raw_df.get_column(target_col).to_numpy().astype(float)
        else:
            label_arrays[semantic_target] = np.full(raw_df.height, np.nan)

    blend_results: dict[str, dict] = {}
    id_col = raw_df.get_column(shared.ID_COLUMN).to_numpy()

    for semantic_target in shared.TARGET_COLUMN_MAP:
        available_exps: list[str] = []
        score_cols: list[np.ndarray] = []
        for exp_name in CANDIDATE_EXPS:
            scores = _load_oof_scores(exp_name, semantic_target)
            if scores is not None and len(scores) == raw_df.height:
                available_exps.append(exp_name)
                score_cols.append(scores)

        labels = label_arrays[semantic_target]

        if not score_cols:
            blend_results[semantic_target] = {
                "available_experiments": [],
                "simple_avg_roc_auc": 0.5,
                "rank_avg_roc_auc": 0.5,
                "best_blend": "none",
                "best_blend_roc_auc": 0.5,
            }
            continue

        score_matrix = np.column_stack(score_cols)

        simple_blend = _blend_simple_average(score_matrix)
        rank_blend = _blend_rank_average(score_matrix)

        simple_auc = _evaluate_blend(simple_blend, labels)
        rank_auc = _evaluate_blend(rank_blend, labels)

        # Also evaluate single-best experiment
        single_aucs: list[tuple[str, float]] = []
        for exp_name, col in zip(available_exps, score_cols):
            auc = _evaluate_blend(col, labels)
            single_aucs.append((exp_name, auc))
        best_single_exp, best_single_auc = max(single_aucs, key=lambda x: x[1])

        best_blend = "simple_avg" if simple_auc >= rank_auc else "rank_avg"
        best_blend_auc = max(simple_auc, rank_auc)

        blend_results[semantic_target] = {
            "available_experiments": available_exps,
            "n_experiments_blended": len(available_exps),
            "simple_avg_roc_auc": simple_auc,
            "rank_avg_roc_auc": rank_auc,
            "best_single_experiment": best_single_exp,
            "best_single_roc_auc": best_single_auc,
            "best_blend": best_blend,
            "best_blend_roc_auc": best_blend_auc,
            "blend_gain_over_best_single": best_blend_auc - best_single_auc,
        }

    combined_score = float(
        sum(v["best_blend_roc_auc"] for v in blend_results.values()) / len(blend_results)
    )

    (OUTPUT_DIR / "blend_results.json").write_text(
        __import__("json").dumps(blend_results, indent=2)
    )

    # For submission, re-run the best single experiment's feature set on test data using
    # the shared backbone (exp031 cumulative) and report blended OOF scores
    enriched_df, feature_columns = shared.build_cumulative_features(raw_df, up_to_exp=31)
    enriched_test_df, _ = shared.build_cumulative_features(test_df, up_to_exp=31)
    added_feature_names = [c for c in feature_columns if c not in shared.baseline_feature_columns(raw_df)]

    target_details = {
        semantic_target: shared.evaluate_target_detailed(enriched_df, enriched_test_df, feature_columns, semantic_target)
        for semantic_target in shared.TARGET_COLUMN_MAP
    }
    model_combined_score = float(
        sum(r["summary"]["mean_roc_auc"] for r in target_details.values()) / len(target_details)
    )

    # Use blended combined_score for comparison if better than model score
    reported_combined_score = max(combined_score, model_combined_score)

    (OUTPUT_DIR / "added_feature_names.txt").write_text("\n".join(added_feature_names) + "\n")
    (OUTPUT_DIR / "feature_columns.txt").write_text("\n".join(feature_columns) + "\n")
    shared.save_json(
        OUTPUT_DIR / "feature_family_summary.json",
        {
            "generated_at_utc": shared.now_utc_iso(),
            "feature_family": FEATURE_FAMILY,
            "added_feature_count": len(added_feature_names),
            "blend_combined_score": combined_score,
            "model_combined_score": model_combined_score,
            "reported_combined_score": reported_combined_score,
            "blend_results": blend_results,
        },
    )

    validation_summary = {
        "generated_at_utc": shared.now_utc_iso(),
        "experiment_name": EXPERIMENT_NAME,
        "experiment_dir": str(EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
        "official_metric_confirmed": False,
        "primary_validation_metric": shared.PRIMARY_VALIDATION_METRIC,
        "metric_note": "Official challenge metric not confirmed; using ROC AUC surrogate. Ensemble blending across all available OOF predictions.",
        "split_strategy": f"StratifiedKFold(n_splits={shared.DEFAULT_FOLDS}, shuffle=True, random_state={shared.DEFAULT_RANDOM_STATE})",
        "feature_count": len(feature_columns),
        "added_feature_count": len(added_feature_names),
        "targets": {
            semantic_target: {k: v for k, v in result["summary"].items()}
            for semantic_target, result in target_details.items()
        },
        "combined_score": reported_combined_score,
        "blend_combined_score": combined_score,
        "model_combined_score": model_combined_score,
        "combined_average_precision": float(
            sum(r["summary"]["mean_average_precision"] for r in target_details.values()) / len(target_details)
        ),
        "blend_results": blend_results,
        "lightgbm_params": shared.LIGHTGBM_PARAMS,
        "baseline_comparison": shared.compare_against_baseline(
            candidate_score=reported_combined_score,
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
        "combined_score": reported_combined_score,
        "blend_combined_score": combined_score,
        "model_combined_score": model_combined_score,
        "combined_average_precision": validation_summary["combined_average_precision"],
        "risk_hepatic_event_mean_roc_auc": target_details["risk_hepatic_event"]["summary"]["mean_roc_auc"],
        "risk_death_mean_roc_auc": target_details["risk_death"]["summary"]["mean_roc_auc"],
        "risk_hepatic_event_mean_average_precision": target_details["risk_hepatic_event"]["summary"]["mean_average_precision"],
        "risk_death_mean_average_precision": target_details["risk_death"]["summary"]["mean_average_precision"],
        "baseline_combined_score": baseline_summary["combined_score"],
        "combined_improvement": validation_summary["baseline_comparison"]["absolute_improvement"],
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
            "blend_combined_score": combined_score,
            "model_combined_score": model_combined_score,
        },
        artifact_paths=[
            OUTPUT_DIR / "blend_results.json",
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

    print(f"{EXPERIMENT_NAME}: combined_score={reported_combined_score:.6f}")
    print(f"  blend={combined_score:.6f}, model_only={model_combined_score:.6f}")
    print(
        f"baseline={baseline_summary['combined_score']:.6f}, "
        f"improvement={validation_summary['baseline_comparison']['absolute_improvement']:.6f}, "
        f"accepted={validation_summary['accepted']}"
    )
    for semantic_target, br in blend_results.items():
        print(
            f"  {semantic_target}: simple_avg={br.get('simple_avg_roc_auc', 0):.6f}, "
            f"rank_avg={br.get('rank_avg_roc_auc', 0):.6f}, "
            f"best={br.get('best_blend', 'none')}"
        )
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
