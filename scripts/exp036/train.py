from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import orjson
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

exp035_core = importlib.import_module("exp035_core")
shared = importlib.import_module("exp_shared")

EXPERIMENT_NAME = "exp036_target_specific_zero_importance_pruning"
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "exp036"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "scripts" / "exp035" / "outputs" / "validation_summary.json"
FEATURE_FAMILY = "Target-specific fold-safe zero-importance pruning on exp035 core"


def evaluate_target_detailed_pruned(
    raw_df: pl.DataFrame,
    feature_columns: list[str],
    semantic_target: str,
) -> dict:
    raw_target = shared.TARGET_COLUMN_MAP[semantic_target]
    age_occur_col = shared.TARGET_EVENT_AGE_COLUMNS[semantic_target]
    features, target, row_ids = shared.target_ready_frame(raw_df, feature_columns, raw_target)

    kept_mask = raw_df.get_column(raw_target).is_not_null().to_numpy()
    all_event_times = shared._build_event_time(raw_df, raw_target, age_occur_col)
    event_times = all_event_times[kept_mask]
    event_indicator = target.astype(bool)

    split = StratifiedKFold(
        n_splits=shared.DEFAULT_FOLDS,
        shuffle=True,
        random_state=shared.DEFAULT_RANDOM_STATE,
    )
    oof_scores = np.zeros(len(target), dtype=np.float64)
    fold_summaries: list[dict] = []
    selected_feature_counts: list[int] = []

    for fold_index, (train_index, valid_index) in enumerate(split.split(features, target), start=1):
        pre_model = LGBMClassifier(**shared.LIGHTGBM_PARAMS)
        pre_model.fit(features[train_index], target[train_index])
        selected_mask = pre_model.feature_importances_ > 0
        if not np.any(selected_mask):
            selected_mask = np.ones(len(feature_columns), dtype=bool)
        selected_feature_counts.append(int(selected_mask.sum()))

        model = LGBMClassifier(**shared.LIGHTGBM_PARAMS)
        model.fit(features[train_index][:, selected_mask], target[train_index])
        probabilities = model.predict_proba(features[valid_index][:, selected_mask])[:, 1]
        oof_scores[valid_index] = probabilities
        fold_ci, *_ = shared.concordance_index_censored(
            event_indicator[valid_index],
            event_times[valid_index],
            probabilities,
        )
        fold_summaries.append(
            {
                "fold": fold_index,
                "cindex": float(fold_ci),
                "roc_auc": float(roc_auc_score(target[valid_index], probabilities)),
                "average_precision": float(average_precision_score(target[valid_index], probabilities)),
                "row_count": int(len(valid_index)),
                "positive_count": int(target[valid_index].sum()),
                "selected_feature_count": int(selected_mask.sum()),
            }
        )

    pre_full_model = LGBMClassifier(**shared.LIGHTGBM_PARAMS)
    pre_full_model.fit(features, target)
    selected_mask_full = pre_full_model.feature_importances_ > 0
    if not np.any(selected_mask_full):
        selected_mask_full = np.ones(len(feature_columns), dtype=bool)
    full_feature_names = [name for name, keep in zip(feature_columns, selected_mask_full) if keep]

    full_model = LGBMClassifier(**shared.LIGHTGBM_PARAMS)
    full_model.fit(features[:, selected_mask_full], target)
    feature_importance = pl.DataFrame(
        {
            "feature": full_feature_names,
            "importance": full_model.feature_importances_.tolist(),
            "target": [semantic_target] * len(full_feature_names),
        }
    ).sort("importance", descending=True)

    oof_ci, *_ = shared.concordance_index_censored(event_indicator, event_times, oof_scores)
    return {
        "summary": {
            "semantic_target": semantic_target,
            "raw_target_column": raw_target,
            "row_count": int(len(target)),
            "positive_count": int(target.sum()),
            "null_excluded_count": int(raw_df.height - len(target)),
            "mean_cindex": float(oof_ci),
            "mean_roc_auc": float(roc_auc_score(target, oof_scores)),
            "mean_average_precision": float(average_precision_score(target, oof_scores)),
            "mean_selected_feature_count": float(np.mean(selected_feature_counts)),
            "full_model_selected_feature_count": int(selected_mask_full.sum()),
            "folds": fold_summaries,
        },
        "feature_importance": feature_importance,
        "selected_features": full_feature_names,
        "oof_predictions": pl.DataFrame(
            {
                shared.ID_COLUMN: row_ids,
                f"{semantic_target}_target": target.tolist(),
                f"{semantic_target}_oof_score": oof_scores.tolist(),
            }
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASELINE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing baseline summary at {BASELINE_SUMMARY_PATH}")

    shared.configure_wandb_env()
    baseline_summary = orjson.loads(BASELINE_SUMMARY_PATH.read_bytes())
    raw_df = shared.load_raw_data()

    enriched_df, feature_columns, added_feature_names = exp035_core.build_selected_features(raw_df)

    target_details = {
        semantic_target: evaluate_target_detailed_pruned(enriched_df, feature_columns, semantic_target)
        for semantic_target in shared.TARGET_COLUMN_MAP
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
            "risk_hepatic_event_selected_feature_count": target_details["risk_hepatic_event"]["summary"]["full_model_selected_feature_count"],
            "risk_death_selected_feature_count": target_details["risk_death"]["summary"]["full_model_selected_feature_count"],
        },
    )

    selected_union = sorted(
        set(target_details["risk_hepatic_event"]["selected_features"]) | set(target_details["risk_death"]["selected_features"])
    )
    (OUTPUT_DIR / "selected_features_union.txt").write_text("\n".join(selected_union) + "\n")

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
        "Accepted because fold-safe zero-importance target-specific pruning improved the official weighted C-index over exp035."
        if validation_summary["accepted"]
        else "Rejected because fold-safe zero-importance target-specific pruning did not improve the official weighted C-index over exp035."
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
            "validation_only": True,
        },
        artifact_paths=[
            OUTPUT_DIR / "added_feature_names.txt",
            OUTPUT_DIR / "feature_columns.txt",
            OUTPUT_DIR / "feature_family_summary.json",
            OUTPUT_DIR / "selected_features_union.txt",
            OUTPUT_DIR / "validation_summary.json",
            OUTPUT_DIR / "metrics.json",
            OUTPUT_DIR / "fold_scores.csv",
            OUTPUT_DIR / "oof_predictions.csv",
            OUTPUT_DIR / "feature_importance.csv",
        ],
    )
    shared.save_json(OUTPUT_DIR / "wandb_summary.json", wandb_summary)

    print(f"{EXPERIMENT_NAME}: combined_score={validation_summary['combined_score']:.6f}")
    print(
        f"baseline={baseline_summary['combined_score']:.6f}, improvement={validation_summary['baseline_comparison']['absolute_improvement']:.6f}, accepted={validation_summary['accepted']}"
    )
    for semantic_target, target_result in target_details.items():
        print(
            f"  {semantic_target}: cindex={target_result['summary']['mean_cindex']:.6f}, roc_auc={target_result['summary']['mean_roc_auc']:.6f}, selected={target_result['summary']['full_model_selected_feature_count']}"
        )


if __name__ == "__main__":
    main()
