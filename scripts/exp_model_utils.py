from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import exp_shared as shared


ModelFactory = Callable[[], Any]
ImportanceGetter = Callable[[Any], np.ndarray]


def evaluate_target_detailed_generic(
    raw_df: pl.DataFrame,
    test_df: pl.DataFrame | None,
    feature_columns: list[str],
    semantic_target: str,
    model_factory: ModelFactory,
    importance_getter: ImportanceGetter,
) -> dict[str, Any]:
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
    fold_summaries: list[dict[str, Any]] = []

    test_features: np.ndarray | None = None
    submission_ids: list[int] | None = None
    if test_df is not None:
        test_features, submission_ids = shared.inference_ready_matrix(test_df, feature_columns)

    for fold_index, (train_index, valid_index) in enumerate(split.split(features, target), start=1):
        model = model_factory()
        model.fit(features[train_index], target[train_index])
        probabilities = model.predict_proba(features[valid_index])[:, 1]
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
            }
        )

    full_model = model_factory()
    full_model.fit(features, target)
    feature_importance = pl.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance_getter(full_model).tolist(),
            "target": [semantic_target] * len(feature_columns),
        }
    ).sort("importance", descending=True)

    oof_ci, *_ = shared.concordance_index_censored(event_indicator, event_times, oof_scores)
    detailed_payload: dict[str, Any] = {
        "summary": {
            "semantic_target": semantic_target,
            "raw_target_column": raw_target,
            "row_count": int(len(target)),
            "positive_count": int(target.sum()),
            "null_excluded_count": int(raw_df.height - len(target)),
            "mean_cindex": float(oof_ci),
            "mean_roc_auc": float(roc_auc_score(target, oof_scores)),
            "mean_average_precision": float(average_precision_score(target, oof_scores)),
            "folds": fold_summaries,
        },
        "feature_importance": feature_importance,
        "oof_predictions": pl.DataFrame(
            {
                shared.ID_COLUMN: row_ids,
                f"{semantic_target}_target": target.tolist(),
                f"{semantic_target}_oof_score": oof_scores.tolist(),
            }
        ),
    }
    if test_features is not None and submission_ids is not None:
        test_probabilities = full_model.predict_proba(test_features)[:, 1]
        detailed_payload["test_predictions"] = pl.DataFrame(
            {
                shared.SUBMISSION_ID_COLUMN: submission_ids,
                f"{semantic_target}_prediction": test_probabilities.tolist(),
            }
        )
    return detailed_payload
