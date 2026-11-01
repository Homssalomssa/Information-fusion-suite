"""Classification metrics and paired spatial block-bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike


def binary_metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """Compute positive-class metrics from confusion counts."""
    tp = int(true_positive)
    fp = int(false_positive)
    fn = int(false_negative)
    iou_denominator = tp + fp + fn
    f1_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    return {
        "iou": tp / iou_denominator if iou_denominator else 0.0,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "precision": tp / precision_denominator if precision_denominator else 0.0,
        "recall": tp / recall_denominator if recall_denominator else 0.0,
    }


def paired_spatial_block_bootstrap(
    truth: ArrayLike,
    predictions: Mapping[str, ArrayLike],
    valid_mask: ArrayLike,
    *,
    block_shape: tuple[int, int] = (25, 25),
    resamples: int = 1000,
    seed: int = 20261031,
    comparisons: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """Estimate paired percentile CIs by resampling fixed spatial blocks.

    Metrics for each replicate are calculated after summing TP, FP, and FN
    over all sampled blocks, including repeated selections of the same block.
    """
    reference = np.asarray(truth, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if reference.ndim != 2 or valid.shape != reference.shape:
        raise ValueError("truth and valid_mask must be same-shaped 2D arrays.")
    if not predictions:
        raise ValueError("At least one prediction is required.")
    if block_shape[0] <= 0 or block_shape[1] <= 0:
        raise ValueError("block_shape dimensions must be positive.")
    if resamples < 1:
        raise ValueError("resamples must be positive.")
    if not np.any(valid):
        raise ValueError("valid_mask must include at least one pixel.")

    names = tuple(predictions)
    predicted_arrays: dict[str, np.ndarray] = {}
    for name, values in predictions.items():
        predicted = np.asarray(values, dtype=bool)
        if predicted.shape != reference.shape:
            raise ValueError(f"Prediction {name!r} does not match truth shape.")
        predicted_arrays[name] = predicted

    if comparisons is None:
        comparisons = {}
    for name, (candidate, baseline) in comparisons.items():
        if candidate not in predicted_arrays or baseline not in predicted_arrays:
            raise ValueError(
                f"Comparison {name!r} references an unknown prediction."
            )

    height, width = reference.shape
    block_rows = (height + block_shape[0] - 1) // block_shape[0]
    block_columns = (width + block_shape[1] - 1) // block_shape[1]
    block_ids = (
        (np.arange(height)[:, None] // block_shape[0]) * block_columns
        + np.arange(width)[None, :] // block_shape[1]
    )
    block_count = block_rows * block_columns
    active_blocks = np.flatnonzero(
        np.bincount(block_ids[valid], minlength=block_count) > 0
    )
    if active_blocks.size == 0:
        raise ValueError("valid_mask must include at least one populated spatial block.")
    counts = np.zeros((len(names), block_count, 3), dtype=np.int64)
    for index, name in enumerate(names):
        predicted = predicted_arrays[name]
        counts[index, :, 0] = np.bincount(
            block_ids[valid & reference & predicted],
            minlength=block_count,
        )
        counts[index, :, 1] = np.bincount(
            block_ids[valid & ~reference & predicted],
            minlength=block_count,
        )
        counts[index, :, 2] = np.bincount(
            block_ids[valid & reference & ~predicted],
            minlength=block_count,
        )

    metric_names = ("iou", "f1")
    metric_samples = {
        name: {metric: np.empty(resamples, dtype=np.float64) for metric in metric_names}
        for name in names
    }
    difference_samples = {
        name: {metric: np.empty(resamples, dtype=np.float64) for metric in metric_names}
        for name in comparisons
    }
    name_indices = {name: index for index, name in enumerate(names)}
    rng = np.random.Generator(np.random.PCG64(seed))

    for replicate in range(resamples):
        sampled_blocks = rng.choice(
            active_blocks,
            size=active_blocks.size,
            replace=True,
        )
        replicate_metrics: dict[str, dict[str, float]] = {}
        for name, index in name_indices.items():
            tp, fp, fn = counts[index, sampled_blocks].sum(axis=0)
            metrics = binary_metrics_from_counts(tp, fp, fn)
            replicate_metrics[name] = metrics
            for metric in metric_names:
                metric_samples[name][metric][replicate] = metrics[metric]
        for name, (candidate, baseline) in comparisons.items():
            for metric in metric_names:
                difference_samples[name][metric][replicate] = (
                    replicate_metrics[candidate][metric]
                    - replicate_metrics[baseline][metric]
                )

    def intervals(
        samples: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, tuple[float, float]]]:
        return {
            name: {
                metric: (
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                )
                for metric, values in metrics.items()
            }
            for name, metrics in samples.items()
        }

    return {
        "metrics": intervals(metric_samples),
        "differences": intervals(difference_samples),
    }
