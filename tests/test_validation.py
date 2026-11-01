import numpy as np
import pytest

from possibility_fusion.core.validation import (
    binary_metrics_from_counts,
    paired_spatial_block_bootstrap,
)


def test_binary_metrics_from_counts():
    assert binary_metrics_from_counts(3, 1, 2) == {
        "iou": pytest.approx(0.5),
        "f1": pytest.approx(0.6666666666666666),
        "precision": pytest.approx(0.75),
        "recall": pytest.approx(0.6),
    }


def test_paired_block_bootstrap_is_deterministic_and_pairs_comparisons():
    truth = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    predictions = {
        "fused": truth.copy(),
        "optical": np.zeros_like(truth),
        "sar": truth.copy(),
    }
    valid = np.ones_like(truth)
    options = {
        "block_shape": (2, 2),
        "resamples": 32,
        "seed": 19,
        "comparisons": {"fused_minus_optical": ("fused", "optical")},
    }
    first = paired_spatial_block_bootstrap(
        truth, predictions, valid, **options
    )
    second = paired_spatial_block_bootstrap(
        truth, predictions, valid, **options
    )

    assert first == second
    assert first["metrics"]["fused"]["iou"] == (1.0, 1.0)
    assert first["metrics"]["fused"]["f1"] == (1.0, 1.0)
    assert first["differences"]["fused_minus_optical"]["iou"] == (1.0, 1.0)
    assert first["differences"]["fused_minus_optical"]["f1"] == (1.0, 1.0)


def test_block_bootstrap_rejects_mismatched_prediction_shape():
    with pytest.raises(ValueError, match="does not match truth shape"):
        paired_spatial_block_bootstrap(
            np.ones((2, 2), dtype=bool),
            {"bad": np.ones((3, 2), dtype=bool)},
            np.ones((2, 2), dtype=bool),
            resamples=2,
        )
