"""Tests for possibility normalization, discounting, fusion, and decisions."""

import numpy as np
import pytest

from dst_fusion.core.possibility import (
    adaptive_fusion,
    conjunctive_fusion,
    decide,
    discount,
    disjunctive_fusion,
    normalize_possibility,
)


def pi_pair(burned: float, unburned: float) -> np.ndarray:
    """Make a one-pixel possibility distribution in [Burned, Unburned] order."""
    return np.array([[[burned]], [[unburned]]], dtype=np.float32)


def assert_normalized(pi: np.ndarray) -> None:
    valid = np.all(np.isfinite(pi), axis=0)
    np.testing.assert_allclose(np.max(pi[:, valid], axis=0), 1.0)


def test_normalize_memberships_to_possibility_distribution():
    memberships = np.array([[[0.25]], [[0.5]]], dtype=np.float32)

    result = normalize_possibility(memberships)

    np.testing.assert_allclose(result[:, 0, 0], [0.5, 1.0])
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("fusion", "expected"),
    [
        (adaptive_fusion, [1.0, 0.2]),
        (conjunctive_fusion, [1.0, 0.2]),
        (disjunctive_fusion, [1.0, 0.4]),
    ],
)
def test_sources_agree(fusion, expected):
    result, conflict = fusion(pi_pair(1.0, 0.2), pi_pair(1.0, 0.4))

    np.testing.assert_allclose(result[:, 0, 0], expected)
    assert conflict[0, 0] == pytest.approx(0.0)
    assert_normalized(result)


@pytest.mark.parametrize(
    "fusion",
    [adaptive_fusion, conjunctive_fusion, disjunctive_fusion],
)
def test_sources_conflict_and_report_conflict(fusion):
    pi1 = pi_pair(1.0, 0.1)
    pi2 = pi_pair(0.1, 1.0)

    result, conflict = fusion(pi1, pi2)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 1.0])
    assert conflict[0, 0] == pytest.approx(0.9)
    assert_normalized(result)


def test_adaptive_rule_matches_hand_computed_partial_conflict():
    pi1 = pi_pair(1.0, 0.3)
    pi2 = pi_pair(0.5, 1.0)

    result, conflict = adaptive_fusion(pi1, pi2)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 0.6])
    assert conflict[0, 0] == pytest.approx(0.5)


def test_full_conflict_decision_is_undecided():
    fused, _ = adaptive_fusion(pi_pair(1.0, 0.1), pi_pair(0.1, 1.0))

    classes, necessity = decide(fused)

    assert classes[0, 0] == 2
    assert necessity[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("fusion", "expected"),
    [
        (adaptive_fusion, [0.2, 1.0]),
        (conjunctive_fusion, [0.2, 1.0]),
        (disjunctive_fusion, [1.0, 1.0]),
    ],
)
def test_total_ignorance_source(fusion, expected):
    result, _ = fusion(pi_pair(1.0, 1.0), pi_pair(0.2, 1.0))

    np.testing.assert_allclose(result[:, 0, 0], expected)
    assert_normalized(result)


def test_discount_alpha_zero_returns_total_ignorance():
    result = discount(pi_pair(0.8, 1.0), 0.0)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 1.0])


def test_discount_alpha_one_preserves_input():
    possibility = pi_pair(0.8, 1.0)

    result = discount(possibility, 1.0)

    np.testing.assert_array_equal(result, possibility)


def test_raster_reliability_discount_and_nodata():
    possibility = np.array(
        [
            [[0.8, 0.8]],
            [[1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    alpha = np.array([[0.0, 0.5]], dtype=np.float32)

    result = discount(possibility, alpha)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 1.0])
    np.testing.assert_allclose(result[:, 0, 1], [0.8, 1.0])


@pytest.mark.parametrize(
    "fusion",
    [adaptive_fusion, conjunctive_fusion, disjunctive_fusion],
)
def test_near_zero_intersection_is_finite_normalized_and_conflicting(fusion):
    pi1 = pi_pair(1.0, 1e-15)
    pi2 = pi_pair(1e-15, 1.0)

    result, conflict = fusion(pi1, pi2)

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[:, 0, 0], [1.0, 1.0])
    assert conflict[0, 0] == pytest.approx(1.0)
    assert_normalized(result)


@pytest.mark.parametrize(
    "fusion",
    [adaptive_fusion, conjunctive_fusion, disjunctive_fusion],
)
def test_nodata_in_either_source_propagates_to_all_fusion_outputs(fusion):
    pi1 = np.array(
        [
            [[1.0, np.nan]],
            [[0.2, 1.0]],
        ],
        dtype=np.float32,
    )
    pi2 = np.array(
        [
            [[1.0, 1.0]],
            [[0.4, 1.0]],
        ],
        dtype=np.float32,
    )

    result, conflict = fusion(pi1, pi2)

    assert np.all(np.isnan(result[:, 0, 1]))
    assert np.isnan(conflict[0, 1])
    assert_normalized(result)


def test_nodata_in_alpha_propagates_through_discounting():
    possibility = np.ones((2, 1, 2), dtype=np.float32)
    alpha = np.array([[1.0, np.nan]], dtype=np.float32)

    result = discount(possibility, alpha)

    assert np.all(np.isnan(result[:, 0, 1]))


def test_decision_returns_class_necessity_and_nodata():
    possibility = np.array(
        [
            [[1.0, 1.0, 1.0, np.nan]],
            [[0.2, 1.0, 0.8, 1.0]],
        ],
        dtype=np.float32,
    )

    classes, necessity = decide(possibility)

    np.testing.assert_array_equal(classes[0, :3], [0, 2, 0])
    np.testing.assert_allclose(necessity[0, :3], [0.8, 0.0, 0.2])
    assert classes.dtype == np.int8
    assert classes[0, 3] == -1
    assert np.isnan(necessity[0, 3])


def test_decision_returns_undecided_below_configured_necessity_threshold():
    possibility = pi_pair(1.0, 0.7)

    classes, necessity = decide(possibility, necessity_threshold=0.4)

    assert classes[0, 0] == 2
    assert necessity[0, 0] == pytest.approx(0.3)


def test_every_fusion_rule_normalizes_each_valid_output_pixel():
    pi1 = np.array(
        [
            [[1.0, 0.4, 1.0]],
            [[0.2, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    pi2 = np.array(
        [
            [[0.8, 1.0, 0.1]],
            [[1.0, 0.2, 1.0]],
        ],
        dtype=np.float32,
    )

    for fusion in (adaptive_fusion, conjunctive_fusion, disjunctive_fusion):
        result, _ = fusion(pi1, pi2)
        assert_normalized(result)
