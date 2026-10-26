"""Tests for trapezoidal optical and SAR possibility memberships."""

import numpy as np
import pytest

from dst_fusion.core.memberships import (
    DNBR_DEFAULT_BREAKPOINTS,
    SAR_DEFAULT_BREAKPOINTS,
    dnbr_possibility,
    sar_possibility,
    suggest_otsu_breakpoints,
    trapezoid,
)


def test_trapezoid_has_rising_plateau_and_falling_regions():
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    result = trapezoid(values, (0.0, 1.0, 2.0, 3.0))

    np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 1.0, 0.0])
    assert result.dtype == np.float32


def test_right_shoulder_and_nodata():
    result = trapezoid(
        np.array([[0.0, 0.5, 1.0, np.nan]], dtype=np.float32),
        (0.0, 0.5, np.inf, np.inf),
    )

    np.testing.assert_allclose(result[0, :3], [0.0, 1.0, 1.0])
    assert np.isnan(result[0, 3])


def test_trapezoid_rejects_misordered_breakpoints():
    with pytest.raises(ValueError, match="ordered"):
        trapezoid(np.array([0.5]), (0.0, 2.0, 1.0, 3.0))


def test_dnbr_possibility_uses_documented_usgs_derived_defaults():
    values = np.array([[-0.2, 0.1, 0.27, 0.8]], dtype=np.float32)

    result = dnbr_possibility(values)

    assert DNBR_DEFAULT_BREAKPOINTS == (0.10, 0.27, np.inf, np.inf)
    np.testing.assert_allclose(result[0, 0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(result[1, 0], [1.0, 1.0, 0.0, 0.0])
    assert result.shape == (2, 1, 4)


@pytest.mark.parametrize(
    ("direction", "change", "expected_burned"),
    [
        ("negative", -1.5, 1.0),
        ("negative", 1.5, 0.0),
        ("positive", 1.5, 1.0),
        ("positive", -1.5, 0.0),
    ],
)
def test_sar_burn_direction_orients_change(direction, change, expected_burned):
    result = sar_possibility(
        np.array([[change]], dtype=np.float32),
        burn_direction=direction,
    )

    assert SAR_DEFAULT_BREAKPOINTS == (0.5, 1.5, np.inf, np.inf)
    assert result[0, 0, 0] == pytest.approx(expected_burned)
    assert result[1, 0, 0] == pytest.approx(1.0 - expected_burned)


def test_memberships_propagate_nodata_in_both_classes():
    result = dnbr_possibility(np.array([[np.nan, 0.3]], dtype=np.float32))

    assert np.all(np.isnan(result[:, 0, 0]))
    assert np.all(np.isfinite(result[:, 0, 1]))


def test_otsu_suggestion_separates_two_histogram_modes():
    values = np.concatenate((np.full(100, -2.0), np.full(100, 2.0)))

    suggested = suggest_otsu_breakpoints(values, bins=64)

    assert suggested[0] < suggested[1]
    assert suggested[2:] == (np.inf, np.inf)
    assert -2.0 < suggested[1] < 2.0


def test_otsu_ignores_nodata_and_rejects_constant_or_empty_data():
    values = np.array([-2.0, -1.9, np.nan, 1.9, 2.0])

    assert np.isfinite(suggest_otsu_breakpoints(values)[1])
    with pytest.raises(ValueError, match="constant"):
        suggest_otsu_breakpoints(np.ones(5))
    with pytest.raises(ValueError, match="finite"):
        suggest_otsu_breakpoints(np.full(5, np.nan))
