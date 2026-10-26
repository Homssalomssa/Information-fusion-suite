"""Tests for SCL reliability and nodata-aware linear-power speckle filtering."""

import numpy as np
import pytest

from dst_fusion.core.preprocessing import (
    boxcar_filter,
    center_by_median,
    db_to_linear_power,
    linear_power_to_db,
    robust_standard_deviation,
    scl_optical_reliability,
)


def test_scl_reliability_discounts_cloud_classes_and_marks_zero_nodata():
    scl = np.array([[0, 3, 4, 5, 6, 8, 9, 10, 11]], dtype=np.uint8)

    alpha = scl_optical_reliability(scl)

    np.testing.assert_allclose(alpha[0, 1:], [0, 1, 1, 1, 0, 0, 0, 1])
    assert np.isnan(alpha[0, 0])


def test_scl_reliability_propagates_nan_and_accepts_configured_classes():
    scl = np.array([[4.0, 7.0, np.nan]], dtype=np.float32)

    alpha = scl_optical_reliability(scl, unreliable_classes=(4, 7))

    np.testing.assert_allclose(alpha[0, :2], [0.0, 0.0])
    assert np.isnan(alpha[0, 2])


@pytest.mark.parametrize(
    ("scl", "unreliable"),
    [
        (np.array([[4.5]]), (3,)),
        (np.array([[12]]), (3,)),
        (np.array([[-1]]), (3,)),
    ],
)
def test_scl_reliability_rejects_invalid_values(scl, unreliable):
    with pytest.raises(ValueError):
        scl_optical_reliability(scl, unreliable_classes=unreliable)


def test_boxcar_filter_averages_linear_power_and_ignores_nodata():
    power = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, np.nan, 9.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    filtered = boxcar_filter(power, window_size=3)

    assert filtered[1, 1] == pytest.approx(2.0)
    assert np.all(np.isfinite(filtered))
    assert filtered.dtype == np.float32


def test_boxcar_filter_validates_size_and_array():
    with pytest.raises(ValueError, match="odd"):
        boxcar_filter(np.ones((2, 2)), window_size=2)
    with pytest.raises(ValueError, match="two-dimensional"):
        boxcar_filter(np.ones(3), window_size=3)
    with pytest.raises(ValueError, match="positive"):
        boxcar_filter(np.ones((2, 2)), window_size=0)


def test_sar_conversion_averages_in_linear_power_and_propagates_nodata():
    db = np.array([[-10.0, 0.0, np.nan]], dtype=np.float32)

    power = db_to_linear_power(db)
    round_trip = linear_power_to_db(power)

    np.testing.assert_allclose(power[0, :2], [0.1, 1.0])
    np.testing.assert_allclose(round_trip[0, :2], [-10.0, 0.0])
    assert np.isnan(power[0, 2])
    assert np.isnan(round_trip[0, 2])


def test_linear_power_to_db_rejects_nonpositive_power():
    with pytest.raises(ValueError, match="positive"):
        linear_power_to_db(np.array([[0.0]], dtype=np.float32))


def test_center_by_median_uses_only_land_samples_and_preserves_nodata():
    change = np.array([[1.0, 2.0, 3.0, 100.0, np.nan]], dtype=np.float32)
    land = np.array([[True, True, True, False, True]])

    centered, offset = center_by_median(change, land)

    assert offset == pytest.approx(2.0)
    np.testing.assert_allclose(centered[0, :4], [-1.0, 0.0, 1.0, 98.0])
    assert np.isnan(centered[0, 4])


def test_robust_standard_deviation_is_scaled_median_absolute_deviation():
    values = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    result = robust_standard_deviation(values)

    assert result == pytest.approx(1.4826)


def test_median_centering_and_robust_scale_require_valid_land_samples():
    values = np.array([[np.nan, 2.0]], dtype=np.float32)
    empty = np.array([[False, False]])

    with pytest.raises(ValueError, match="At least one"):
        center_by_median(values, empty)
    with pytest.raises(ValueError, match="At least one"):
        robust_standard_deviation(values, empty)
