"""Tests for Chios SAR sign-diagnostic statistics and shifted masks."""

import numpy as np
import pytest

from examples.chios_sar_diagnostics import (
    _contrast_summary,
    _empirical_upper_tail_p,
    _iter_random_null_placements,
    _null_placement_difference,
    _significant_histogram_modes,
    _translate_mask,
)
from examples.run_chios import _get_breakpoints


def test_contrast_summary_reports_inside_outside_stats_and_pooled_effect():
    result = _contrast_summary(
        np.array([2.0, 4.0]),
        np.array([0.0, 2.0]),
    )

    assert result["inside_mean"] == pytest.approx(3.0)
    assert result["inside_median"] == pytest.approx(3.0)
    assert result["inside_std"] == pytest.approx(np.sqrt(2.0))
    assert result["outside_mean"] == pytest.approx(1.0)
    assert result["outside_median"] == pytest.approx(1.0)
    assert result["outside_std"] == pytest.approx(np.sqrt(2.0))
    assert result["difference"] == pytest.approx(2.0)
    assert result["effect_size"] == pytest.approx(np.sqrt(2.0))


def test_contrast_summary_ignores_nodata_and_rejects_empty_samples():
    result = _contrast_summary(
        np.array([1.0, np.nan, 3.0]),
        np.array([0.0, 2.0]),
    )

    assert result["inside_count"] == 2
    assert result["difference"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="At least one"):
        _contrast_summary(np.array([np.nan]), np.array([1.0]))


def test_empirical_upper_tail_p_uses_finite_sample_correction():
    result = _empirical_upper_tail_p(np.array([0.0, 0.1, 0.2, np.nan]), 0.15)

    assert result == pytest.approx(2.0 / 4.0)


def test_null_contrast_uses_only_remaining_unburned_ems_land():
    values = np.arange(25, dtype=np.float64).reshape(5, 5)
    land = np.ones((5, 5), dtype=bool)
    ems = np.zeros((5, 5), dtype=bool)
    ems[1:4, 1:4] = True
    fire = np.zeros((5, 5), dtype=bool)
    fire[2, 2] = True
    candidate = np.zeros((5, 5), dtype=bool)
    candidate[1, 1] = True

    difference = _null_placement_difference(
        values,
        candidate[1:2, 1:2],
        (slice(1, 2), slice(1, 2)),
        land,
        ems,
        fire,
    )

    expected_inside = values[1, 1]
    expected_outside = np.mean(values[ems & ~fire & ~candidate])
    assert difference == pytest.approx(expected_inside - expected_outside)


def test_random_null_placements_preserve_area_and_respect_ems_land():
    class Geometry:
        @staticmethod
        def GetEnvelope():
            return (0.0, 100.0, 0.0, 100.0)

    fire = np.zeros((9, 9), dtype=bool)
    fire[3:5, 3:5] = True
    ems = np.ones((9, 9), dtype=bool)
    land = np.ones((9, 9), dtype=bool)
    scene = {
        "fire_bbox": (40.0, 40.0, 60.0, 60.0),
        "ems_geometry": Geometry(),
        "fire_mask": fire,
        "ems_mask": ems,
        "land": land,
    }

    placements = list(
        _iter_random_null_placements(scene, requested=10, seed=7, max_attempts=200)
    )

    accepted = [placement for _, _, placement in placements if placement is not None]
    assert len(accepted) == 10
    for column_shift, row_shift, (rows, columns), candidate in accepted:
        assert candidate.sum() == fire.sum()
        assert not np.any(candidate & fire[rows, columns])
        assert np.all(ems[rows, columns][candidate] & land[rows, columns][candidate])
        assert isinstance(column_shift, int)
        assert isinstance(row_shift, int)


def test_random_null_placements_stop_after_attempt_limit():
    class Geometry:
        @staticmethod
        def GetEnvelope():
            return (0.0, 100.0, 0.0, 100.0)

    fire = np.zeros((9, 9), dtype=bool)
    fire[3:5, 3:5] = True
    scene = {
        "fire_bbox": (40.0, 40.0, 60.0, 60.0),
        "ems_geometry": Geometry(),
        "fire_mask": fire,
        "ems_mask": np.zeros_like(fire),
        "land": np.ones_like(fire),
    }

    placements = list(
        _iter_random_null_placements(scene, requested=5, seed=7, max_attempts=2)
    )

    assert len(placements) == 5
    assert all(placement is None and attempts == 2 for _, attempts, placement in placements)


def test_translate_mask_moves_without_wraparound_or_area_change():
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:3, 2:4] = True

    translated = _translate_mask(mask, column_shift=1, row_shift=-1)

    assert int(translated.sum()) == int(mask.sum())
    assert np.all(translated[0:2, 3:5])
    assert not np.any(translated[:, :3])


def test_significant_histogram_modes_distinguishes_one_and_two_modes():
    rng = np.random.default_rng(42)
    unimodal = rng.normal(0.0, 1.0, size=10000)
    bimodal = np.concatenate(
        (rng.normal(-4.0, 0.5, size=5000), rng.normal(4.0, 0.5, size=5000))
    )

    assert _significant_histogram_modes(unimodal) == 1
    assert _significant_histogram_modes(bimodal) == 2


def test_robust_sar_breakpoints_use_median_and_scaled_mad():
    _, points = _get_breakpoints(
        np.array([0.0, 0.1]),
        np.array([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]),
        dnbr_method="usgs",
        sar_method="robust",
        burn_direction="negative",
        bins=256,
    )

    assert points[0] == pytest.approx(1.4826)
    assert points[1] == pytest.approx(2.9652)
    assert np.isinf(points[2]) and np.isinf(points[3])
