"""Tests for Chios SAR sign-diagnostic statistics and shifted masks."""

import numpy as np
import pytest

from examples.chios_sar_diagnostics import (
    _contrast_summary,
    _empirical_upper_tail_p,
    _significant_histogram_modes,
    _translate_mask,
)


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
