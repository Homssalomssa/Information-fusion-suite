# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests for Chios SAR sign-diagnostic statistics and shifted masks."""

import numpy as np
import pytest

from examples.chios_sar_diagnostics import (
    _contrast_summary,
    _empirical_upper_tail_p,
    _empirical_two_sided_p,
    _filtered_sar_change,
    _iter_random_null_placements,
    _null_difference_from_sums,
    _placement_is_accepted,
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


def test_empirical_two_sided_p_uses_corrected_lower_and_upper_tails():
    result = _empirical_two_sided_p(np.array([0.0, 1.0, 2.0, 3.0]), 3.0)

    assert result == pytest.approx(0.8)


def test_null_contrast_subtracts_placement_from_remaining_background():
    difference = _null_difference_from_sums(
        placed_sum=10.0,
        placed_count=1,
        background_sum=12.0,
        background_count=3,
    )

    assert difference == pytest.approx(9.0)


def _placement_scene(*, placeable=True):
    shape = (9, 9)
    class_mix = np.tile(np.array([[4, 5], [4, 5]], dtype=np.uint8), (5, 5))[:9, :9]
    layers = {
        "placeable": np.full(shape, placeable, dtype=bool),
        "land": np.ones(shape, dtype=bool),
        "coast_distance": np.full(shape, 600.0),
        "fire_exclusion": np.zeros(shape, dtype=bool),
        "grading_exclusion": np.zeros(shape, dtype=bool),
        "dnbr_values": np.full(shape, 0.05),
        "optical_alpha": np.ones(shape, dtype=np.float32),
        "pre_scl": class_mix,
        "layover_shadow": np.zeros(shape, dtype=bool),
    }

    def read_layer(name, window):
        return layers[name][window]

    return {
        "width": shape[1],
        "height": shape[0],
        "template_mask": np.ones((2, 2), dtype=bool),
        "template_x": 4,
        "template_y": 4,
        "reference_scl_mix": (0.5, 0.5),
        "read_layer": read_layer,
    }


def test_random_null_placements_preserve_area_on_the_full_grid():
    placements = list(
        _iter_random_null_placements(
            _placement_scene(), requested=10, seed=7, max_attempts=200
        )
    )

    accepted = [placement for _, _, placement in placements if placement is not None]
    assert len(accepted) == 10
    for column_shift, row_shift, window, candidate in accepted:
        assert candidate.sum() == 4
        assert isinstance(window[0], slice)
        assert isinstance(column_shift, int)
        assert isinstance(row_shift, int)


def test_random_null_placements_stop_after_attempt_limit():
    placements = list(
        _iter_random_null_placements(
            _placement_scene(placeable=False),
            requested=5,
            seed=7,
            max_attempts=2,
        )
    )

    assert len(placements) == 5
    assert all(placement is None and attempts == 2 for _, attempts, placement in placements)


def test_halo_filtered_sar_blocks_match_full_raster_filter():
    rng = np.random.default_rng(16)
    pre = rng.uniform(0.01, 0.8, size=(11, 13)).astype(np.float32)
    post = rng.uniform(0.01, 0.8, size=(11, 13)).astype(np.float32)
    pre[4, 5] = np.nan
    post[6, 8] = np.nan
    whole = _filtered_sar_change(
        pre,
        post,
        (slice(None), slice(None)),
    )

    for y0, x0, height, width in ((0, 0, 4, 5), (4, 5, 4, 5)):
        ext_y0, ext_x0 = max(0, y0 - 2), max(0, x0 - 2)
        ext_y1, ext_x1 = min(pre.shape[0], y0 + height + 2), min(
            pre.shape[1], x0 + width + 2
        )
        core = (
            slice(y0 - ext_y0, y0 - ext_y0 + height),
            slice(x0 - ext_x0, x0 - ext_x0 + width),
        )
        actual = _filtered_sar_change(
            pre[ext_y0:ext_y1, ext_x0:ext_x1],
            post[ext_y0:ext_y1, ext_x0:ext_x1],
            core,
        )
        np.testing.assert_allclose(
            actual,
            whole[y0 : y0 + height, x0 : x0 + width],
            equal_nan=True,
        )


def _clean_placement_inputs():
    candidate = np.ones((4, 4), dtype=bool)
    pre_scl = np.array(
        [
            [4, 4, 5, 5],
            [4, 4, 5, 5],
            [4, 4, 5, 5],
            [4, 4, 5, 5],
        ],
        dtype=np.uint8,
    )
    shape = candidate.shape
    inputs = {
        "land": np.ones(shape, dtype=bool),
        "coast_distance": np.full(shape, 600.0),
        "fire_exclusion": np.zeros(shape, dtype=bool),
        "grading_exclusion": np.zeros(shape, dtype=bool),
        "dnbr_values": np.full(shape, 0.05),
        "optical_alpha": np.ones(shape, dtype=np.float32),
        "pre_scl": pre_scl,
        "layover_shadow": np.zeros(shape, dtype=bool),
        "reference_scl_mix": (0.5, 0.5),
    }
    return candidate, inputs


def test_clean_placement_passes_all_acceptance_constraints():
    candidate, inputs = _clean_placement_inputs()

    assert _placement_is_accepted(candidate, **inputs)


def test_placement_overlapping_fire_exclusion_is_rejected():
    candidate, inputs = _clean_placement_inputs()
    inputs["fire_exclusion"][1, 1] = True

    assert not _placement_is_accepted(candidate, **inputs)


@pytest.mark.parametrize(
    "input_name, index, value",
    [
        ("land", (0, 0), False),
        ("coast_distance", (0, 0), 499.9),
        ("grading_exclusion", (1, 1), True),
        ("dnbr_values", (0, 0), 0.11),
        ("optical_alpha", (0, 0), 0.0),
        ("layover_shadow", (0, 0), True),
    ],
)
def test_placement_rejects_invalid_land_evidence_and_scl_inputs(
    input_name, index, value
):
    candidate, inputs = _clean_placement_inputs()
    inputs[input_name][index] = value

    assert not _placement_is_accepted(candidate, **inputs)


def test_placement_rejects_mismatched_prefire_scl_class_mix():
    candidate, inputs = _clean_placement_inputs()
    inputs["pre_scl"][:] = 5

    assert not _placement_is_accepted(candidate, **inputs)


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
