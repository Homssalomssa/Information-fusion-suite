# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests for temporal reliability, asymmetry, and date plausibility."""

from datetime import date

import numpy as np
import pytest

from possibility_fusion.core.temporal import (
    check_temporal_plausibility,
    combine_reliability,
    discount_for_date_offset,
    firms_temporal_reliability,
    pre_fire_date_gap_days,
    temporal_reliability,
)


def pi_pair(burned: float, unburned: float) -> np.ndarray:
    return np.array([[[burned]], [[unburned]]], dtype=np.float32)


def test_temporal_reliability_matches_exponential_date_gap():
    result = temporal_reliability("2024-08-10", date(2024, 8, 15), tau_days=14)

    assert result == pytest.approx(np.exp(-5 / 14))


@pytest.mark.parametrize("tau_days", [0, -1, np.inf, np.nan])
def test_temporal_reliability_rejects_invalid_tau(tau_days):
    with pytest.raises(ValueError, match="tau_days"):
        temporal_reliability(date(2024, 1, 1), date(2024, 1, 2), tau_days=tau_days)


def test_sensor_and_temporal_reliability_support_product_and_min():
    sensor = np.array([[0.8, 0.4]], dtype=np.float32)

    product = combine_reliability(sensor, 0.5, method="product")
    minimum = combine_reliability(sensor, 0.5, method="min")

    np.testing.assert_allclose(product, [[0.4, 0.2]])
    np.testing.assert_allclose(minimum, [[0.5, 0.4]])


def test_reliability_nodata_propagates():
    result = combine_reliability(
        np.array([[1.0, np.nan]], dtype=np.float32),
        0.5,
    )

    assert result[0, 0] == pytest.approx(0.5)
    assert np.isnan(result[0, 1])


def test_earlier_post_fire_source_uses_burned_class_floor_from_exponent():
    alpha_time = float(np.exp(-5 / 14))

    unchanged_burn = discount_for_date_offset(
        pi_pair(0.9, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
        asymmetry_exponent=2.0,
        source_date="2024-07-02",
        fire_end_date="2024-07-03",
    )
    raised_burn_floor = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
        asymmetry_exponent=2.0,
        source_date="2024-07-02",
        fire_end_date="2024-07-03",
    )

    np.testing.assert_allclose(unchanged_burn[:, 0, 0], [0.9, 1.0])
    np.testing.assert_allclose(
        raised_burn_floor[:, 0, 0],
        [1.0 - alpha_time**2, 1.0],
    )


def test_later_post_fire_source_gets_uniform_temporal_discount():
    alpha_time = float(np.exp(-5 / 14))

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=False,
    )

    np.testing.assert_allclose(
        result[:, 0, 0],
        [1.0 - alpha_time, 1.0],
    )


def test_pre_fire_pair_does_not_apply_temporal_asymmetry():
    alpha_time = float(np.exp(-5 / 14))

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
        is_post_fire_pair=False,
    )

    np.testing.assert_allclose(
        result[:, 0, 0],
        [1.0 - alpha_time, 1.0],
    )


def test_asymmetry_is_gated_off_when_earlier_post_date_is_after_fire_end():
    alpha_time = float(np.exp(-3 / 14))

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
        source_date="2024-07-08",
        fire_end_date="2024-07-03",
    )

    np.testing.assert_allclose(result[:, 0, 0], [1.0 - alpha_time, 1.0])


def test_asymmetry_remains_enabled_when_earlier_post_date_is_before_fire_end():
    alpha_time = float(np.exp(-3 / 14))

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
        source_date="2024-07-02",
        fire_end_date="2024-07-03",
        asymmetry_exponent=2.0,
    )

    np.testing.assert_allclose(
        result[:, 0, 0],
        [1.0 - alpha_time**2, 1.0],
    )


def test_asymmetry_date_gate_requires_both_dates():
    with pytest.raises(ValueError, match="source_date and fire_end_date"):
        discount_for_date_offset(
            pi_pair(0.1, 1.0),
            alpha_time=0.5,
            source_is_earlier=True,
            source_date="2024-07-02",
        )


def test_earlier_source_without_fire_dates_gets_uniform_discount():
    alpha_time = 0.7

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=alpha_time,
        source_is_earlier=True,
    )

    np.testing.assert_allclose(result[:, 0, 0], [1.0 - alpha_time, 1.0])


def test_temporal_discount_switch_off_preserves_sensor_discount_only():
    pi = pi_pair(0.1, 1.0)
    result = discount_for_date_offset(
        pi,
        sensor_alpha=0.8,
        alpha_time=float(np.exp(-3 / 14)),
        source_is_earlier=True,
        source_date="2024-07-08",
        fire_end_date="2024-07-03",
        temporal_enabled=False,
    )

    np.testing.assert_allclose(result[:, 0, 0], [0.2, 1.0])


def test_temporal_discount_switch_off_does_not_activate_asymmetry():
    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        alpha_time=float(np.exp(-3 / 14)),
        source_is_earlier=True,
        source_date="2024-07-02",
        fire_end_date="2024-07-03",
        asymmetry_exponent=2.0,
        temporal_enabled=False,
    )

    np.testing.assert_allclose(result[:, 0, 0], [0.1, 1.0])


def test_zero_date_offset_has_no_temporal_discount():
    result = discount_for_date_offset(
        pi_pair(0.2, 1.0),
        alpha_time=1.0,
        source_is_earlier=False,
    )

    np.testing.assert_allclose(result[:, 0, 0], [0.2, 1.0])


def test_sensor_reliability_is_combined_and_earlier_burn_is_temporally_discounted():
    alpha_time = 0.7

    result = discount_for_date_offset(
        pi_pair(0.1, 1.0),
        sensor_alpha=0.8,
        alpha_time=alpha_time,
        source_is_earlier=True,
        asymmetry_exponent=2.0,
        source_date="2024-07-02",
        fire_end_date="2024-07-03",
    )

    np.testing.assert_allclose(
        result[:, 0, 0],
        [1.0 - 0.8 * alpha_time**2, 1.0],
    )


@pytest.mark.parametrize("exponent", [0.99, 0.0, -1.0, np.inf, np.nan])
def test_asymmetry_exponent_must_be_finite_and_at_least_one(exponent):
    with pytest.raises(ValueError, match="asymmetry_exponent"):
        discount_for_date_offset(
            pi_pair(0.2, 1.0),
            alpha_time=0.5,
            source_is_earlier=True,
            asymmetry_exponent=exponent,
        )


def test_temporal_discount_propagates_nodata():
    pi = np.array([[[0.2, np.nan]], [[1.0, 1.0]]], dtype=np.float32)
    sensor = np.array([[1.0, 1.0]], dtype=np.float32)

    result = discount_for_date_offset(
        pi,
        sensor_alpha=sensor,
        alpha_time=0.5,
        source_is_earlier=True,
    )

    assert np.all(np.isnan(result[:, 0, 1]))


def test_pre_fire_gap_is_reported_above_configured_threshold():
    assert pre_fire_date_gap_days(date(2024, 1, 1), date(2024, 1, 20)) == 19

    with pytest.warns(UserWarning, match="Pre-fire acquisitions differ by 19 days"):
        warnings = check_temporal_plausibility(
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 1, 20),
            date(2024, 2, 4),
            pre_fire_gap_warning_days=10,
        )
    assert any("19 days" in warning for warning in warnings)


def test_temporal_plausibility_warns_for_implausible_fire_order():
    with pytest.warns(UserWarning) as captured:
        warnings = check_temporal_plausibility(
            date(2024, 1, 10),
            date(2024, 1, 9),
            date(2024, 1, 15),
            date(2024, 1, 8),
            fire_start=date(2024, 1, 12),
            fire_end=date(2024, 1, 20),
        )

    messages = [str(item.message) for item in captured]
    assert warnings
    assert any("post-fire" in message and "fire end" in message for message in messages)
    assert any("pre-fire" in message and "fire start" in message for message in messages)


def test_firms_date_raster_is_an_explicit_unimplemented_stub():
    with pytest.raises(NotImplementedError, match="FIRMS"):
        firms_temporal_reliability(
            np.zeros((2, 2), dtype=np.int32),
            date(2024, 1, 1),
            tau_days=14,
        )
