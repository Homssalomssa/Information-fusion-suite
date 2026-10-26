"""Temporal reliability, acquisition-date warnings, and date-raster extension point."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .possibility import discount

DateInput = date | datetime | str
ReliabilityMethod = Literal["min", "product"]


def _to_date(value: DateInput, name: str) -> date:
    """Parse supported acquisition date values as calendar dates."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD).") from error
    raise TypeError(f"{name} must be a date, datetime, or ISO date string.")


def _reliability_array(value: ArrayLike, name: str) -> NDArray[np.float32]:
    """Validate a scalar or 2D reliability value and convert it to float32."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim not in (0, 2):
        raise ValueError(f"{name} must be a scalar or a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinite values.")
    finite = array[np.isfinite(array)]
    if np.any((finite < 0.0) | (finite > 1.0)):
        raise ValueError(f"{name} finite values must be in [0, 1].")
    return array


def temporal_reliability(
    date_a: DateInput,
    date_b: DateInput,
    *,
    tau_days: float,
) -> float:
    """Return ``exp(-abs(date_a - date_b) / tau_days)`` as reliability.

    Inputs may be ``date``, ``datetime`` (calendar date is used), or ISO date
    strings. Tau must be finite and strictly positive.
    """
    if not np.isfinite(tau_days) or tau_days <= 0.0:
        raise ValueError("tau_days must be finite and greater than zero.")
    first = _to_date(date_a, "date_a")
    second = _to_date(date_b, "date_b")
    delta_days = abs((first - second).days)
    return float(np.exp(-delta_days / float(tau_days)))


def combine_reliability(
    sensor_alpha: ArrayLike,
    alpha_time: ArrayLike,
    *,
    method: ReliabilityMethod = "product",
) -> float | NDArray[np.float32]:
    """Combine sensor and temporal reliabilities by minimum or product.

    Inputs are scalar reliabilities or matching ``(H, W)`` rasters in ``[0,1]``.
    NaN is NoData and propagates. Scalar inputs produce a float; if either input
    is a raster, the result is a float32 raster.
    """
    if method not in ("min", "product"):
        raise ValueError("method must be 'min' or 'product'.")
    sensor = _reliability_array(sensor_alpha, "sensor_alpha")
    temporal = _reliability_array(alpha_time, "alpha_time")
    if sensor.ndim == temporal.ndim == 2 and sensor.shape != temporal.shape:
        raise ValueError("sensor_alpha and alpha_time rasters must have the same shape.")
    if sensor.ndim == 2 and temporal.ndim == 2:
        left, right = sensor, temporal
    else:
        left, right = np.broadcast_arrays(sensor, temporal)

    combined = np.minimum(left, right) if method == "min" else left * right
    if combined.ndim == 0:
        return float(combined)
    return combined.astype(np.float32, copy=False)


def discount_for_date_offset(
    pi: ArrayLike,
    sensor_alpha: ArrayLike = 1.0,
    *,
    alpha_time: ArrayLike,
    source_is_earlier: bool,
    is_post_fire_pair: bool = True,
    asymmetric: bool = True,
    asymmetry_exponent: float = 2.0,
    combine_method: ReliabilityMethod = "product",
    source_date: DateInput | None = None,
    fire_end_date: DateInput | None = None,
) -> NDArray[np.float32]:
    """Apply sensor/time reliability, with post-fire asymmetry for earlier data.

    For an earlier source in the post-fire pair, ``alpha_time**k`` is applied
    only to its Burned claim by raising the floor on ``pi(Burned)``:
    ``1 - sensor_alpha * alpha_time**k``. Its Unburned floor uses sensor
    reliability alone, because burned scars persist. If ``source_date`` and
    ``fire_end_date`` are provided, the asymmetry is active only when this
    earlier post-fire source predates fire end; if dates are omitted or the
    source is not before fire end, it receives the uniform temporal discount.
    Supplying only one date is invalid. This asymmetric branch follows the
    specified product formula; otherwise
    ``combine_method`` selects min or product.
    """
    if not np.isfinite(asymmetry_exponent) or asymmetry_exponent < 1.0:
        raise ValueError("asymmetry_exponent must be finite and at least 1.")
    sensor = _reliability_array(sensor_alpha, "sensor_alpha")
    temporal = _reliability_array(alpha_time, "alpha_time")
    if (source_date is None) != (fire_end_date is None):
        raise ValueError("source_date and fire_end_date must be provided together.")
    date_gate = (
        source_date is not None
        and fire_end_date is not None
        and _to_date(source_date, "source_date") < _to_date(fire_end_date, "fire_end_date")
    )
    # Validate compatibility and the configured combination even when the
    # explicitly specified asymmetric branch uses its product formula.
    combined = combine_reliability(sensor, temporal, method=combine_method)
    if not (is_post_fire_pair and asymmetric and source_is_earlier and date_gate):
        return discount(pi, combined)

    # Start with sensor-only discounting so the earlier Unburned class does not
    # retain the stronger uniform time floor from the symmetric path.
    result = discount(pi, sensor)
    valid = np.all(np.isfinite(result), axis=0)
    if sensor.ndim == temporal.ndim == 2 and sensor.shape != temporal.shape:
        raise ValueError("sensor_alpha and alpha_time rasters must have the same shape.")
    sensor_grid, temporal_grid = np.broadcast_arrays(sensor, temporal)
    if temporal_grid.ndim == 2 and temporal_grid.shape != result.shape[1:]:
        raise ValueError("alpha_time raster must have the same shape as pi.")
    burned_reliability = sensor_grid * np.power(temporal_grid, asymmetry_exponent)
    result[0] = np.maximum(result[0], 1.0 - burned_reliability)
    result[1] = np.maximum(result[1], 1.0 - sensor_grid)
    if temporal_grid.ndim == 0:
        if not np.isfinite(temporal_grid):
            valid[:] = False
    else:
        valid &= np.isfinite(temporal_grid)
    result[:, ~valid] = np.nan
    return result


def pre_fire_date_gap_days(s2_pre_date: DateInput, s1_pre_date: DateInput) -> int:
    """Return the absolute number of calendar days between pre-fire acquisitions."""
    s2_date = _to_date(s2_pre_date, "s2_pre_date")
    s1_date = _to_date(s1_pre_date, "s1_pre_date")
    return abs((s2_date - s1_date).days)


def check_temporal_plausibility(
    s2_pre_date: DateInput,
    s2_post_date: DateInput,
    s1_pre_date: DateInput,
    s1_post_date: DateInput,
    *,
    fire_start: DateInput | None = None,
    fire_end: DateInput | None = None,
    pre_fire_gap_warning_days: int = 30,
) -> list[str]:
    """Emit and return warnings for date gaps or implausible fire chronology.

    The pre-fire pair only triggers a seasonal-change warning when its date gap
    exceeds the configured threshold. Post-fire asymmetry is not applied here;
    this function only checks dates and reports anomalies.
    """
    if (
        isinstance(pre_fire_gap_warning_days, bool)
        or not isinstance(pre_fire_gap_warning_days, (int, np.integer))
        or pre_fire_gap_warning_days < 0
    ):
        raise ValueError("pre_fire_gap_warning_days must be a non-negative integer.")

    s2_pre = _to_date(s2_pre_date, "s2_pre_date")
    s2_post = _to_date(s2_post_date, "s2_post_date")
    s1_pre = _to_date(s1_pre_date, "s1_pre_date")
    s1_post = _to_date(s1_post_date, "s1_post_date")
    messages: list[str] = []

    if s2_pre >= s2_post:
        messages.append("Implausible Sentinel-2 order: pre-fire date is not before post-fire date.")
    if s1_pre >= s1_post:
        messages.append("Implausible Sentinel-1 order: pre-fire date is not before post-fire date.")

    gap = abs((s2_pre - s1_pre).days)
    if gap > pre_fire_gap_warning_days:
        messages.append(
            f"Pre-fire acquisitions differ by {gap} days, exceeding the "
            f"{pre_fire_gap_warning_days}-day warning threshold; seasonal change may matter."
        )

    start = _to_date(fire_start, "fire_start") if fire_start is not None else None
    end = _to_date(fire_end, "fire_end") if fire_end is not None else None
    if start is not None:
        if s2_pre >= start or s1_pre >= start:
            messages.append("A pre-fire acquisition is on or after the fire start date.")
        if s2_post < start or s1_post < start:
            messages.append("A post-fire acquisition predates the fire start date.")
    if end is not None:
        if s2_post < end or s1_post < end:
            messages.append("A post-fire acquisition predates the fire end date.")
        if start is not None and end < start:
            messages.append("Implausible fire chronology: fire end date is before fire start date.")

    for message in messages:
        warnings.warn(message, UserWarning, stacklevel=2)
    return messages


def firms_temporal_reliability(
    fire_date_raster: ArrayLike,
    acquisition_date: DateInput,
    *,
    tau_days: float,
) -> NDArray[np.float32]:
    """Reserved extension point for per-pixel FIRMS/fire-progression dates.

    Pixel-wise date decoding, NoData conventions, and temporal reliability
    calculation are intentionally deferred until a real date-raster format is
    selected.
    """
    raise NotImplementedError(
        "FIRMS fire-date-raster temporal reliability is reserved for a future "
        f"milestone (received date={acquisition_date!r}, tau_days={tau_days}, "
        f"raster_shape={np.shape(fire_date_raster)})."
    )
