# SPDX-License-Identifier: GPL-2.0-or-later

"""Small pure-NumPy preprocessing helpers for Sentinel input rasters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def scl_optical_reliability(
    scl: ArrayLike,
    *,
    unreliable_classes: tuple[int, ...] = (3, 8, 9, 10),
) -> NDArray[np.float32]:
    """Map SCL classes to optical reliability, preserving class-0/NaN NoData.

    Pixels in ``unreliable_classes`` receive alpha 0; all other valid SCL
    classes receive alpha 1. Water validity is a separate land-mask decision.
    """
    values = np.asarray(scl, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("scl must be a two-dimensional raster.")
    if np.any(np.isinf(values)):
        raise ValueError("scl must not contain infinite values.")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 11)):
        raise ValueError("finite SCL classes must be in the range 0 through 11.")
    if np.any(values[finite] != np.floor(values[finite])):
        raise ValueError("finite SCL classes must be integers.")
    if any(
        isinstance(code, bool) or not isinstance(code, (int, np.integer)) or not 0 <= code <= 11
        for code in unreliable_classes
    ):
        raise ValueError("unreliable_classes must contain SCL integer codes from 0 through 11.")

    result = np.ones(values.shape, dtype=np.float32)
    result[~finite | (values == 0)] = np.nan
    result[np.isin(values, unreliable_classes)] = 0.0
    return result


def boxcar_filter(
    values: ArrayLike,
    *,
    window_size: int = 5,
) -> NDArray[np.float32]:
    """Apply a nodata-aware local mean to a 2D raster, typically linear SAR power.

    Edges use the available in-bounds neighbors. NaNs are ignored, and output
    is NaN where the window has no finite samples. The function itself is unit
    agnostic; SAR callers convert dB to linear power before filtering.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, (int, np.integer))
        or window_size < 1
    ):
        raise ValueError("window_size must be a positive odd integer.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    radius = window_size // 2
    valid = np.isfinite(array)
    values_sum = np.where(valid, array, 0.0).astype(np.float64)
    counts = valid.astype(np.float64)
    padded_sum = np.pad(values_sum, radius, mode="constant")
    padded_counts = np.pad(counts, radius, mode="constant")

    def summed_area(data: np.ndarray) -> np.ndarray:
        integral = np.pad(data, ((1, 0), (1, 0)), mode="constant")
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        return (
            integral[window_size:, window_size:]
            - integral[:-window_size, window_size:]
            - integral[window_size:, :-window_size]
            + integral[:-window_size, :-window_size]
        )

    total = summed_area(padded_sum)
    count = summed_area(padded_counts)
    result = np.full(array.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=result, where=count > 0)
    return result


def db_to_linear_power(values: ArrayLike) -> NDArray[np.float32]:
    """Convert backscatter dB to linear power, preserving NaN as NoData."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    result = np.full(array.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(array)
    result[valid] = np.power(10.0, array[valid] / 10.0)
    return result


def linear_power_to_db(values: ArrayLike) -> NDArray[np.float32]:
    """Convert positive linear power to dB, preserving NaN as NoData."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    valid = np.isfinite(array)
    if np.any(array[valid] <= 0.0):
        raise ValueError("finite linear power values must be positive.")
    result = np.full(array.shape, np.nan, dtype=np.float32)
    result[valid] = 10.0 * np.log10(array[valid])
    return result


def center_by_median(
    values: ArrayLike,
    reference_mask: ArrayLike | None = None,
) -> tuple[NDArray[np.float32], float]:
    """Subtract a finite-sample median, optionally estimated on selected pixels.

    The median is estimated from finite values where ``reference_mask`` is
    true, then subtracted from every finite pixel. NaNs remain NoData. A mask
    is useful for estimating an AOI-land offset without including water.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    finite = np.isfinite(array)
    if reference_mask is None:
        selected = finite
    else:
        mask = np.asarray(reference_mask)
        if mask.dtype != np.bool_ or mask.shape != array.shape:
            raise ValueError("reference_mask must be a boolean array matching values.")
        selected = finite & mask
    if not np.any(selected):
        raise ValueError("At least one finite value must be selected for median centering.")
    offset = float(np.median(array[selected].astype(np.float64)))
    centered = np.full(array.shape, np.nan, dtype=np.float32)
    centered[finite] = array[finite] - np.float32(offset)
    return centered, offset


def robust_standard_deviation(
    values: ArrayLike,
    reference_mask: ArrayLike | None = None,
) -> float:
    """Estimate spread as ``1.4826 * median(abs(x - median(x)))``.

    NaNs are ignored. When supplied, ``reference_mask`` selects the pixels
    used to estimate the median and median absolute deviation.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional raster.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    finite = np.isfinite(array)
    if reference_mask is None:
        selected = finite
    else:
        mask = np.asarray(reference_mask)
        if mask.dtype != np.bool_ or mask.shape != array.shape:
            raise ValueError("reference_mask must be a boolean array matching values.")
        selected = finite & mask
    if not np.any(selected):
        raise ValueError("At least one finite value must be selected for robust scale estimation.")
    sample = array[selected].astype(np.float64)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return 1.4826 * mad
