"""Trapezoidal possibility memberships for optical and SAR change evidence."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

DNBR_DEFAULT_BREAKPOINTS = (0.10, 0.27, np.inf, np.inf)
"""Right-shoulder rise from unburned/low severity to moderate-low dNBR."""

SAR_DEFAULT_BREAKPOINTS = (0.5, 1.5, np.inf, np.inf)
"""Placeholder rise in oriented SAR change magnitude, in dB; calibrate per scene."""


def _validate_breakpoints(
    breakpoints: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Check ordered trapezoid breakpoints, allowing infinite outer shoulders."""
    if len(breakpoints) != 4:
        raise ValueError("breakpoints must contain exactly four values.")
    points = tuple(float(point) for point in breakpoints)
    if any(np.isnan(point) for point in points):
        raise ValueError("breakpoints must not contain NaN.")
    if points[0] == np.inf or points[-1] == -np.inf:
        raise ValueError("Only the first breakpoint may be -inf and last may be inf.")
    if not points[0] <= points[1] <= points[2] <= points[3]:
        raise ValueError("breakpoints must be ordered non-decreasingly.")
    return points


def trapezoid(
    values: ArrayLike,
    breakpoints: tuple[float, float, float, float],
) -> NDArray[np.float32]:
    """Evaluate a trapezoidal membership function, preserving NaN as NoData.

    ``breakpoints`` are ``(a, b, c, d)``: membership rises from 0 to 1 over
    ``a..b``, stays 1 over ``b..c``, then falls to 0 over ``c..d``. Infinite
    outer points express left- or right-shoulder trapezoids.
    """
    a, b, c, d = _validate_breakpoints(breakpoints)
    array = np.asarray(values, dtype=np.float32)
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")

    result = np.full(array.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(array)
    x = array[finite]
    membership = np.zeros(x.shape, dtype=np.float32)

    rising = (x > a) & (x < b)
    if np.isfinite(a) and np.isfinite(b) and b > a:
        membership[rising] = (x[rising] - a) / (b - a)

    plateau = (x >= b) & (x <= c)
    membership[plateau] = 1.0

    falling = (x > c) & (x < d)
    if np.isfinite(c) and np.isfinite(d) and d > c:
        membership[falling] = (d - x[falling]) / (d - c)

    result[finite] = np.clip(membership, 0.0, 1.0)
    return result


def _possibility_pair(burned: NDArray[np.float32]) -> NDArray[np.float32]:
    """Pair burn membership with its complement, preserving two-class validity."""
    unburned = 1.0 - burned
    result = np.stack((burned, unburned)).astype(np.float32, copy=False)
    invalid = ~np.isfinite(burned)
    result[:, invalid] = np.nan
    return result


def dnbr_possibility(
    dnbr: ArrayLike,
    breakpoints: tuple[float, float, float, float] = DNBR_DEFAULT_BREAKPOINTS,
) -> NDArray[np.float32]:
    """Convert dNBR to ``(Burned, Unburned)`` possibilities.

    The default rising shoulder uses the USGS dNBR ``0.10`` and ``0.27``
    thresholds as the unburned/low-to-moderate transition. The resulting
    burned membership is a right shoulder; the unburned membership is its
    complement. Breakpoints can be supplied to tune the response.
    """
    values = np.asarray(dnbr, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("dnbr must be a two-dimensional raster.")
    return _possibility_pair(trapezoid(values, breakpoints))


def sar_possibility(
    change_db: ArrayLike,
    breakpoints: tuple[float, float, float, float] = SAR_DEFAULT_BREAKPOINTS,
    *,
    burn_direction: str = "negative",
) -> NDArray[np.float32]:
    """Convert SAR post-minus-pre dB change to two-class possibilities.

    ``burn_direction='negative'`` treats a backscatter decrease as burn
    evidence; ``'positive'`` treats an increase as burn evidence. The default
    breakpoints are provisional oriented-change magnitudes in dB and must be
    calibrated for the scene and polarization.
    """
    if burn_direction not in ("negative", "positive"):
        raise ValueError("burn_direction must be 'negative' or 'positive'.")
    values = np.asarray(change_db, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("change_db must be a two-dimensional raster.")
    if np.any(np.isinf(values)):
        raise ValueError("change_db must not contain infinite values.")
    oriented_change = -values if burn_direction == "negative" else values
    return _possibility_pair(trapezoid(oriented_change, breakpoints))


def suggest_otsu_breakpoints(
    values: ArrayLike,
    *,
    bins: int = 256,
) -> tuple[float, float, float, float]:
    """Suggest a rising right-shoulder trapezoid using a histogram Otsu split.

    Otsu's between-class variance selects the split threshold. One histogram
    bin below that split supplies the rising ramp start; membership reaches 1
    at the threshold and remains there. NaN is ignored; empty or constant
    finite input raises ``ValueError``.
    """
    array = np.asarray(values, dtype=np.float64)
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if np.any(np.isinf(array)):
        raise ValueError("values must not contain infinite values.")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Otsu requires at least one finite value.")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum == maximum:
        raise ValueError("Otsu cannot split constant finite values.")

    histogram, edges = np.histogram(finite, bins=bins, range=(minimum, maximum))
    counts = histogram.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative_count = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * centers)
    total_count = cumulative_count[-1]
    total_sum = cumulative_sum[-1]
    denominator = cumulative_count * (total_count - cumulative_count)
    between_class_variance = np.zeros(counts.shape, dtype=np.float64)
    valid = denominator > 0
    between_class_variance[valid] = (
        (total_sum * cumulative_count[valid] - cumulative_sum[valid] * total_count)
        ** 2
        / denominator[valid]
    )
    split_index = int(np.argmax(between_class_variance))
    threshold = float(edges[split_index + 1])
    ramp_start = max(minimum, threshold - float(edges[1] - edges[0]))
    return ramp_start, threshold, np.inf, np.inf
