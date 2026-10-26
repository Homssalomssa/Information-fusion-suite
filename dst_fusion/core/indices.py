"""Sentinel-2 NBR and dNBR calculations using scaled surface reflectance."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_bands(
    bands: tuple[ArrayLike, ...],
    *,
    unscaled_threshold: float,
) -> tuple[NDArray[np.float32], ...]:
    """Validate aligned finite-or-NaN reflectance bands and warn on likely scaling."""
    if not np.isfinite(unscaled_threshold) or unscaled_threshold <= 0.0:
        raise ValueError("unscaled_threshold must be a finite positive number.")

    arrays = tuple(np.asarray(band, dtype=np.float32) for band in bands)
    if not arrays:
        raise ValueError("At least one reflectance band is required.")
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Reflectance bands must be two-dimensional arrays.")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("Reflectance bands must have the same shape.")
    if any(np.any(np.isinf(array)) for array in arrays):
        raise ValueError("Reflectance bands must not contain infinite values.")

    if any(
        np.any(np.isfinite(array) & (array > unscaled_threshold))
        for array in arrays
    ):
        warnings.warn(
            "Sentinel-2 reflectance values look unscaled; expected surface "
            "reflectance with /10000 scaling and the BOA offset already applied.",
            UserWarning,
            stacklevel=3,
        )
    return arrays


def _normalized_difference(
    band_a: NDArray[np.float32],
    band_b: NDArray[np.float32],
    denominator_epsilon: float,
) -> NDArray[np.float32]:
    """Calculate a normalized difference, propagating NoData and unsafe sums."""
    if not np.isfinite(denominator_epsilon) or denominator_epsilon < 0.0:
        raise ValueError("denominator_epsilon must be finite and non-negative.")

    valid = np.isfinite(band_a) & np.isfinite(band_b)
    denominator = band_a + band_b
    valid &= np.isfinite(denominator) & (np.abs(denominator) > denominator_epsilon)
    result = np.full(band_a.shape, np.nan, dtype=np.float32)
    np.divide(band_a - band_b, denominator, out=result, where=valid)
    return result


def nbr(
    b8a: ArrayLike,
    b12: ArrayLike,
    *,
    denominator_epsilon: float = 1e-6,
    unscaled_threshold: float = 1.5,
) -> NDArray[np.float32]:
    """Calculate ``(B8A - B12) / (B8A + B12)`` for aligned 20 m bands.

    Bands are expected to contain surface reflectance with Sentinel-2's
    ``/10000`` scaling and processing-baseline BOA offset already applied.
    A warning is emitted if any finite reflectance exceeds the likely-unscaled
    threshold. NoData is NaN; it propagates, as does a near-zero denominator.
    """
    b8a_array, b12_array = _validated_bands(
        (b8a, b12),
        unscaled_threshold=unscaled_threshold,
    )
    return _normalized_difference(b8a_array, b12_array, denominator_epsilon)


def dnbr(
    b8a_pre: ArrayLike,
    b12_pre: ArrayLike,
    b8a_post: ArrayLike,
    b12_post: ArrayLike,
    *,
    denominator_epsilon: float = 1e-6,
    unscaled_threshold: float = 1.5,
) -> NDArray[np.float32]:
    """Calculate ``NBR_pre - NBR_post`` from aligned B8A/B12 reflectance bands.

    Larger positive values indicate greater burn evidence. All four bands use
    scaled surface reflectance; NaN in any band propagates to the dNBR pixel.
    """
    bands = _validated_bands(
        (b8a_pre, b12_pre, b8a_post, b12_post),
        unscaled_threshold=unscaled_threshold,
    )
    pre = _normalized_difference(bands[0], bands[1], denominator_epsilon)
    post = _normalized_difference(bands[2], bands[3], denominator_epsilon)
    result = pre - post
    result[~(np.isfinite(pre) & np.isfinite(post))] = np.nan
    return result.astype(np.float32, copy=False)
