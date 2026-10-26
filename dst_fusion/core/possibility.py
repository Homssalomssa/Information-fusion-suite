"""Possibility distributions, reliability discounting, fusion, and decisions.

Arrays use class order ``[Burned, Unburned]`` and shape ``(2, H, W)``.
NaN marks NoData; if either class of a pixel is NaN, that pixel is NoData
for both classes and all derived outputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_CLASS_COUNT = 2
_NORMALIZATION_ATOL = 1e-6
H_EPSILON = 1e-12


def _as_possibility_array(pi: ArrayLike, name: str) -> NDArray[np.float32]:
    """Validate array shape, values, and per-pixel possibility normalization."""
    result = np.asarray(pi, dtype=np.float32)
    if result.ndim != 3 or result.shape[0] != _CLASS_COUNT:
        raise ValueError(f"{name} must have shape (2, H, W) in [Burned, Unburned] order.")
    if np.any(np.isinf(result)):
        raise ValueError(f"{name} must not contain infinite values.")

    finite_values = result[np.isfinite(result)]
    if np.any((finite_values < 0.0) | (finite_values > 1.0)):
        raise ValueError(f"{name} finite values must be in the range [0, 1].")

    valid = np.all(np.isfinite(result), axis=0)
    if np.any(valid):
        maxima = np.max(result[:, valid], axis=0)
        if not np.all(np.isclose(maxima, 1.0, rtol=0.0, atol=_NORMALIZATION_ATOL)):
            raise ValueError(f"{name} must be normalized so each valid pixel has a maximum of 1.")

    result = result.copy()
    result[:, ~valid] = np.nan
    return result


def _as_matching_pair(
    pi1: ArrayLike,
    pi2: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Validate two same-grid possibility distributions and return their valid mask."""
    first = _as_possibility_array(pi1, "pi1")
    second = _as_possibility_array(pi2, "pi2")
    if first.shape != second.shape:
        raise ValueError("pi1 and pi2 must have the same shape.")
    valid = np.all(np.isfinite(first), axis=0) & np.all(np.isfinite(second), axis=0)
    return first, second, valid


def _normalize_valid_pixels(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize each finite pixel by its largest class value."""
    result = values.copy()
    valid = np.all(np.isfinite(result), axis=0)
    if np.any(valid):
        maxima = np.max(result[:, valid], axis=0)
        result[:, valid] /= maxima
    result[:, ~valid] = np.nan
    return result


def normalize_possibility(pi: ArrayLike) -> NDArray[np.float32]:
    """Convert class memberships to a normalized possibility distribution.

    For each pixel, divide both class values by their maximum. A zero maximum
    cannot define a possibility distribution and raises ``ValueError``.
    NaN in either class marks the whole pixel as NoData.
    """
    result = np.asarray(pi, dtype=np.float32)
    if result.ndim != 3 or result.shape[0] != _CLASS_COUNT:
        raise ValueError("pi must have shape (2, H, W) in [Burned, Unburned] order.")
    if np.any(np.isinf(result)):
        raise ValueError("pi must not contain infinite values.")

    finite_values = result[np.isfinite(result)]
    if np.any((finite_values < 0.0) | (finite_values > 1.0)):
        raise ValueError("pi finite values must be in the range [0, 1].")

    valid = np.all(np.isfinite(result), axis=0)
    if np.any(valid):
        maxima = np.max(result[:, valid], axis=0)
        if np.any(maxima <= 0.0):
            raise ValueError("Cannot normalize a pixel whose class values are all zero.")

    return _normalize_valid_pixels(result)


def discount(pi: ArrayLike, alpha: float | ArrayLike) -> NDArray[np.float32]:
    """Discount possibility by reliability using ``max(pi, 1 - alpha)``.

    ``alpha`` is either a scalar or a reliability raster with shape ``(H, W)``,
    where 0 means unusable and 1 means fully reliable. NaN in the distribution
    or reliability marks the output pixel as NoData.
    """
    possibility = _as_possibility_array(pi, "pi")
    reliability = np.asarray(alpha, dtype=np.float32)
    if reliability.ndim not in (0, 2) or (
        reliability.ndim == 2 and reliability.shape != possibility.shape[1:]
    ):
        raise ValueError("alpha must be a scalar or have shape (H, W) matching pi.")
    if np.any(np.isinf(reliability)):
        raise ValueError("alpha must not contain infinite values.")
    finite_reliability = reliability[np.isfinite(reliability)]
    if np.any((finite_reliability < 0.0) | (finite_reliability > 1.0)):
        raise ValueError("alpha finite values must be in the range [0, 1].")

    result = np.maximum(possibility, 1.0 - reliability)
    if reliability.ndim == 2:
        invalid = np.any(~np.isfinite(possibility), axis=0) | ~np.isfinite(reliability)
        result[:, invalid] = np.nan
    else:
        result[:, ~np.all(np.isfinite(possibility), axis=0)] = np.nan
    return result.astype(np.float32, copy=False)


def conjunctive_fusion(
    pi1: ArrayLike,
    pi2: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Fuse by class-wise minimum, normalize, and return conflict ``1 - h``.

    If ``h`` is zero, the minimum has no nonzero value to normalize. In that
    total-conflict case, use the disjunctive result to preserve normalization;
    conflict still reports ``1 - h``.
    """
    first, second, valid = _as_matching_pair(pi1, pi2)
    lower = np.minimum(first, second)
    h = np.max(lower, axis=0)
    conflict = 1.0 - h

    total_conflict = valid & (h <= H_EPSILON)
    result = np.maximum(first, second)
    normalizable = valid & ~total_conflict
    if np.any(normalizable):
        result[:, normalizable] = lower[:, normalizable] / h[normalizable]
    result[:, ~valid] = np.nan
    conflict[~valid] = np.nan
    return result.astype(np.float32, copy=False), conflict.astype(np.float32, copy=False)


def disjunctive_fusion(
    pi1: ArrayLike,
    pi2: ArrayLike,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Fuse by class-wise maximum and return conflict ``1 - h``."""
    first, second, valid = _as_matching_pair(pi1, pi2)
    result = np.maximum(first, second)
    h = np.max(np.minimum(first, second), axis=0)
    conflict = 1.0 - h
    result[:, ~valid] = np.nan
    conflict[~valid] = np.nan
    return result.astype(np.float32, copy=False), conflict.astype(np.float32, copy=False)


def adaptive_fusion(
    pi1: ArrayLike,
    pi2: ArrayLike,
    eps: float = H_EPSILON,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Fuse using the Dubois-Prade adaptive possibility rule.

    Per class, compute
    ``max(min(pi1, pi2) / h, min(max(pi1, pi2), 1 - h))``, where
    ``h = max_c min(pi1[c], pi2[c])``. When ``h <= eps``, omit the undefined
    division and use the disjunctive limit. Return conflict ``1 - h``.
    """
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be a finite non-negative number.")

    first, second, valid = _as_matching_pair(pi1, pi2)
    lower = np.minimum(first, second)
    upper = np.maximum(first, second)
    h = np.max(lower, axis=0)

    safe_h = np.where(h > eps, h, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_conjunction = lower / safe_h
    adaptive_result = np.maximum(
        normalized_conjunction,
        np.minimum(upper, 1.0 - h),
    )
    result = np.where((h <= eps)[np.newaxis, :, :], upper, adaptive_result)
    conflict = 1.0 - h
    result[:, ~valid] = np.nan
    conflict[~valid] = np.nan
    return result.astype(np.float32, copy=False), conflict.astype(np.float32, copy=False)


def decide(
    pi: ArrayLike,
    necessity_threshold: float = 0.0,
) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    """Choose a supported class or return undecided, with its necessity.

    Class indices are 0 for Burned and 1 for Unburned. Ties select Burned.
    A tie or necessity below ``necessity_threshold`` returns class 2
    (undecided). Necessity is ``1 - runner_up_possibility``. NoData pixels
    return class -1 and NaN necessity.
    """
    if not np.isfinite(necessity_threshold) or not 0.0 <= necessity_threshold <= 1.0:
        raise ValueError("necessity_threshold must be a finite value in [0, 1].")

    possibility = _as_possibility_array(pi, "pi")
    valid = np.all(np.isfinite(possibility), axis=0)
    classes = np.argmax(possibility, axis=0).astype(np.int8)
    runner_up = np.min(possibility, axis=0)
    necessity = 1.0 - runner_up
    tied = possibility[0] == possibility[1]
    undecided = valid & (tied | (necessity < necessity_threshold))
    classes[undecided] = 2
    classes[~valid] = -1
    necessity[~valid] = np.nan
    return classes, necessity.astype(np.float32, copy=False)
