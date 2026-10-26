"""Tests for Sentinel-2 NBR and dNBR calculations."""

import numpy as np
import pytest

from dst_fusion.core.indices import dnbr, nbr


def test_nbr_and_dnbr_use_b8a_and_b12():
    b8a_pre = np.array([[0.8, 0.6]], dtype=np.float32)
    b12_pre = np.array([[0.2, 0.3]], dtype=np.float32)
    b8a_post = np.array([[0.4, 0.3]], dtype=np.float32)
    b12_post = np.array([[0.4, 0.3]], dtype=np.float32)

    pre = nbr(b8a_pre, b12_pre)
    result = dnbr(b8a_pre, b12_pre, b8a_post, b12_post)

    np.testing.assert_allclose(pre, [[0.6, 0.33333334]], rtol=1e-6)
    np.testing.assert_allclose(result, [[0.6, 0.33333334]], rtol=1e-6)
    assert result.dtype == np.float32


def test_nbr_denominator_near_zero_becomes_nodata():
    result = nbr(
        np.array([[0.2, 0.0]], dtype=np.float32),
        np.array([[-0.2, 0.0]], dtype=np.float32),
    )

    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])


def test_dnbr_propagates_nodata_from_any_input():
    b8a_pre = np.array([[0.8, np.nan]], dtype=np.float32)
    b12_pre = np.array([[0.2, 0.2]], dtype=np.float32)
    b8a_post = np.array([[0.4, 0.4]], dtype=np.float32)
    b12_post = np.array([[0.4, 0.4]], dtype=np.float32)

    result = dnbr(b8a_pre, b12_pre, b8a_post, b12_post)

    assert np.isfinite(result[0, 0])
    assert np.isnan(result[0, 1])


def test_unscaled_reflectance_emits_warning():
    with pytest.warns(UserWarning, match="look unscaled"):
        nbr(
            np.array([[10000.0]], dtype=np.float32),
            np.array([[2000.0]], dtype=np.float32),
        )


def test_nbr_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        nbr(np.ones((2, 2)), np.ones((2, 3)))


def test_infinite_band_values_are_rejected():
    with pytest.raises(ValueError, match="infinite"):
        nbr(np.array([[np.inf]]), np.ones((1, 1)))
