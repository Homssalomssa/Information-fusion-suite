# SPDX-License-Identifier: GPL-2.0-or-later

"""Regression tests for Chios Sentinel-2 reflectance scaling."""

import numpy as np
import pytest

pytest.importorskip("osgeo.gdal")
from osgeo import ogr, osr

from examples.audit_chios import (  # noqa: E402
    _derived_s1_orbit,
    _read_boa_metadata,
    _rasterize_geometry,
    _scale_reflectance,
)


def test_reflectance_uses_signed_add_offset_and_quantification():
    dn = np.array([[0, 500, 11000]], dtype=np.uint16)

    reflectance = _scale_reflectance(
        dn,
        boa_add_offset=-1000,
        boa_quantification_value=10000,
    )

    assert np.isnan(reflectance[0, 0])
    assert reflectance[0, 1] == pytest.approx(-0.05)
    assert reflectance[0, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("absolute_orbit", ["054533", "054708"])
def test_relative_orbit_is_derived_from_absolute_orbit(absolute_orbit):
    relative, direction, reported_absolute = _derived_s1_orbit(
        ("not supplied", "not supplied", absolute_orbit),
    )

    assert relative == "36 (derived from absolute orbit)"
    assert direction == "Descending (derived from ~04:15 UTC)"
    assert reported_absolute == absolute_orbit


def test_boa_offset_and_quantification_are_read_from_product_xml(tmp_path):
    safe = tmp_path / "S2A_test.SAFE"
    safe.mkdir()
    (safe / "MTD_MSIL2A.xml").write_text(
        "<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>"
        '<BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>'
        '<BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET></root>',
        encoding="utf-8",
    )
    s2 = {
        f"{period}_{band}": safe / f"{period}_{band}_20m.jp2"
        for period in ("pre", "post")
        for band in ("B8A", "B12")
    }

    metadata = _read_boa_metadata(s2)

    assert metadata["pre"]["B8A"] == (-1000, 10000)
    assert metadata["pre"]["B12"] == (-1000, 10000)
    assert metadata["post"] == metadata["pre"]


def test_delineation_rasterization_uses_pixel_centers():
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(32635)
    geometry = ogr.CreateGeometryFromWkt(
        "POLYGON ((20 80, 60 80, 60 40, 20 40, 20 80))"
    )

    mask = _rasterize_geometry(
        geometry,
        spatial_reference,
        (0.0, 20.0, 0.0, 100.0, 0.0, -20.0),
        4,
        4,
    )

    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(mask, expected)
