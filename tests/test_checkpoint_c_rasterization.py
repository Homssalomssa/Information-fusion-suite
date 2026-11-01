# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
from osgeo import ogr, osr

from examples.audit_chios import _rasterize_geometry
from examples.checkpoint_c_supporting import _select_cloud_blocks
from examples.score_checkpoint_c import _rasterize_difference


def test_reference_rasterization_uses_pixel_center_rule():
    geometry = ogr.CreateGeometryFromWkt(
        "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0))"
    )
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(32635)

    mask = _rasterize_geometry(
        geometry,
        spatial_reference,
        (0.0, 20.0, 0.0, 40.0, 0.0, -20.0),
        2,
        2,
    )

    np.testing.assert_array_equal(mask, [[False, False], [True, False]])


def test_reference_difference_is_applied_before_pixel_center_rasterization():
    fire = ogr.CreateGeometryFromWkt(
        "POLYGON ((0 0, 40 0, 40 40, 0 40, 0 0))"
    )
    excluded = ogr.CreateGeometryFromWkt(
        "POLYGON ((0 0, 20 0, 20 40, 0 40, 0 0))"
    )
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(32635)

    mask = _rasterize_difference(
        [fire],
        [excluded],
        spatial_reference,
        (0.0, 20.0, 0.0, 40.0, 0.0, -20.0),
        2,
        2,
    )

    np.testing.assert_array_equal(mask, [[False, True], [False, True]])


def test_synthetic_blocks_sort_by_overlap_then_row_and_column():
    fire = np.zeros((4, 4), dtype=bool)
    fire[:2, :2] = [[1, 1], [1, 0]]
    fire[:2, 2:] = [[1, 1], [1, 0]]
    fire[2:, :2] = True
    fire[2:, 2:] = [[1, 0], [0, 0]]

    selected = _select_cloud_blocks(fire, (2, 2), count=3)

    assert [(row, column) for _, row, column, *_ in selected] == [
        (2, 0),
        (0, 0),
        (0, 2),
    ]
