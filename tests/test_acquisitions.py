"""Tests for acquisition metadata detection from Sentinel raster paths."""

from datetime import date

import pytest

from dst_fusion.core.acquisitions import (
    detect_acquisition_date,
    detect_sar_polarization,
)


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        (
            "/data/T35SMC_20240628T090601_B8A_20m.jp2",
            date(2024, 6, 28),
        ),
        (
            "/data/S2A_MSIL2A_20240628T090601_N0510_R050_T35SMC_20240628T120000.SAFE/"
            "GRANULE/G/IMG_DATA/T35SMC_B8A_20m.jp2",
            date(2024, 6, 28),
        ),
        (
            "/data/S1A_IW_20240629T041546_DVP_RTC20_G_gduned_713A_VV.tif",
            date(2024, 6, 29),
        ),
    ],
)
def test_detect_acquisition_date_from_sentinel_paths(source_path, expected):
    assert detect_acquisition_date(source_path) == expected


def test_acquisition_date_detection_fails_without_date():
    assert detect_acquisition_date("/data/image_without_acquisition_date.tif") is None


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        ("/data/S1A_scene_VV.tif", "VV"),
        ("/data/S1A_scene_vh.tif", "VH"),
        ("/data/S1A_scene.tif", None),
    ],
)
def test_detect_sar_polarization_from_filename(source_path, expected):
    assert detect_sar_polarization(source_path) == expected
