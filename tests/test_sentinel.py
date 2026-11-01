"""Tests for Sentinel-2 mode and BOA calibration discovery."""

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal")
gdal.UseExceptions()

from possibility_fusion.io.sentinel import infer_s2_input_mode, resolve_boa_parameters

_INPUT_KEYS = ("b8a_pre", "b12_pre", "b8a_post", "b12_post")


def _create_inputs(directory, *, dtype, values, names=None):
    directory.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = {key: f"{key}.tif" for key in _INPUT_KEYS}
    paths = {}
    data_type = {
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("float32"): gdal.GDT_Float32,
    }
    for key in _INPUT_KEYS:
        path = directory / names[key]
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(path),
            2,
            2,
            1,
            data_type[np.dtype(dtype[key] if isinstance(dtype, dict) else dtype)],
        )
        dataset.GetRasterBand(1).WriteArray(
            np.asarray(values[key] if isinstance(values, dict) else values, dtype=dtype[key] if isinstance(dtype, dict) else dtype)
        )
        dataset = None
        paths[key] = path
    return paths


def test_auto_s2_mode_selects_integer_inputs_as_dn(tmp_path):
    paths = _create_inputs(
        tmp_path,
        dtype=np.uint16,
        values=np.full((2, 2), 5000),
    )

    inference = infer_s2_input_mode(paths)

    assert inference.mode == "dn"
    assert "UInt16" in inference.reason


def test_auto_s2_mode_selects_float_inputs_through_maximum_1_5_as_reflectance(tmp_path):
    paths = _create_inputs(
        tmp_path,
        dtype=np.float32,
        values=np.array([[0.2, 1.5], [0.0, 0.8]], dtype=np.float32),
    )

    inference = infer_s2_input_mode(paths)

    assert inference.mode == "reflectance"
    assert "max=1.5" in inference.reason


def test_auto_s2_mode_rejects_high_float_maximum_with_explicit_mode_request(tmp_path):
    paths = _create_inputs(
        tmp_path,
        dtype=np.float32,
        values=np.full((2, 2), 2.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="choose S2_INPUT_MODE explicitly"):
        infer_s2_input_mode(paths)


def test_auto_s2_mode_rejects_mixed_input_types(tmp_path):
    paths = _create_inputs(
        tmp_path,
        dtype={
            "b8a_pre": np.uint16,
            "b12_pre": np.float32,
            "b8a_post": np.float32,
            "b12_post": np.float32,
        },
        values={
            "b8a_pre": np.full((2, 2), 5000),
            "b12_pre": np.full((2, 2), 0.2),
            "b8a_post": np.full((2, 2), 0.2),
            "b12_post": np.full((2, 2), 0.2),
        },
    )

    with pytest.raises(ValueError, match="do not share one unambiguous mode"):
        infer_s2_input_mode(paths)


def test_boa_parameters_are_read_from_fake_safe_xml(tmp_path):
    safe = tmp_path / "S2A_MSIL2A_20240628T090601_N0510_T35SMC.SAFE"
    paths = _create_inputs(
        safe,
        dtype=np.uint16,
        values=np.ones((2, 2), dtype=np.uint16),
    )
    (safe / "MTD_MSIL2A.xml").write_text(
        "<root><BOA_QUANTIFICATION_VALUE>12000</BOA_QUANTIFICATION_VALUE>"
        '<BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>'
        '<BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET></root>',
        encoding="utf-8",
    )

    metadata = resolve_boa_parameters(paths)

    assert metadata.offset == -1000.0
    assert metadata.quantification == 12000.0
    assert "MTD_MSIL2A.xml" in metadata.offset_source
    assert not metadata.warnings


@pytest.mark.parametrize(
    ("baseline", "expected_offset"),
    [("N0400", -1000.0), ("N0300", 0.0)],
)
def test_boa_offset_falls_back_to_processing_baseline(
    tmp_path,
    baseline,
    expected_offset,
):
    product = tmp_path / f"S2A_MSIL2A_{baseline}_T35SMC.SAFE"
    paths = _create_inputs(
        product,
        dtype=np.uint16,
        values=np.ones((2, 2), dtype=np.uint16),
    )

    metadata = resolve_boa_parameters(paths)

    assert metadata.offset == expected_offset
    assert metadata.quantification == 10000.0
    assert "processing baseline" in metadata.offset_source
    assert metadata.warnings


def test_boa_metadata_without_xml_or_baseline_requires_explicit_values(tmp_path):
    paths = _create_inputs(
        tmp_path / "unknown",
        dtype=np.uint16,
        values=np.ones((2, 2), dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="No MTD_MSIL2A.xml or processing baseline"):
        resolve_boa_parameters(paths)
