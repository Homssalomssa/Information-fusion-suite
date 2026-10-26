"""Generated GeoTIFF integration tests for raster alignment and block IO."""

import math

import numpy as np
import pytest

gdal = pytest.importorskip("osgeo.gdal")
osr = pytest.importorskip("osgeo.osr")
gdal.UseExceptions()

from dst_fusion.io.raster import (  # noqa: E402
    align_rasters,
    compute_intersection_grid,
    iter_aligned_blocks,
    iter_raster_blocks,
    resample_sar_db_to_grid,
)


def _spatial_reference(epsg: int):
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(epsg)
    reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return reference


def _create_raster(
    path,
    data: np.ndarray,
    *,
    geotransform: tuple[float, float, float, float, float, float],
    epsg: int = 32631,
    nodata: float = np.nan,
):
    height, width = data.shape
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(_spatial_reference(epsg).ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data.astype(np.float32))
    dataset.FlushCache()
    dataset = None


def test_intersection_uses_s2_reference_grid_across_crs_and_resolutions(tmp_path):
    reference_path = tmp_path / "s2_reference.tif"
    other_path = tmp_path / "other_crs.tif"
    _create_raster(
        reference_path,
        np.ones((10, 10), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    target_srs = _spatial_reference(32631)
    other_srs = _spatial_reference(4326)
    transform = osr.CoordinateTransformation(target_srs, other_srs)
    corners = [
        transform.TransformPoint(500040.0, 840.0),
        transform.TransformPoint(500160.0, 960.0),
    ]
    xmin = min(point[0] for point in corners)
    xmax = max(point[0] for point in corners)
    ymin = min(point[1] for point in corners)
    ymax = max(point[1] for point in corners)
    pixel_size = 0.00002
    width = math.ceil((xmax - xmin) / pixel_size)
    height = math.ceil((ymax - ymin) / pixel_size)
    _create_raster(
        other_path,
        np.ones((height, width), dtype=np.float32),
        geotransform=(xmin, pixel_size, 0.0, ymax, 0.0, -pixel_size),
        epsg=4326,
    )

    grid = compute_intersection_grid(
        {"s2": reference_path, "other": other_path},
        reference_key="s2",
    )

    assert grid.width < 10
    assert grid.height < 10
    assert grid.geotransform[1] == pytest.approx(20.0)
    assert grid.geotransform[5] == pytest.approx(-20.0)
    assert grid.projection_wkt


def test_sar_downsampling_averages_linear_power_not_db(tmp_path):
    reference_path = tmp_path / "s2.tif"
    sar_path = tmp_path / "sar_db.tif"
    output_path = tmp_path / "sar_aligned.tif"
    _create_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )
    sar = np.array(
        [
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [0.0, 10.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    _create_raster(
        sar_path,
        sar,
        geotransform=(500000.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
    )

    grid = compute_intersection_grid(
        {"s2": reference_path, "sar": sar_path},
        reference_key="s2",
    )
    resample_sar_db_to_grid(sar_path, output_path, grid, block_size=1)

    aligned = gdal.Open(str(output_path)).ReadAsArray()
    expected = 10.0 * np.log10(np.mean(np.power(10.0, sar[:2, :2] / 10.0)))
    assert aligned[0, 0] == pytest.approx(expected, abs=1e-4)
    assert aligned[0, 0] != pytest.approx(float(np.mean(sar[:2, :2])))


def test_align_rasters_writes_common_grid_and_resamples_sar(tmp_path):
    reference_path = tmp_path / "s2.tif"
    sar_path = tmp_path / "sar.tif"
    output_dir = tmp_path / "aligned"
    transform = (500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0)
    _create_raster(
        reference_path,
        np.ones((2, 2), dtype=np.float32),
        geotransform=transform,
    )
    _create_raster(
        sar_path,
        np.full((4, 4), 10.0, dtype=np.float32),
        geotransform=(500000.0, 10.0, 0.0, 1000.0, 0.0, -10.0),
    )

    result = align_rasters(
        {"s2": reference_path, "sar": sar_path},
        output_dir,
        reference_key="s2",
        sar_db_keys=("sar",),
        block_size=1,
    )

    assert result.grid.width == result.grid.height == 2
    assert set(result.outputs) == {"s2", "sar"}
    s2 = gdal.Open(str(result.outputs["s2"]))
    sar = gdal.Open(str(result.outputs["sar"]))
    assert s2.GetGeoTransform() == pytest.approx(result.grid.geotransform)
    assert sar.GetGeoTransform() == pytest.approx(result.grid.geotransform)
    np.testing.assert_allclose(sar.ReadAsArray(), 10.0)


def test_aligned_block_reader_propagates_any_layer_nodata_to_all_layers(tmp_path):
    first_path = tmp_path / "first.tif"
    second_path = tmp_path / "second.tif"
    transform = (500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0)
    _create_raster(first_path, np.ones((2, 2), dtype=np.float32), geotransform=transform)
    second = np.full((2, 2), 2.0, dtype=np.float32)
    second[0, 1] = np.nan
    _create_raster(second_path, second, geotransform=transform)

    blocks = list(
        iter_aligned_blocks(
            {"first": first_path, "second": second_path},
            block_size=1,
        )
    )

    nodata_block = next(block for block in blocks if block.x_offset == 1 and block.y_offset == 0)
    assert np.isnan(nodata_block.arrays["first"][0, 0])
    assert np.isnan(nodata_block.arrays["second"][0, 0])


def test_raster_blocks_are_bounded_and_cover_each_pixel_once(tmp_path):
    path = tmp_path / "values.tif"
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    _create_raster(
        path,
        values,
        geotransform=(500000.0, 20.0, 0.0, 1000.0, 0.0, -20.0),
    )

    blocks = list(iter_raster_blocks(path, block_size=3))

    reconstructed = np.full(values.shape, np.nan, dtype=np.float32)
    for block in blocks:
        x_offset, y_offset, x_size, y_size = block.window
        assert x_size <= 3 and y_size <= 3
        reconstructed[y_offset:y_offset + y_size, x_offset:x_offset + x_size] = block.data
    np.testing.assert_array_equal(reconstructed, values)
